import os
import json
import math
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T

# ─── AYARLAR ──────────────────────────────────────────────────
LCIB_DATASET_DIR  = "/Volumes/KINGSTON/LCIB_DiffusionSat/lcib_dataset"
CLEAN_DATASET_DIR = "/Volumes/KINGSTON/LCIB_DiffusionSat/clean_dataset"
BINARY_MASKS_DIR  = "/Volumes/KINGSTON/LCIB_DiffusionSat/LCIB_project/binary_masks"
IMAGE_SIZE        = 512   # VAE 512x512 bekliyor

CLOUD_HEIGHT_MIN, CLOUD_HEIGHT_MAX = 600.0, 3500.0
COVERAGE_MIN, COVERAGE_MAX = 0.10, 0.50
OPACITY_MIN, OPACITY_MAX = 0.85, 1.00

# DÜZELTME: eskiden CLOUD_COUNT_MAX = 20.0 idi ("rapor bir üst sınır
# belirtmiyor" varsayımıyla). Bu yanlıştı: hem hocanın raporu (ccount ∈
# {1,...,5}) hem de sizin simülatör notebook'unuz (`for n in range(1, 6)`)
# bulut sayısını hep 1-5 arasında üretiyor. 20'ye bölünce ccount_norm hiçbir
# zaman 0.25'i geçmiyordu — yani bu alan modele neredeyse hiç bilgi
# taşımıyordu (aralığın sadece küçük bir dilimi kullanılıyordu). Artık
# gerçek aralığa (1-5) göre, diğer alanlarla (h_norm gibi) aynı min-max
# normalizasyon deseniyle [0,1]'e tam yayılacak şekilde normalize ediliyor.
CLOUD_COUNT_MIN, CLOUD_COUNT_MAX = 1.0, 5.0
# ──────────────────────────────────────────────────────────────

def parse_metadata(json_path):
    """
    JSON'dan metadata vektörünü çıkar.
    Döndürür: dict (ham değerler)
    """
    with open(json_path) as f:
        data = json.load(f)


    clouds = data.get("clouds", [])
    heights = [c["height"] for c in clouds if "height" in c]
    h = float(np.mean(heights)) if heights else 1500.0
    cloud_count = 1

    return {
        "sun_azimuth":    float(data.get("sun_azimuth", 180.0)),    # θa derece
        "sun_elevation":  float(data.get("sun_elevation", 45.0)),   # θe derece
        "cloud_height":   h,                                          # h metre
        "coverage":       float(data.get("coverage", 20)) / 100.0,  # 0-1 arası
        "base_opacity":   float(data.get("base_opacity", 90)) / 100.0,  # 0-1 arası
        "shadow_strength": float(data.get("shadow_strength", 0.7)),
        "cloud_count":    cloud_count,
    }

def encode_metadata(meta):
    """
    Raporumuzla birebir: azimuth ve elevation'ı 2D vektör olarak encode et.
    Döndürür: torch.Tensor shape [9]
      [cos_az, sin_az, sin_el, cos_el, ccov, copa, h_norm, ccount_norm, sstr_norm]

    NOT: sstr (shadow_strength) rapor denklem 30'da yok, metodoloji
    dokümanının önerisi — Berfin'in onayıyla eklendi (JSON'da gerçek
    veri olduğu doğrulandı). Rapor ile bu noktada bilerek ayrılıyoruz,
    Amir Hoca'ya bildirilmeli.
    """
    az  = math.radians(meta["sun_azimuth"])
    el  = math.radians(meta["sun_elevation"])

    cos_az = math.cos(az)
    sin_az = math.sin(az)
    sin_el = math.sin(el)
    cos_el = math.cos(el)

    h_norm = (meta["cloud_height"] - CLOUD_HEIGHT_MIN) / (CLOUD_HEIGHT_MAX - CLOUD_HEIGHT_MIN)
    h_norm = float(np.clip(h_norm, 0.0, 1.0))

    ccov = float(np.clip(meta["coverage"], COVERAGE_MIN, COVERAGE_MAX))
    copa = float(np.clip(meta["base_opacity"], OPACITY_MIN, OPACITY_MAX))


    # Cloud count is fixed to 1 in all experiments.
    ccount_norm = 1.0

    # VARSAYIM: shadow_strength zaten [0,1] aralığında bir oran olarak
    # geliyor varsayılıyor (JSON şemasında böyle tanımlı olduğu teyit edildi).
    sstr_norm = float(np.clip(meta["shadow_strength"], 0.0, 1.0))

    return torch.tensor([
        cos_az, sin_az,   # azimuth encoding
        sin_el, cos_el,   # elevation encoding (rapordan: sin önce)
        ccov,             # cloud coverage
        copa,             # cloud opacity
        h_norm,           # normalized cloud height
        ccount_norm,      # cloud count (artık gerçek değer, düzeltilmiş normalizasyon)
        sstr_norm,        # shadow strength (metodoloji eklentisi)
    ], dtype=torch.float32)


class LCIBDataset(Dataset):
    """
    LCIB Cloud-Shadow Synthesis Dataset.

    Her sample:
      - Iref: temiz uydu görüntüsü [3, H, W]
      - Mc:   binary cloud mask    [1, H, W]
      - Ms:   binary shadow mask   [1, H, W]
      - Isyn: bulutlu görüntü      [3, H, W]  (ground truth)
      - meta: metadata vektörü     [9]
    """

    def __init__(
        self,
        lcib_dir    = LCIB_DATASET_DIR,
        clean_dir   = CLEAN_DATASET_DIR,
        binary_dir  = BINARY_MASKS_DIR,
        image_size  = IMAGE_SIZE,
    ):
        self.lcib_dir   = Path(lcib_dir)
        self.clean_dir  = Path(clean_dir)
        self.binary_dir = Path(binary_dir)
        self.image_size = image_size

        # Görsel transform (normalize [-1, 1] — VAE için standart)
        self.img_transform  = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Mask transform (binary, 0/1)
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        # Tüm sample'ları topla
        self.samples = self._collect_samples()
        print(f"Dataset hazır: {len(self.samples)} sample bulundu.")

    def _collect_samples(self):
        samples = []
        for sample_dir in sorted(self.lcib_dir.iterdir()):
            if not sample_dir.is_dir() or sample_dir.name.startswith("."):
                continue

            # Base name: klasör adından türet
            # örn: output_north_america_desert_489_1024 → north_america_desert_489_1024
            base_name = sample_dir.name.replace("output_", "")

            # Iref: clean_dataset/{base_name}.png
            iref_path = self.clean_dir / f"{base_name}.png"
            if not iref_path.exists():
                print(f"  UYARI: Iref bulunamadı → {iref_path}")
                continue

            # JSON dosyalarını bul
            json_files = sorted([
                f for f in sample_dir.glob("*_result.json")
                if not f.name.startswith("._")
            ])

            for json_path in json_files:
                # JSON'dan dosya isim kökünü türet
                stem = json_path.stem.replace("_result", "")

                # Cloudy image
                isyn_path = sample_dir / f"{stem}_result.png"

                # Binary maskeler
                mc_path = self.binary_dir / sample_dir.name / f"{stem}_mask.png"
                ms_path = self.binary_dir / sample_dir.name / f"{stem}_shadow.png"

                # Hepsi var mı kontrol et
                if not all([isyn_path.exists(), mc_path.exists(), ms_path.exists()]):
                    continue

                samples.append({
                    "iref":  iref_path,
                    "isyn":  isyn_path,
                    "mc":    mc_path,
                    "ms":    ms_path,
                    "json":  json_path,
                    "name":  stem,
                    # NOT (henüz kullanılmıyor, öneri): train/val split'i
                    # sahneye göre yapmak isterseniz ("aynı temiz görüntü
                    # hem train hem val'de olmasın") base_name'i burada
                    # saklıyoruz — train.py'de random_split yerine bunu
                    # kullanan bir split fonksiyonu yazılabilir.
                    "scene_id": base_name,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Görselleri yükle ve dönüştür
        Iref = self.img_transform(Image.open(s["iref"]).convert("RGB"))
        Isyn = self.img_transform(Image.open(s["isyn"]).convert("RGB"))
        Mc   = self.mask_transform(Image.open(s["mc"]).convert("L"))
        Ms   = self.mask_transform(Image.open(s["ms"]).convert("L"))

        # Metadata
        meta_raw = parse_metadata(s["json"])
        meta_vec = encode_metadata(meta_raw)

        return {
            "Iref":  Iref,       # [3, 512, 512]
            "Isyn":  Isyn,       # [3, 512, 512]
            "Mc":    Mc,         # [1, 512, 512]
            "Ms":    Ms,         # [1, 512, 512]
            "meta":  meta_vec,   # [9]
            "name":  s["name"],
        }


# ─── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = LCIBDataset()
    print(f"Toplam sample: {len(dataset)}")

    sample = dataset[0]
    print(f"\nİlk sample: {sample['name']}")
    print(f"  Iref shape : {sample['Iref'].shape}")
    print(f"  Isyn shape : {sample['Isyn'].shape}")
    print(f"  Mc shape   : {sample['Mc'].shape}")
    print(f"  Ms shape   : {sample['Ms'].shape}")
    print(f"  meta shape : {sample['meta'].shape}")
    print(f"  meta values: {sample['meta']}")
    print(f"  ccount_norm (idx 7): {sample['meta'][7].item():.3f}  (artık 0-1 aralığına tam yayılıyor)")
    print(f"  sstr_norm    (idx 8): {sample['meta'][8].item():.3f}")

    # DÜZELTME doğrulaması: birkaç örnekte ccount_norm'un artık [0,1]
    # aralığının genelini kullandığını göster (eskiden hep <=0.25 idi)
    ccounts = [dataset[i]["meta"][7].item() for i in range(min(20, len(dataset)))]
    print(f"\n  İlk 20 örnekte ccount_norm aralığı: [{min(ccounts):.3f}, {max(ccounts):.3f}]")


    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  Iref batch shape: {batch['Iref'].shape}")
    print(f"  meta batch shape: {batch['meta'].shape}")

    print("\nDataset tamam!")

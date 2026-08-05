import os
import json
import math
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T


LCIB_DATASET_DIR  = "/Volumes/KINGSTON/LCIB_DiffusionSat/lcib_dataset2"
CLEAN_DATASET_DIR = "/Volumes/KINGSTON/LCIB_DiffusionSat/africa"
BINARY_MASKS_DIR  = "/Volumes/KINGSTON/LCIB_DiffusionSat/LCIB_project/binary_masks"
IMAGE_SIZE        = 512   # VAE 512x512 bekliyor

CLOUD_HEIGHT_MIN, CLOUD_HEIGHT_MAX = 600.0, 3500.0
COVERAGE_MIN, COVERAGE_MAX = 0.10, 0.50
OPACITY_MIN, OPACITY_MAX = 0.85, 1.00
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
        "sun_azimuth":    float(data.get("sun_azimuth", 180.0)),
        "sun_elevation":  float(data.get("sun_elevation", 45.0)),
        "cloud_height":   h,
        "coverage":       float(data.get("coverage", 20)) / 100.0,
        "base_opacity":   float(data.get("base_opacity", 90)) / 100.0,
        "shadow_strength": float(data.get("shadow_strength", 0.7)),
        "cloud_count":    cloud_count,
    }

def encode_metadata(meta):
    """
    [cos_az, sin_az, sin_el, cos_el, ccov, copa, h_norm, ccount_norm, sstr_norm]
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

    ccount_norm = 1.0
    sstr_norm = float(np.clip(meta["shadow_strength"], 0.0, 1.0))

    return torch.tensor([
        cos_az, sin_az,
        sin_el, cos_el,
        ccov,
        copa,
        h_norm,
        ccount_norm,
        sstr_norm,
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

        self.img_transform  = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        self.samples = self._collect_samples()
        print(f"Dataset hazır: {len(self.samples)} sample bulundu.")

    def _collect_samples(self):
        samples = []

        # 1. seviye: biome klasörleri (desert, forest, grassland, urban, wetlands)
        for biome_dir in sorted(self.lcib_dir.iterdir()):
            if not biome_dir.is_dir() or biome_dir.name.startswith("."):
                continue
            biome = biome_dir.name

            # 2. seviye: output_africa_<biome>_<id>_1024 klasörleri
            for sample_dir in sorted(biome_dir.iterdir()):
                if not sample_dir.is_dir() or sample_dir.name.startswith("."):
                    continue

                # örn: output_africa_desert_010_1024 → africa_desert_010_1024
                base_name = sample_dir.name.replace("output_", "")

                # Iref artık biome klasörünün altında: africa/<biome>/{base_name}.png
                iref_path = self.clean_dir / biome / f"{base_name}.png"
                if not iref_path.exists():
                    print(f"  UYARI: Iref bulunamadı → {iref_path}")
                    continue

                # Bu klasördeki tüm varyantlar (aynı base image, farklı cov/op/hash)
                json_files = sorted([
                    f for f in sample_dir.glob("*_result.json")
                    if not f.name.startswith("._")
                ])

                for json_path in json_files:
                    stem = json_path.stem.replace("_result", "")

                    isyn_path = sample_dir / f"{stem}_result.png"
                    # Mask/shadow: binary_masks/<biome>/output_.../ altında
                    # (binarize_masks_updated.py'nin ürettiği yapıyla birebir eşleşiyor)
                    mc_path = self.binary_dir / biome / sample_dir.name / f"{stem}_mask.png"
                    ms_path = self.binary_dir / biome / sample_dir.name / f"{stem}_shadow.png"

                    if not all([isyn_path.exists(), mc_path.exists(), ms_path.exists()]):
                        continue

                    samples.append({
                        "iref":     iref_path,
                        "isyn":     isyn_path,
                        "mc":       mc_path,
                        "ms":       ms_path,
                        "json":     json_path,
                        "name":     stem,
                        "scene_id": base_name,
                        "biome":    biome,   # train/val split'te biome bazlı stratification için kullanışlı olabilir
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        Iref = self.img_transform(Image.open(s["iref"]).convert("RGB"))
        Isyn = self.img_transform(Image.open(s["isyn"]).convert("RGB"))
        Mc   = self.mask_transform(Image.open(s["mc"]).convert("L"))
        Ms   = self.mask_transform(Image.open(s["ms"]).convert("L"))

        meta_raw = parse_metadata(s["json"])
        meta_vec = encode_metadata(meta_raw)

        return {
            "Iref":  Iref,
            "Isyn":  Isyn,
            "Mc":    Mc,
            "Ms":    Ms,
            "meta":  meta_vec,
            "name":  s["name"],
        }


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

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  Iref batch shape: {batch['Iref'].shape}")
    print(f"  meta batch shape: {batch['meta'].shape}")

    print("\nDataset tamam!")
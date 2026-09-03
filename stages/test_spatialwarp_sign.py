"""
test_spatialwarp_sign.py — SpatialWarp'ın yön kuralını + gerçek veride
gölgenin fiziksel kayma yönünü doğrulayan birim test.

BÖLÜM 1 (orijinal path dışında değiştirilmedi):
    Rapor denklem 22:
        Fwarp(x, y) = zc(x - Δx, y - Δy)
    Sentetik tek pikselli bir zc üzerinde bilinen bir (Δx, Δy) uygulayıp
    pikselin gerçekte hangi yöne kaydığını ölçer.

BÖLÜM 2 (YENİ, ekleme): Bölüm 1 SpatialWarp'ın kodunun rapor formülüyle
tutarlı mı ters mi çalıştığını söylüyor — ama "hangi yön fiziksel olarak
DOĞRU" sorusuna cevap vermiyor. Bölüm 2 bunu, modelden ve VAE'den tamamen
bağımsız olarak, gerçek dataset üzerinde test ediyor: her örnekte Mc (bulut
maskesi) ile Ms (gölge maskesi) ağırlık merkezleri arasındaki GERÇEK yönü
ölçüp, bunu modelin va=(cosθa, sinθa) azimut kodlamasıyla karşılaştırıyor.

Üç olası sonuç:
    ~0°   → gözlenen Mc→Ms yönü va ile AYNI (paralel)
    ~180° → gözlenen yön va'nın TAM TERSİ (anti-paralel)
    ~±90° → gözlenen yön va'ya DİK (90° dönüklük şüphesi doğrulanır)

Bölüm 1 + Bölüm 2 birlikte değerlendirilmeli: SpatialWarp'ın işareti VE
Ldir'in hizalama yönü, Bölüm 2'nin bulduğu gerçek fiziksel ilişkiyle
tutarlı olmalı. Değilse, ikisinin de gözden geçirilmesi gerekir.

Çalıştırma:
    python test_spatialwarp_sign.py
"""
import sys
import math
import torch

sys.path.append("/Volumes/KIOXIA/LCIB_DiffusionSat/LCIB_project/stages")
from stage3_displacement import SpatialWarp


# ─────────────────────────────────────────────────────────────
# BÖLÜM 1 — orijinal test (path dışında değiştirilmedi)
# ─────────────────────────────────────────────────────────────
def run_test_part1():
    H, W = 16, 16
    zc = torch.zeros(1, 1, H, W)

    # Merkeze yakın, kolay okunur bir noktaya tek beyaz piksel koy
    y0, x0 = 8, 8
    zc[0, 0, y0, x0] = 1.0

    # Sadece x ekseninde, pozitif ve büyük bir kayma dene (görünür olsun diye)
    dx_latent = 4.0   # latent-piksel cinsinden Δx
    dy_latent = 0.0
    d = torch.tensor([[dx_latent, dy_latent]])  # [1, 2]

    warp = SpatialWarp()
    Fwarp = warp(zc, d)

    # Çıktıda beyaz pikselin yeni konumunu bul
    out = Fwarp[0, 0]
    y_new, x_new = (out == out.max()).nonzero()[0].tolist()

    observed_dx = x_new - x0
    observed_dy = y_new - y0

    print(f"{'='*60}")
    print("BÖLÜM 1: SpatialWarp kod işareti (sentetik tek piksel)")
    print(f"{'='*60}\n")

    print(f"Girdi piksel konumu   : (x={x0}, y={y0})")
    print(f"Uygulanan d           : (Δx={dx_latent}, Δy={dy_latent})")
    print(f"Çıktıda piksel konumu : (x={x_new}, y={y_new})")
    print(f"Gözlenen kayma        : (Δx_gözlenen={observed_dx}, Δy_gözlenen={observed_dy})")
    print()

    # ─── Yorumlama ───
    # Rapor (denklem 22): Fwarp(x,y) = zc(x-Δx, y-Δy)
    #   => çıktının x noktası, zc'nin (x-Δx) noktasını okur
    #   => zc'deki bir özellik (örn. beyaz piksel x0'da), çıktıda
    #      x0+Δx konumunda GÖRÜNÜR (çünkü x=x0+Δx için x-Δx=x0 olur)
    #   => yani rapora göre: observed_dx == +dx_latent olmalı
    expected_dx_report = dx_latent

    if abs(observed_dx - expected_dx_report) < 1e-6:
        print("SONUÇ: Kod, rapor denklem 22 ile TUTARLI.")
        print("  (Pozitif Δx verildiğinde piksel +x yönünde kayıyor,")
        print("   rapordaki 'zc(x-Δx,y-Δy)' kuralıyla eşleşiyor.)")
    else:
        print("SONUÇ: Kod, rapor denklem 22 ile TERS YÖNDE ÇALIŞIYOR.")
        print(f"  Beklenen (rapora göre): Δx_gözlenen={expected_dx_report}")
        print(f"  Gerçekte ölçülen      : Δx_gözlenen={observed_dx}")
        print("  Bu ya kasıtlı bir kural farkı (göreli hareket yönü tanımı)")
        print("  ya da gerçek bir işaret hatası olabilir — Amir Hoca'ya")
        print("  sorulmalı: 'gölgenin kayma yönü fiziksel olarak doğru mu?'")
    print()


# ─────────────────────────────────────────────────────────────
# BÖLÜM 2 — YENİ: gerçek veride fiziksel yön testi
# ─────────────────────────────────────────────────────────────
def centroid(mask_tensor):
    """
    mask_tensor: [1, H, W] veya [H, W], 0/1 ya da 0/255 değerli binary mask.
    Döndürür: (x_c, y_c) ağırlıklı piksel merkezi, ya da maske boşsa None.
    """
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor[0]
    mask = mask_tensor.float()
    if mask.max() > 1.0:
        mask = mask / 255.0

    H, W = mask.shape
    ys = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
    xs = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)

    total = mask.sum()
    if total < 1e-6:
        return None
    x_c = (mask * xs).sum() / total
    y_c = (mask * ys).sum() / total
    return x_c.item(), y_c.item()


def run_test_part2(n_samples=5):
    """
    Gerçek dataset örnekleri üzerinde: Mc merkezinden Ms merkezine giden
    GÖZLENEN yön ile, metadata'daki güneş azimutundan türetilen
    va=(cosθa, sinθa) arasındaki açıyı ölçer. SpatialWarp'a veya VAE'ye
    hiç ihtiyaç duymaz — doğrudan simülatörün ürettiği gerçek maskeler
    üzerinde çalışır, yani "modelden bağımsız zemin gerçeği" (ground truth)
    testi budur.
    """
    from dataset import LCIBDataset
    from torch.utils.data import DataLoader

    print(f"{'='*60}")
    print("BÖLÜM 2: Gerçek veride Mc→Ms yönü vs. va=(cosθa, sinθa)")
    print(f"{'='*60}\n")

    dataset = LCIBDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    checked = 0
    angle_diffs = []

    for batch in loader:
        if checked >= n_samples:
            break

        Mc = batch["Mc"][0]     # [1, H, W]
        Ms = batch["Ms"][0]     # [1, H, W]
        meta = batch["meta"][0]  # [9]
        name = batch["name"][0]

        c_cloud = centroid(Mc)
        c_shadow = centroid(Ms)
        if c_cloud is None or c_shadow is None:
            continue  # boş maske, bu örneği atla

        # Gözlenen yön: bulut merkezinden gölge merkezine giden vektör
        obs_dx = c_shadow[0] - c_cloud[0]
        obs_dy = c_shadow[1] - c_cloud[1]
        obs_angle = math.degrees(math.atan2(obs_dy, obs_dx))

        # va = (cos_az, sin_az) — dataset.py encode_metadata sırasına göre meta[0], meta[1]
        cos_az, sin_az = meta[0].item(), meta[1].item()
        va_angle = math.degrees(math.atan2(sin_az, cos_az))

        # İki açı arasındaki fark, -180..180 aralığına normalize edilmiş
        diff = (obs_angle - va_angle + 180) % 360 - 180
        angle_diffs.append(diff)

        print(f"[{name}]")
        print(f"  Mc merkezi         : ({c_cloud[0]:.1f}, {c_cloud[1]:.1f})")
        print(f"  Ms merkezi         : ({c_shadow[0]:.1f}, {c_shadow[1]:.1f})")
        print(f"  Gözlenen yön açısı : {obs_angle:.1f}°")
        print(f"  va (azimut) açısı  : {va_angle:.1f}°")
        print(f"  Fark               : {diff:+.1f}°")

        if abs(diff) < 20:
            print("  → YORUM: Gözlenen yön va ile ~AYNI (paralel)")
        elif abs(abs(diff) - 180) < 20:
            print("  → YORUM: Gözlenen yön va'nın ~TERSİ (anti-paralel)")
        elif abs(abs(diff) - 90) < 20:
            print("  → YORUM: Gözlenen yön va'ya ~DİK (90° dönüklük şüphesi doğrulandı)")
        else:
            print("  → YORUM: Belirgin bir 0/90/180 kalıbına uymuyor")
        print()

        checked += 1

    print(f"{'='*60}")
    print(f"{checked} örnek kontrol edildi.")
    if angle_diffs:
        avg_diff = sum(angle_diffs) / len(angle_diffs)
        print(f"Ortalama fark: {avg_diff:+.1f}°")
    print("Yukarıdaki YORUM satırlarının çoğunluğu hangi kalıba uyuyorsa")
    print("(paralel / anti-paralel / dik), gerçek fiziksel ilişki odur.")
    print("Bu sonucu Bölüm 1'deki SpatialWarp işaretiyle birlikte")
    print("değerlendirin: ikisi tutarlı olmalı. Örneğin gerçek ilişki")
    print("~90° (dik) çıkıyorsa, sadece SpatialWarp'ın işaretini çevirmek")
    print("yetmez — Ldir'in va ile hizalama mantığı da gözden geçirilmeli.")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_test_part1()
    run_test_part2(n_samples=5)
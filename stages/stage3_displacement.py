import torch
import torch.nn as nn
import torch.nn.functional as F


class DisplacementMLP(nn.Module):
    """
    Stage 3: Learnable Displacement Prediction.
    Rapor denklem 18:
      d = (Δx, Δy) = MLPwarp(va, ve, ccov, copa, h)

    Input: meta [B, 8]
      [cos_az, sin_az, sin_el, cos_el, ccov, copa, h_norm, ccount]
    Output: d [B, 2] — latent uzayda displacement (Δx, Δy)

    DisplacementMLP metadata'dan (sadece ilk 7 boyut: azimuth+elevation+coverage+opacity+height) bir kayma vektörü d=(Δx,Δy) tahmin ediyor.
    SpatialWarp bulut maskesinin latent temsilini (zc) bu d kadar kaydırıyor → Fwarp.

    DÜZELTME (Lmin/Minfo çökme fix'i, 3. parça — displacement sınırlama):
    d daha önce sınırsız bir ham nn.Linear çıktısıydı (bkz. git geçmişindeki
    ilk taslak notu — orada zaten tanh + max_shift ile sınırlanmıştı, ama
    kodda bu sınır hiç uygulanmamıştı). Sınırsız d, eğitim sırasında büyük
    değerlere ulaşabiliyor; bu da iki ayrı soruna yol açıyor:
      1) SpatialWarp'ın grid'i zaten [-1,1]'e clamp'leniyor (aşağıda), yani
         |d| latent'in yarı genişliğini (W/2=32) aştığında Fwarp tamamen
         doyuma ulaşıp anlamsızlaşıyor.
      2) Daha kritik: stage9_losses.mask_target_loss, Mc'yi d*f kadar
         kaydırıp Laplacian/kenar haritası çıkarıyor (Lmin'e karşı-kuvvet
         olarak eklenen Lpreserve_mask'ın hedefi). |d| yeteri kadar
         büyüdüğünde (ölçüldü: latent-piksel ~30-40 civarı, img-piksel
         ~240-320) kaydırılmış bulut çerçevenin tamamen dışına çıkıyor,
         kenar haritası TAMAMEN sıfırlanıyor, ve Lpreserve_mask sessizce
         Lmin ile aynı yöne (Minfo→0) dönüyor — tam da karşı-kuvvete en
         çok ihtiyaç duyulan anda devre dışı kalıyor.
      Rapor'un kendi Lele formülüne göre (h/((tanθe)·s·f), s=15, f=8) bu
      büyüklükte kaymalar sıradan/beklenen metadata kombinasyonlarında bile
      ortaya çıkabiliyor — yani nadir bir uç durum değil.
    Çözüm: d'yi tanh ile [-max_shift, max_shift] aralığına sınırlıyoruz.
    max_shift=24 seçildi: hem grid doygunluğundan (32) hem de
    mask_target_loss'un dejenere olduğu bölgeden (~30+) güvenli marj
    bırakıyor. NOT: bu, Lele'nin çok düşük güneş açısı / çok yüksek bulut
    kombinasyonlarında beklediği (>24) değerlere hiç ulaşamayacağı anlamına
    gelir — Lele o örneklerde asla 0'a inemez. Bu, mevcut latent
    çözünürlük/AMM tasarımıyla rapor'un fiziksel ölçeğinin (s=15, f=8)
    tam örtüşmediğini gösteriyor; kalıcı çözüm için Amir Hoca ile
    netleştirilmeli (TODO).
    """
    def __init__(self, hidden_dim=128, max_shift=24.0):
        super().__init__()
        self.max_shift = max_shift

        # 7 boyutlu input: va(2) + ve(2) + ccov(1) + copa(1) + h(1)
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)   # (Δx, Δy)
        )

    def forward(self, meta):
        """
        meta: [B, 8]
        Returns: d [B, 2] — |d| <= max_shift (latent piksel)
        """
        # idx 7 (ccount) kullanılmıyor, ilk 7'yi al
        x = meta[:, :7]                          # [B, 7]
        d = torch.tanh(self.mlp(x)) * self.max_shift   # [B, 2]
        return d


class SpatialWarp(nn.Module):
    """
    Stage 4: Latent-Space Warping.
    Rapor denklem 21-22:
      Fwarp(x, y) = zc(x - Δx, y - Δy)

    Translation warp via bilinear grid sampling.
    TPS değil, rigid translation — fiziksel motivasyon:
    gölge desplamanı rigid bir geometrik kaymadır.
    """
    def __init__(self):
        super().__init__()

    def forward(self, zc, d):
        """
        zc: [B, 4, H, W] — cloud mask latent
        d : [B, 2]        — (Δx, Δy) displacement latent piksel cinsinden
        Returns: Fwarp [B, 4, H, W]
        """
        B, C, H, W = zc.shape

        # Displacement'ı normalize et: latent piksel → [-1, 1] grid aralığı
        # Grid sampling -1 ile 1 arasında çalışır
        dx = d[:, 0] / (W / 2.0)  # [B]
        dy = d[:, 1] / (H / 2.0)  # [B]

        # Base grid oluştur: [B, H, W, 2]
        base_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, W, device=zc.device),
                torch.linspace(-1, 1, H, device=zc.device),
                indexing='xy'
            ), dim=-1
        ).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # Displacement ekle (translation)
        # dx → x ekseninde kayma, dy → y ekseninde kayma
        #
        # DÜZELTME (SpatialWarp işaret fix'i): grid_sample semantiği
        # output(x,y) = input(grid(x,y)) olduğundan, base_grid + shift
        # kullanmak Fwarp(x,y) = zc(x+Δx, y+Δy) üretiyordu — rapor denklem
        # 22'nin (Fwarp(x,y) = zc(x-Δx, y-Δy)) TAM TERSİ. Bu,
        # test_spatialwarp_sign_extended.py Bölüm 1 ile empirik olarak
        # doğrulandı (Δx=+4 verildiğinde gözlenen kayma -4 çıkıyordu).
        # base_grid'den shift ÇIKARARAK rapor formülüyle eşleştiriyoruz.
        shift = torch.stack([dx, dy], dim=-1)        # [B, 2]
        shift = shift.view(B, 1, 1, 2)               # [B, 1, 1, 2]
        grid  = base_grid - shift                     # [B, H, W, 2] — rapor eq22: zc(x-Δx, y-Δy)
        grid  = grid.clamp(-1, 1)

        # Bilinear sampling ile warp uygula
        Fwarp = F.grid_sample(
            zc, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        return Fwarp


# ─── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KIOXIA/LCIB_DiffusionSat/LCIB_project/stages")
    from dataset import LCIBDataset
    from stage1_encoder import VAEEncoder
    from torch.utils.data import DataLoader

    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=2, shuffle=False)
    batch   = next(iter(loader))

    Mc   = batch["Mc"]    # [2, 1, 512, 512]
    meta = batch["meta"]  # [2, 8]

    # VAE ile zc üret
    print("VAE yükleniyor...")
    vae_enc = VAEEncoder()
    zc = vae_enc.encode(Mc)   # [2, 4, 64, 64]
    print(f"zc shape: {zc.shape}")

    # Displacement tahmin et
    disp_mlp = DisplacementMLP(hidden_dim=128)
    d = disp_mlp(meta)
    print(f"d shape : {d.shape}")    # [2, 2]
    print(f"d values: {d}")          # (Δx, Δy)

    # Warp uygula
    warp = SpatialWarp()
    Fwarp = warp(zc, d)
    print(f"Fwarp shape: {Fwarp.shape}")  # [2, 4, 64, 64]

    # Directional alignment loss — rapor denklem 20
    va  = meta[:, 0:2]                            # [B, 2] azimuth vektörü
    d_norm   = F.normalize(d, dim=-1)
    va_norm  = F.normalize(va, dim=-1)
    Ldir = 1.0 - (d_norm * va_norm).sum(dim=-1).mean()
    print(f"\nLdir (directional loss): {Ldir.item():.4f}")
    print("  (0'a yakın = azimuth ile hizalı, 2'ye yakın = tam ters)")

    print("\nStage 3+4 tamam!")
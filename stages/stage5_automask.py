import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(num_channels, max_groups=8):
    """
    GroupNorm için grup sayısını kanal sayısına göre güvenli seçer
    (num_channels, num_groups'a tam bölünmeli). AMM'deki kanal sayıları
    (64, 128) için normalde 8 grup kullanılır; küçük/garip bir kanal
    sayısı gelirse otomatik olarak bölen bir değere düşer.
    """
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ResBlock(nn.Module):
    """
    Residual block: Conv-GroupNorm-SiLU-Conv-GroupNorm + skip connection.
    AMM'nin her FPN seviyesinde 2 tanesi kullanılıyor (rapor 2.5.1).

    DÜZELTME (Lmin/Minfo çökme fix'i — kök neden): bu blokta hiç
    normalization yoktu (sadece Conv+SiLU). Katmanlar derinleştikçe
    (level1 → level2 → level3_pre, toplam ~6 conv) aktivasyon varyansı
    kontrolsüz büyüyordu — eğitim başlamadan, sırf rastgele ağırlık
    init'iyle bile fusion katmanına giren pre-sigmoid değerleri
    [-10, +28] gibi aşırı aralıklara ulaşıyordu (ölçüldü, bkz. debug
    testi). sigmoid(±10+) pratikte 0 veya 1'de doyuma uğrar ve o
    bölgede gradyan ~0'dır (vanishing gradient) — Minfo bir kere öyle
    bir bölgeye düşünce Lmin'in ufak bir baskısı bile onu kalıcı olarak
    0'a kilitliyordu; warm-up bunu çözmedi çünkü sorun gradyan baskısının
    zamanlaması değil, forward pass'in kendisiydi.

    Çözüm: WindowLocalAttention'da (stage 6) daha önce aynı sebeple
    yapılan düzeltmeyle tutarlı olarak, her conv'dan sonra GroupNorm
    eklendi. Bu, aktivasyon varyansını katman katman kontrol altında
    tutar, sigmoid girişini eğitim başında makul (~[-3,+3]) bir aralıkta
    tutmaya yardımcı olur.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = _make_group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _make_group_norm(channels)
        self.act   = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class FPNLevel(nn.Module):
    """
    Tek bir FPN seviyesi: (opsiyonel terrain feature ile concat) →
    stride-2 conv (downsample) → GroupNorm → 2x ResBlock.
    """
    def __init__(self, in_channels, terrain_channels, out_channels):
        super().__init__()
        total_in = in_channels + terrain_channels
        self.down = nn.Conv2d(total_in, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = _make_group_norm(out_channels)
        self.act  = nn.SiLU()
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)

    def forward(self, x, terrain_feat):
        h = torch.cat([x, terrain_feat], dim=1)
        h = self.act(self.norm(self.down(h)))
        h = self.res1(h)
        h = self.res2(h)
        return h


class AutoMaskModule(nn.Module):
    """
    Stage 5: Auto Mask Module (AMM).
    Rapor denklem 23-24.

    Minfo = UNetAMM(Fwarp, Fterrain)   (23)
    Ffused = Fwarp ⊙ Minfo             (24)

    Hafif 3 seviyeli FPN: her seviye stride-2 conv + 2 residual block.
    F1 (256x256) → F2 (128x128) → F3 (64x64) sırasıyla downsample edilip
    her adımda ilgili terrain feature ile concat edilir. Son seviyede
    Fwarp (zaten 64x64, latent çözünürlük) ile birleştirilip Minfo üretilir.

    F4 kullanılmıyor: stage1 testinde F3 ile aynı shape'de çıktığı için
    (aynı son downblock'un çıktısı), F3 yeterli terrain bilgisini taşıyor.

    DÜZELTME (Lmin/Minfo çökme fix'i, 2. parça — zero-init fusion):
    `fusion` katmanı artık ağırlık ve bias'ı SIFIR ile başlatılıyor
    (ConditioningProjector'daki (stage 8) ControlNet zero-conv
    konvansiyonuyla aynı mantık). Bunun etkisi: eğitimin ilk adımında
    pre-sigmoid değeri KESİN OLARAK 0'dır → Minfo = sigmoid(0) = 0.5
    her yerde. Yani Minfo, ne tam açık (1) ne tam kapalı (0) bir
    doyum noktasından değil, tam ortadan (gradyanın en güçlü olduğu
    bölgeden) öğrenmeye başlıyor. GroupNorm fix'iyle birlikte, ilk
    adımlardaki aşırı pre-sigmoid değerlerini engelleyip Lmin'in
    kademeli/gerçek bir öğrenme sinyali olarak işlev görmesini sağlaması
    bekleniyor.
    """
    def __init__(self, terrain_channels=(128, 256, 512), fpn_channels=(64, 128, 128), warp_channels=4):
        super().__init__()
        c1, c2, c3 = fpn_channels
        t1, t2, t3 = terrain_channels

        self.level1 = nn.Sequential(
            nn.Conv2d(t1, c1, kernel_size=3, stride=2, padding=1),
            _make_group_norm(c1),
            nn.SiLU(),
            ResBlock(c1),
            ResBlock(c1),
        )

        self.level2 = FPNLevel(in_channels=c1, terrain_channels=t2, out_channels=c2)

        self.level3_pre = nn.Sequential(
            nn.Conv2d(c2 + t3, c3, kernel_size=3, stride=1, padding=1),
            _make_group_norm(c3),
            nn.SiLU(),
            ResBlock(c3),
            ResBlock(c3),
        )

        self.fusion = nn.Conv2d(c3 + warp_channels, 1, kernel_size=1)
        # Zero-init: eğitim başında Minfo = sigmoid(0) = 0.5'ten başlasın.
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, Fwarp, F_terrain):
        """
        Fwarp    : [B, 4, 64, 64]   — Stage 4'ten warp edilmiş cloud latent
        F_terrain: dict {"F1","F2","F3","F4"} — Stage 1 TerrainEncoder çıktısı
          F1: [B, 128, 256, 256]
          F2: [B, 256, 128, 128]
          F3: [B, 512, 64, 64]

        Returns:
            Minfo : [B, 1, 64, 64]  — [0,1] aralığında soft confidence map
            Ffused: [B, 4, 64, 64]  — Fwarp ⊙ Minfo
        """
        F1, F2, F3 = F_terrain["F1"], F_terrain["F2"], F_terrain["F3"]

        h = self.level1(F1)              # [B, c1, 128, 128]
        h = self.level2(h, F2)           # [B, c2, 64, 64]
        h = torch.cat([h, F3], dim=1)    # [B, c2+t3, 64, 64]
        h = self.level3_pre(h)           # [B, c3, 64, 64]

        h = torch.cat([h, Fwarp], dim=1)  # [B, c3+4, 64, 64]
        Minfo = torch.sigmoid(self.fusion(h))  # [B, 1, 64, 64]

        Ffused = Fwarp * Minfo  # element-wise gating (denklem 24)
        return Minfo, Ffused


# ─── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KIOXIA/LCIB_DiffusionSat/LCIB_project/stages")
    from dataset import LCIBDataset
    from stage1_encoder import TerrainEncoder, VAEEncoder
    from stage3_displacement import DisplacementMLP, SpatialWarp
    from torch.utils.data import DataLoader

    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=2, shuffle=False)
    batch   = next(iter(loader))

    Iref = batch["Iref"]   # [2, 3, 512, 512]
    Mc   = batch["Mc"]     # [2, 1, 512, 512]
    meta = batch["meta"]   # [2, 8]

    print("Encoderlar yükleniyor...")
    terrain_enc = TerrainEncoder()
    vae_enc     = VAEEncoder()

    _, F_terrain = terrain_enc(Iref)
    zc = vae_enc.encode(Mc)   # [2, 4, 64, 64]

    disp_mlp = DisplacementMLP(hidden_dim=128)
    d = disp_mlp(meta)
    warp = SpatialWarp()
    Fwarp = warp(zc, d)      # [2, 4, 64, 64]

    print(f"Fwarp shape: {Fwarp.shape}")
    for k in ["F1", "F2", "F3"]:
        print(f"{k} shape: {F_terrain[k].shape}")

    amm = AutoMaskModule()
    Minfo, Ffused = amm(Fwarp, F_terrain)

    print(f"\nMinfo shape : {Minfo.shape}")
    print(f"Minfo range : [{Minfo.min():.3f}, {Minfo.max():.3f}]")
    print(f"Minfo mean  : {Minfo.mean():.3f}  (zero-init sayesinde ~0.500 olmalı)")
    print(f"Ffused shape: {Ffused.shape}")

    print("\nStage 5 tamam!")
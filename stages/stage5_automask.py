import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Basit residual block: Conv-SiLU-Conv + skip connection.
    AMM'nin her FPN seviyesinde 2 tanesi kullanılıyor (rapor 2.5.1).
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act   = nn.SiLU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class FPNLevel(nn.Module):
    """
    Tek bir FPN seviyesi: (opsiyonel terrain feature ile concat) →
    stride-2 conv (downsample) → 2x ResBlock.
    """
    def __init__(self, in_channels, terrain_channels, out_channels):
        super().__init__()
        total_in = in_channels + terrain_channels
        self.down = nn.Conv2d(total_in, out_channels, kernel_size=3, stride=2, padding=1)
        self.act  = nn.SiLU()
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)

    def forward(self, x, terrain_feat):
        h = torch.cat([x, terrain_feat], dim=1)
        h = self.act(self.down(h))
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
    """
    def __init__(self, terrain_channels=(128, 256, 512), fpn_channels=(64, 128, 128), warp_channels=4):
        super().__init__()
        c1, c2, c3 = fpn_channels
        t1, t2, t3 = terrain_channels

        self.level1 = nn.Sequential(
            nn.Conv2d(t1, c1, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            ResBlock(c1),
            ResBlock(c1),
        )

        self.level2 = FPNLevel(in_channels=c1, terrain_channels=t2, out_channels=c2)

        self.level3_pre = nn.Sequential(
            nn.Conv2d(c2 + t3, c3, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            ResBlock(c3),
            ResBlock(c3),
        )

        self.fusion = nn.Conv2d(c3 + warp_channels, 1, kernel_size=1)

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
    sys.path.append("/Volumes/KINGSTON/LCIB_DiffusionSat/LCIB_project/stages")
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
    print(f"Ffused shape: {Ffused.shape}")

    print("\nStage 5 tamam!")

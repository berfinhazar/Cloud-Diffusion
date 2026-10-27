import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_PATH = "/Volumes/KIOXIA/LCIB_checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"


class ConditioningProjector(nn.Module):
    """
    Stage 8 - Cok olcekli (ControlNet tarzi) conditioning enjeksiyonu.

    Rapor 2.6.4 / 2.7.4: Fattn / Fout, UNet'in decoder olceklerine enjekte
    edilir. SatUNet'te decoder'a giden bilgi skip baglantilari uzerinden
    tasindigi icin, her skip tensorune (down_block_additional_residuals)
    ve mid_block cikisina (mid_block_additional_residual) bir residual ekliyoruz.

    Girdi : Fout [B,4,64,64]   (Stage 7 cikisi: bulut+golge sinyali)
            F3   [B,512,64,64] (Stage 1 terrain feature, rapor 2.1.2)
    Cikti : (down_residuals, mid_residual)
            down_residuals: UNet'in down_block_res_samples sirasiyla, tensor
              demeti. Sayi/kanal/cozunurluk block_out_channels ve
              layers_per_block'tan OTOMATIK hesaplanir (SD2.1: 12 adet).
            mid_residual  : [B, C_son, 8, 8]

    Yapi: kucuk bir ozellik piramidi (64->32->16->8, P kanal) ve her skip icin
    1x1 "zero conv" (ControlNet konvansiyonu: cok kucuk baslangic, egitim
    basinda hazir UNet'in davranisi bozulmaz).
    """
    def __init__(self, block_out_channels=(320, 640, 1280, 1280), layers_per_block=2,
                 in_channels=4, terrain_channels=512, pyramid_channels=128):
        super().__init__()
        P = pyramid_channels
        n_levels = len(block_out_channels)
        self.n_levels = n_levels

        # UNet'in down_block_res_samples sirasi: conv_in cikisi, sonra her
        # down_block icin layers_per_block resnet cikisi, sonra (son blok haric)
        # downsample cikisi. Her eleman: (kanal, piramit seviyesi)
        specs = [(block_out_channels[0], 0)]
        for i, ch in enumerate(block_out_channels):
            for _ in range(layers_per_block):
                specs.append((ch, i))
            if i < n_levels - 1:
                specs.append((ch, i + 1))
        self.skip_specs = specs

        # Seviye 0 (latent cozunurluk): Fout ve terrain (F3) birlestirilir
        self.in_conv = nn.Conv2d(in_channels, P, kernel_size=3, padding=1)
        self.terrain_proj = nn.Conv2d(terrain_channels, P, kernel_size=1)
        self.norm0 = nn.GroupNorm(8, P)
        self.act = nn.SiLU()

        # Seviye 1..n-1: her biri cozunurlugu yariya indirir
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(P, P, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(P, P, kernel_size=3, padding=1), nn.SiLU(),
            ) for _ in range(n_levels - 1)
        ])

        self.zero_convs = nn.ModuleList([nn.Conv2d(P, ch, kernel_size=1) for ch, _ in specs])
        self.mid_zero_conv = nn.Conv2d(P, block_out_channels[-1], kernel_size=1)
        # Tam sifir yerine cok kucuk random init (onceki tasarimdaki secimle ayni):
        # gradyan sinyali en bastan akabilsin.
        for conv in list(self.zero_convs) + [self.mid_zero_conv]:
            nn.init.normal_(conv.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(conv.bias)

    def forward(self, Fout, F_terrain_local):
        """
        Fout: [B,4,64,64], F_terrain_local: [B,512,64,64]
        Returns: (down_residuals: tuple, mid_residual: Tensor)
        """
        h = self.act(self.norm0(self.in_conv(Fout) + self.terrain_proj(F_terrain_local)))
        feats = [h]
        for down in self.downs:
            h = down(h)
            feats.append(h)
        down_res = tuple(zc(feats[lvl]) for zc, (_, lvl) in zip(self.zero_convs, self.skip_specs))
        mid_res = self.mid_zero_conv(feats[-1])
        return down_res, mid_res


def forward_diffusion(z0, t, alphas_cumprod):
    """
    Rapor denklem 33-34: q(zt | z0) = N(sqrt(αt) z0, (1-αt) I)
    zt = sqrt(αt) z0 + sqrt(1-αt) ε

    z0            : [B, 4, 64, 64] — temiz latent (VAE ile encode edilmiş Isyn)
    t             : [B] — timestep indeksleri
    alphas_cumprod: [T] — noise scheduler'dan cumulative alpha değerleri

    Returns: zt [B,4,64,64], eps [B,4,64,64] (eklenen gürültü, ground truth)
    """
    eps = torch.randn_like(z0)
    sqrt_at   = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_1_at = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    zt = sqrt_at * z0 + sqrt_1_at * eps
    return zt, eps


# ─── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KIOXIA/LCIB_DiffusionSat/LCIB_project/stages")

    from dataset import LCIBDataset
    from stage1_encoder import TerrainEncoder, VAEEncoder
    from stage3_displacement import DisplacementMLP, SpatialWarp
    from stage5_automask import AutoMaskModule
    from stage6_local_attention import WindowLocalAttention
    from stage7_film import FiLMModulation
    from torch.utils.data import DataLoader
    from diffusers import DDPMScheduler

    # SatUNet — stages/sat_unet.py içinde (DiffusionSat reposundan alınan,
    # değiştirilmeden kopyalanmış tek dosya)
    from sat_unet import SatUNet

    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=2, shuffle=False)
    batch   = next(iter(loader))

    Iref = batch["Iref"]
    Isyn = batch["Isyn"]
    Mc   = batch["Mc"]
    Ms   = batch["Ms"]
    meta = batch["meta"]

    print("Encoderlar yükleniyor...")
    terrain_enc = TerrainEncoder()
    vae_enc     = VAEEncoder()

    _, F_terrain = terrain_enc(Iref)
    zc = vae_enc.encode(Mc)
    z0 = vae_enc.encode(Isyn)   # Stage 8.1 — denklem 32: z0 = E(Isyn)

    disp_mlp = DisplacementMLP(hidden_dim=128)
    d = disp_mlp(meta)
    warp = SpatialWarp()
    Fwarp = warp(zc, d)

    # DÜZELTME: AutoMaskModule artık Ms (gölge maskesi) zorunlu 3. parametre
    # istiyor (bkz. stage5_automask.py, Ms entegrasyonu) — bu test bloğu
    # güncellenmemiş kalmıştı, eksik parametre hatası veriyordu.
    amm = AutoMaskModule()
    _, Ffused = amm(Fwarp, F_terrain, Ms)

    local_attn = WindowLocalAttention(
        fused_channels=4, terrain_channels=512,
        d_model=64, num_heads=4, window_size=8
    )
    Fattn = local_attn(Ffused, F_terrain["F3"])

    film = FiLMModulation(fused_channels=4, hidden_dim=64)
    Fout, _, _ = film(meta, Fattn)
    print(f"Fout shape: {Fout.shape}")

    # Forward diffusion (denklem 33-34)
    print("\nNoise scheduler yükleniyor...")
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    B = z0.shape[0]
    t = torch.randint(0, scheduler.config.num_train_timesteps, (B,))
    zt, eps = forward_diffusion(z0, t, scheduler.alphas_cumprod)
    print(f"zt shape: {zt.shape}, t: {t}")

    # Conditioning projector (Fout + F3 → down/mid residual'lar)
    projector = ConditioningProjector()
    down_res, mid_residual = projector(Fout, F_terrain["F3"])
    print(f"down residual sayısı: {len(down_res)}, mid shape: {mid_residual.shape}")

    # SatUNet yükleniyor
    print("\nSatUNet yükleniyor (biraz sürebilir)...")
    unet = SatUNet.from_pretrained(CHECKPOINT_PATH, subfolder="unet")
    unet.eval()

    # Placeholder conditioning (TODO: gerçek cross-attention + metadata tasarımı)
    encoder_hidden_states = torch.zeros(B, 1, unet.config.cross_attention_dim)
    dummy_metadata = torch.zeros(B, getattr(unet, "num_metadata", 7))

    with torch.no_grad():
        out = unet(
            sample=zt,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
            metadata=dummy_metadata,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_residual,
        )
    eps_hat = out.sample   # NOT: checkpoint v-prediction; bu test sadece sekil kontrolu
    print(f"\neps_hat shape: {eps_hat.shape}")

    # Ldiff (denklem 37/41)
    Ldiff = F.mse_loss(eps_hat, eps)
    print(f"Ldiff: {Ldiff.item():.4f}")

    print("\nStage 8 (provisional) tamam!")
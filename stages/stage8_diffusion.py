import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_PATH = "/Volumes/KIOXIA/LCIB_checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"


class ConditioningProjector(nn.Module):
    """
    Stage 8 — GEÇİCİ / PROVISIONAL conditioning injection.

    Fout (Stage 7 çıktısı, [B,4,64,64]) SatUNet'in mid_block'una
    (1280 kanal, 8x8 çözünürlük — 64/8=8, 3 stride-2 downsample) enjekte
    edilecek şekilde projekte edilir.
    3 kademeli stride-2 conv (64→32→16→8 çözünürlük, 4→320→640→1280 kanal) ile downsample, sonra zero_conv (1×1 conv).

    ControlNet konvansiyonu: son katman SIFIR init edilir. Böylece eğitim
    başında bu ek conditioning UNet'in pretrained davranışını bozmaz,
    gradyanlar aktığında kademeli olarak devreye girer.

    """
    def __init__(self, in_channels=4, mid_channels=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 320, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(320, 640, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(640, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.zero_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        # DÜZELTME: tam sıfır yerine çok küçük random init — Ldiff'ten Minfo'ya
        # giden gradyan sinyalinin, bu katman "ısınmayı" beklemeden en baştan
        # itibaren akmasını sağlamak için (Lmin'e karşı denge kuvvetini güçlendirir)
        nn.init.normal_(self.zero_conv.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, Fout):
        """
        Fout: [B, 4, 64, 64] → mid_block_additional_residual: [B, 1280, 8, 8]
        """
        h = self.net(Fout)
        return self.zero_conv(h)


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

    # Conditioning projector (Fout → mid_block_additional_residual)
    projector = ConditioningProjector(in_channels=4, mid_channels=1280)
    mid_residual = projector(Fout)
    print(f"mid_block_additional_residual shape: {mid_residual.shape}")  # [2,1280,8,8]

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
            mid_block_additional_residual=mid_residual,
        )
    eps_hat = out.sample
    print(f"\neps_hat shape: {eps_hat.shape}")

    # Ldiff (denklem 37/41)
    Ldiff = F.mse_loss(eps_hat, eps)
    print(f"Ldiff: {Ldiff.item():.4f}")

    print("\nStage 8 (provisional) tamam!")
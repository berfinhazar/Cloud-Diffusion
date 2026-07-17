import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Yardımcı: kapalı formül z0 tahmini ─────────────────────────
def predict_z0(zt, eps_hat, t, alphas_cumprod):
    """
    DDPM kapalı formülü: zt = sqrt(αt) z0 + sqrt(1-αt) ε
      ⇒ ẑ0 = (zt - sqrt(1-αt) ε̂) / sqrt(αt)

    Eğitim sırasında tam iteratif sampling (denklem 38-39) yerine
    tek adımda z0 tahmini — Lrec ve Lpreserve gibi pixel-space
    loss'ları için pratik/standart bir yaklaşım.

    zt, eps_hat: [B, 4, 64, 64]
    t          : [B]
    alphas_cumprod: [T]
    Returns: z0_hat [B, 4, 64, 64]
    """
    sqrt_at   = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_1_at = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    z0_hat = (zt - sqrt_1_at * eps_hat) / sqrt_at
    return z0_hat


# ─── 3.1 Reconstruction Loss (denklem 40) ───────────────────────
def reconstruction_loss(Isyn_pred, Igt):
    """
    Lrec = ||Isyn - Igt||_1

    Isyn_pred, Igt: [B, 3, 512, 512]
    """
    return F.l1_loss(Isyn_pred, Igt)


# ─── 3.2 Diffusion Noise Loss (denklem 41) ──────────────────────
def diffusion_loss(eps, eps_hat):
    """
    Ldiff = ||ε - εθ||_2^2
    """
    return F.mse_loss(eps_hat, eps)


# ─── 3.3 Laplacian Preservation Loss (denklem 42) ───────────────
_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]]
).view(1, 1, 3, 3)


def _laplacian(x):
    """
    x: [B, C, H, W] → ∇²x: [B, C, H, W]
    Her kanala bağımsız (depthwise) 3x3 Laplacian filtresi uygular.
    """
    B, C, H, W = x.shape
    kernel = _LAPLACIAN_KERNEL.to(x.device, x.dtype).repeat(C, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=C)


def preserve_loss(Isyn_pred, Iref, Mc, eps=1e-6):
    """
    Lpreserve = || ∇²Isyn - ∇²Iref ||_1,  Ω ∉ Mc  (bulut dışı bölgeler)

    Isyn_pred, Iref: [B, 3, 512, 512]
    Mc             : [B, 1, 512, 512] — binary cloud mask (1=bulut)
    """
    lap_pred = _laplacian(Isyn_pred)
    lap_ref  = _laplacian(Iref)

    non_cloud_mask = (1.0 - Mc)          # Ω ∉ Mc
    non_cloud_mask = non_cloud_mask.expand_as(lap_pred)   # [B,1,H,W] → [B,3,H,W]

    diff = (lap_pred - lap_ref).abs() * non_cloud_mask
    return diff.sum() / (non_cloud_mask.sum() + eps)


# ─── 3.4 Directional Alignment Loss (denklem 43) ────────────────
def directional_loss(d, meta, eps=1e-6):
    """
    Ldir = 1 - (d·va) / (||d|| ||va|| + ε)

    d   : [B, 2] — Stage 3 displacement
    meta: [B, 8] — meta[:,0:2] = (cos_az, sin_az) = va
    """
    va = meta[:, 0:2]
    dot = (d * va).sum(dim=-1)
    denom = d.norm(dim=-1) * va.norm(dim=-1) + eps
    return (1.0 - dot / denom).mean()


# ─── 3.5 Mask Sparsity Loss (denklem 44) ────────────────────────
def mask_sparsity_loss(Minfo):
    """
    Lmin = ||Minfo||_1   (ortalama olarak uygulanır, bkz. yukarıdaki not)

    Minfo: [B, 1, 64, 64] — [0,1] aralığında (Stage 5 çıktısı)
    """
    return Minfo.abs().mean()


# ─── 3.6 Elevation Consistency Loss (denklem 45) ────────────────
def elevation_consistency_loss(d, meta, s=15.0, f=8.0, eps=1e-6):
    """
    Lele = | ||d||_2 - h / ((tan(θe) + ε) · s · f) |

    d   : [B, 2] — Stage 3 displacement (latent piksel)
    meta: [B, 8] — theta_e ve h, encode edilmiş değerlerden geri çıkarılır:
      theta_e = atan2(sin_el, cos_el)   [meta[:,2], meta[:,3]]
      h       = h_norm * (3500-600) + 600   [meta[:,6]]
    s = 15 m (simülatör GSD), f = 8 (VAE downscaling faktörü)
    """
    sin_el, cos_el = meta[:, 2], meta[:, 3]
    theta_e = torch.atan2(sin_el, cos_el)

    h_norm = meta[:, 6]
    h = h_norm * (3500.0 - 600.0) + 600.0

    d_mag = d.norm(dim=-1)
    expected = h / ((torch.tan(theta_e) + eps) * s * f)

    return (d_mag - expected).abs().mean()


# ─── 3.7 Final Objective (denklem 46) ───────────────────────────
def total_loss(Ldiff, Lrec, Lpreserve, Ldir, Lmin, Lele,
               lambdas=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
    """
    Ltotal = λ1 Ldiff + λ2 Lrec + λ3 Lpreserve + λ4 Ldir + λ5 Lmin + λ6 Lele

    lambdas: rapor kesin katsayı vermiyor — şimdilik hepsi 1.0,
    eğitim başladığında tune edilecek (TODO: Amir Hoca ile netleştir).
    """
    l1, l2, l3, l4, l5, l6 = lambdas
    return (l1 * Ldiff + l2 * Lrec + l3 * Lpreserve +
            l4 * Ldir + l5 * Lmin + l6 * Lele)


# ─── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KINGSTON/LCIB_DiffusionSat/LCIB_project/stages")

    from dataset import LCIBDataset
    from stage1_encoder import TerrainEncoder, VAEEncoder
    from stage3_displacement import DisplacementMLP, SpatialWarp
    from stage5_automask import AutoMaskModule
    from stage6_local_attention import WindowLocalAttention
    from stage7_film import FiLMModulation
    from stage8_diffusion import forward_diffusion, ConditioningProjector
    from sat_unet import SatUNet
    from torch.utils.data import DataLoader
    from diffusers import DDPMScheduler

    CHECKPOINT_PATH = "/Volumes/KINGSTON/LCIB_checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"

    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=2, shuffle=False)
    batch   = next(iter(loader))

    Iref = batch["Iref"]
    Igt  = batch["Isyn"]   # ground truth bulutlu görüntü
    Mc   = batch["Mc"]
    meta = batch["meta"]

    print("Encoderlar yükleniyor...")
    terrain_enc = TerrainEncoder()
    vae_enc     = VAEEncoder()

    _, F_terrain = terrain_enc(Iref)
    zc = vae_enc.encode(Mc)
    z0 = vae_enc.encode(Igt)

    disp_mlp = DisplacementMLP(hidden_dim=128)
    d = disp_mlp(meta)
    warp = SpatialWarp()
    Fwarp = warp(zc, d)

    amm = AutoMaskModule()
    Minfo, Ffused = amm(Fwarp, F_terrain)

    local_attn = WindowLocalAttention(
        fused_channels=4, terrain_channels=512,
        d_model=64, num_heads=4, window_size=8
    )
    Fattn = local_attn(Ffused, F_terrain["F3"])

    film = FiLMModulation(fused_channels=4, hidden_dim=64)
    copa = meta[:, 5]
    Fout, _, _ = film(copa, Fattn)

    print("Noise scheduler ve UNet yükleniyor...")
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    B = z0.shape[0]
    t = torch.randint(0, scheduler.config.num_train_timesteps, (B,))
    zt, eps = forward_diffusion(z0, t, scheduler.alphas_cumprod)

    projector = ConditioningProjector(in_channels=4, mid_channels=1280)
    mid_residual = projector(Fout)

    unet = SatUNet.from_pretrained(CHECKPOINT_PATH, subfolder="unet")
    unet.eval()

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

    # ─── Stage 9: Loss hesaplama ───
    print("\nLosslar hesaplanıyor...")

    z0_hat = predict_z0(zt, eps_hat, t, scheduler.alphas_cumprod)
    with torch.no_grad():
        Isyn_pred = vae_enc.decode(z0_hat)   # [-1,1] aralığında, Iref/Igt ile aynı normalize

    Ldiff     = diffusion_loss(eps, eps_hat)
    Lrec      = reconstruction_loss(Isyn_pred, Igt)
    Lpreserve = preserve_loss(Isyn_pred, Iref, Mc)
    Ldir      = directional_loss(d, meta)
    Lmin      = mask_sparsity_loss(Minfo)
    Lele      = elevation_consistency_loss(d, meta)

    Ltotal = total_loss(Ldiff, Lrec, Lpreserve, Ldir, Lmin, Lele)

    print(f"Ldiff     : {Ldiff.item():.4f}")
    print(f"Lrec      : {Lrec.item():.4f}")
    print(f"Lpreserve : {Lpreserve.item():.4f}")
    print(f"Ldir      : {Ldir.item():.4f}")
    print(f"Lmin      : {Lmin.item():.4f}")
    print(f"Lele      : {Lele.item():.4f}")
    print(f"Ltotal    : {Ltotal.item():.4f}")

    print("\nStage 9 (loss functions) tamam!")
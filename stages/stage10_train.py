import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_PATH = "/Volumes/KINGSTON/LCIB_checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"


class EmbeddingProjector(nn.Module):
    """
    YENİ: Stage 2'nin metadata embedding'i (em, [B,256]) daha önce
    hiçbir yere bağlanmıyordu — üretiliyor ama kullanılmıyordu.

    Bu katman em'i UNet'in cross-attention girişine (encoder_hidden_states,
    cross_attention_dim=1024) uygun hale getirir. Tek token olarak veriyoruz
    ([B,1,1024]) — rapor bunun için bir token sayısı belirtmiyor, en basit
    seçenek tek token; birden fazla token (örn. her metadata bileşeni için
    ayrı token) daha zengin bir conditioning olurdu ama şimdilik basit tutuyoruz.
    """
    def __init__(self, embed_dim=256, cross_attention_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, cross_attention_dim),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, cross_attention_dim),
        )

    def forward(self, em):
        """
        em: [B, 256] → encoder_hidden_states: [B, 1, 1024]
        """
        h = self.proj(em)          # [B, 1024]
        return h.unsqueeze(1)       # [B, 1, 1024]


class LCIBPipeline(nn.Module):
    """
    Stage 1-8'in tüm trainable modüllerini tek bir nn.Module'de toplar.
    Böylece optimizer tüm parametreleri (frozen olanlar hariç) tek seferde
    görebilir ve birden fazla training step boyunca aynı ağırlıklar korunur
    (her stage dosyasının kendi test bloğunda yaptığı gibi her seferinde
    yeniden init etmek yerine).

    FROZEN (eğitilmeyecek):
      - TerrainEncoder, VAEEncoder (zaten rapor gereği frozen)
      - SatUNet (pretrained backbone — ControlNet eğitim paradigmasına
        uygun olarak dondurduk: sadece bizim eklediğimiz conditioning
        modülleri eğitiliyor, ana model bozulmuyor. Gerçek ölçekli eğitimde
        GPU'ya geçilince SatUNet'in de fine-tune edilip edilmeyeceği
        Amir Hoca ile netleştirilmeli.)

    TRAINABLE:
      - MetadataEmbedder (Stage 2)
      - DisplacementMLP (Stage 3)
      - AutoMaskModule (Stage 5)
      - WindowLocalAttention (Stage 6)
      - FiLMModulation (Stage 7)
      - ConditioningProjector (Stage 8, mid_block injection)
      - EmbeddingProjector (YENİ — em'i cross-attention'a bağlıyor)
    """
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        super().__init__()
        # Lokal importlar — dairesel bağımlılığı önlemek için burada
        from stage1_encoder import TerrainEncoder, VAEEncoder
        from stage2_metadata import MetadataEmbedder
        from stage3_displacement import DisplacementMLP, SpatialWarp
        from stage5_automask import AutoMaskModule
        from stage6_local_attention import WindowLocalAttention
        from stage7_film import FiLMModulation
        from stage8_diffusion import ConditioningProjector
        from sat_unet import SatUNet

        # Frozen
        self.terrain_enc = TerrainEncoder(checkpoint_path)
        self.vae_enc     = VAEEncoder(checkpoint_path)
        self.unet = SatUNet.from_pretrained(checkpoint_path, subfolder="unet")
        self.unet.eval()   # frozen: stokastik katman davranışı (varsa dropout) kapalı kalsın
        for p in self.unet.parameters():
            p.requires_grad = False

        # Trainable
        self.metadata_embedder = MetadataEmbedder(embed_dim=256)
        self.disp_mlp          = DisplacementMLP(hidden_dim=128)
        self.warp               = SpatialWarp()   # parametresiz, ama forward'da kullanılıyor
        self.amm                = AutoMaskModule()
        self.local_attn         = WindowLocalAttention(
            fused_channels=4, terrain_channels=512,
            d_model=64, num_heads=4, window_size=8
        )
        self.film                = FiLMModulation(fused_channels=4, hidden_dim=64)
        self.cond_projector     = ConditioningProjector(in_channels=4, mid_channels=1280)
        self.embed_projector    = EmbeddingProjector(
            embed_dim=256, cross_attention_dim=self.unet.config.cross_attention_dim
        )

    def trainable_parameters(self):
        modules = [
            self.metadata_embedder, self.disp_mlp, self.amm,
            self.local_attn, self.film, self.cond_projector, self.embed_projector,
        ]
        for m in modules:
            for p in m.parameters():
                yield p

    def forward(self, batch, t=None, scheduler=None):
        """
        batch: dataset'ten gelen dict (Iref, Isyn, Mc, meta)
        t     : [B] belirli timestep'ler (None ise scheduler'dan rastgele seçilir)

        Returns: dict — loss hesaplamak için gereken tüm ara çıktılar
        """
        Iref = batch["Iref"]
        Igt  = batch["Isyn"]
        Mc   = batch["Mc"]
        meta = batch["meta"]
        B    = Iref.shape[0]

        with torch.no_grad():
            _, F_terrain = self.terrain_enc(Iref)
            zc = self.vae_enc.encode(Mc)
            z0 = self.vae_enc.encode(Igt)

        # Stage 2 — artık gerçekten kullanılıyor
        em = self.metadata_embedder(meta)                  # [B, 256]
        encoder_hidden_states = self.embed_projector(em)    # [B, 1, 1024]

        # Stage 3-4
        d = self.disp_mlp(meta)
        Fwarp = self.warp(zc, d)

        # Stage 5-7
        Minfo, Ffused = self.amm(Fwarp, F_terrain)
        Fattn = self.local_attn(Ffused, F_terrain["F3"])
        Fout, _, _ = self.film(meta, Fattn)

        # Forward diffusion
        if t is None:
            t = torch.randint(0, scheduler.config.num_train_timesteps, (B,))
        eps = torch.randn_like(z0)
        sqrt_at   = scheduler.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1_at = (1 - scheduler.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        zt = sqrt_at * z0 + sqrt_1_at * eps

        # Stage 8 — hem mid_block residual hem de cross-attention artık dolu
        mid_residual = self.cond_projector(Fout)
        dummy_native_metadata = torch.zeros(B, getattr(self.unet, "num_metadata", 7))

        out = self.unet(
            sample=zt,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
            metadata=dummy_native_metadata,
            mid_block_additional_residual=mid_residual,
        )
        eps_hat = out.sample

        return {
            "eps": eps, "eps_hat": eps_hat, "zt": zt, "t": t,
            "d": d, "Minfo": Minfo, "meta": meta,
            "Iref": Iref, "Igt": Igt, "Mc": Mc, "z0": z0,
        }


# ─── TEST / SANITY EĞİTİM ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KINGSTON/LCIB_DiffusionSat/LCIB_project/stages")

    from dataset import LCIBDataset
    from stage9_losses import (
        predict_z0, reconstruction_loss, diffusion_loss,
        preserve_loss, directional_loss, mask_sparsity_loss,
        elevation_consistency_loss, total_loss,
    )
    from torch.utils.data import DataLoader
    from diffusers import DDPMScheduler

    # DÜZELTME: λ6 (Lele katsayısı) küçültüldü. Lele diğer loss'lardan
    # ~40x büyük ölçekte çıkıyordu (latent piksel biriminde ham fark).
    # 0.025 ile başlangıçta diğer loss'larla benzer büyüklüğe geliyor
    # (40 * 0.025 ≈ 1.0). Eğitim ilerledikçe tekrar ayarlanabilir.
    LAMBDAS = (1.0, 1.0, 1.0, 1.0, 1.0, 0.025)   # λ1..λ6

    print("Dataset yükleniyor...")
    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=2, shuffle=True)

    # SANITY TEST: aynı küçük batch üzerinde overfit deneyip loss'un
    # düştüğünü doğruluyoruz — CPU'da tam eğitim pratik değil, ama bu
    # pipeline'ın gerçekten öğrenebildiğini kanıtlamak için yeterli.
    fixed_batch = next(iter(loader))

    print("\nModel yükleniyor (biraz sürebilir)...")
    model = LCIBPipeline()
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)

    N_STEPS = 15
    print(f"\n{N_STEPS} adımlık overfit sanity testi başlıyor (sabit batch)...\n")

    for step in range(N_STEPS):
        optimizer.zero_grad()

        out = model(fixed_batch, scheduler=scheduler)

        z0_hat = predict_z0(out["zt"], out["eps_hat"], out["t"], scheduler.alphas_cumprod)
        with torch.no_grad():
            Isyn_pred = model.vae_enc.decode(z0_hat)

        Ldiff     = diffusion_loss(out["eps"], out["eps_hat"])
        Lrec      = reconstruction_loss(Isyn_pred, out["Igt"])
        Lpreserve = preserve_loss(Isyn_pred, out["Iref"], out["Mc"])
        Ldir      = directional_loss(out["d"], out["meta"])
        Lmin      = mask_sparsity_loss(out["Minfo"])
        Lele      = elevation_consistency_loss(out["d"], out["meta"])

        Ltotal = total_loss(Ldiff, Lrec, Lpreserve, Ldir, Lmin, Lele, lambdas=LAMBDAS)

        Ltotal.backward()
        optimizer.step()

        print(f"step {step+1:2d}/{N_STEPS} | Ltotal={Ltotal.item():.4f} "
              f"| Ldiff={Ldiff.item():.4f} Lrec={Lrec.item():.4f} "
              f"Lpreserve={Lpreserve.item():.4f} Ldir={Ldir.item():.4f} "
              f"Lmin={Lmin.item():.4f} Lele={Lele.item():.4f}")

    print("\nSanity eğitim testi tamam! Ltotal'ın genel eğiliminin düştüğünü kontrol et.")
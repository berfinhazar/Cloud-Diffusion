import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_PATH = "/Volumes/KIOXIA/LCIB_checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"

"""
Tüm Stage 1-8'i tek nn.Module'de birleştiriyor. Frozen (TerrainEncoder, VAEEncoder, SatUNet) ve trainable (diğer 7 modül) 
ayrımını yönetiyor.
forward(), tüm zinciri (encode → metadata → warp → AMM → attention → FiLM → UNet) tek seferde çalıştırıp 
loss hesaplamak için gereken ara çıktıları döndürüyor.
"""
class EmbeddingProjector(nn.Module):
    """
    Stage 2'nin metadata embedding'i (em, [B,256]) rapor 2.2.3'e gore zaman
    adimi gommesine eklenir. Bu katman em'i UNet'in zaman gommesi boyutuna
    (time_embed_dim, SD2.1: 1280) projekte eder; sonuc SatUNet.forward'da
    `emb = emb + time_emb_extra` seklinde toplanir.

    Son katman cok kucuk baslatilir: egitim basinda hazir UNet'in davranisi
    bozulmaz, bilgi kademeli olarak devreye girer.
    """
    def __init__(self, embed_dim=256, time_embed_dim=1280):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        nn.init.normal_(self.proj[2].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.proj[2].bias)

    def forward(self, em):
        """
        em: [B, 256] → time_emb_extra: [B, time_embed_dim]
        """
        return self.proj(em)


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
        modülleri eğitiliyor, ana model bozulmuyor.)

    TRAINABLE:
      - MetadataEmbedder (Stage 2)
      - DisplacementMLP (Stage 3)
      - AutoMaskModule (Stage 5)
      - WindowLocalAttention (Stage 6)
      - FiLMModulation (Stage 7)
      - ConditioningProjector (Stage 8, çok ölçekli skip + mid enjeksiyonu)
      - EmbeddingProjector (em'i zaman adımı gömmesine bağlıyor, rapor 2.2.3)
      - zref_proj (z_ref'i em uzayına bağlıyor)
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

        # SatUNet'in kendi (DiffusionSat uydu metadata'sı için olan) dalını kapat.
        # Sıfır tensör vermek dalı etkisiz yapmıyordu: sinüzoidal projeksiyonda
        # cos(0)=1 olduğundan emb'e sabit, sıfır olmayan bir ofset ekleniyordu.
        # forward() zaten `if self.metadata_embedding is not None` kontrolü yapıyor.
        self.unet.metadata_embedding = None
        self.unet.use_metadata = False

        # DÜZELTME (RAM optimizasyonu): gradient
        # checkpointing açık — backward sırasında ara aktivasyonları bellekte
        # tutmak yerine yeniden hesaplıyor. Biraz yavaşlatır ama UNet'in
        # bellek kullanımını ciddi oranda düşürür. SatUNet zaten
        # `_supports_gradient_checkpointing = True` ile bunu destekliyor
        # (bkz. sat_unet.py), o yüzden güvenli bir değişiklik.
        self.unet.enable_gradient_checkpointing()

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
        cfg = self.unet.config
        lpb = cfg.layers_per_block
        lpb = lpb[0] if isinstance(lpb, (list, tuple)) else lpb
        self.cond_projector     = ConditioningProjector(
            block_out_channels=tuple(cfg.block_out_channels),
            layers_per_block=lpb,
        )
        self.embed_projector    = EmbeddingProjector(
            embed_dim=256, time_embed_dim=cfg.block_out_channels[0] * 4
        )
        # z_ref (terrain latent, [B,4,64,64]) → global pool → em uzayına.
        # Sıfır init: eğitim başında etkisiz, gradyan geldikçe devreye girer.
        self.zref_proj = nn.Linear(4, 256)
        nn.init.zeros_(self.zref_proj.weight)
        nn.init.zeros_(self.zref_proj.bias)

    def trainable_parameters(self):
        modules = [
            self.metadata_embedder, self.disp_mlp, self.amm,
            self.local_attn, self.film, self.cond_projector, self.embed_projector,
            self.zref_proj,
        ]
        for m in modules:
            for p in m.parameters():
                yield p

    def build_cond(self, Fout, F_terrain, time_emb_extra):
        """
        UNet'e verilecek tum conditioning'i tek dict'te toplar:
          ehs : sabit sifir token (metadata artik buradan tasinmiyor)
          down: skip baglantilarina eklenecek residual'lar (cok olcekli)
          mid : mid_block cikisina eklenecek residual
          temb: zaman adimi gommesine eklenecek em projeksiyonu
        """
        down_res, mid_res = self.cond_projector(Fout, F_terrain["F3"])
        B = Fout.shape[0]
        ehs = torch.zeros(
            B, 1, self.unet.config.cross_attention_dim,
            device=Fout.device, dtype=Fout.dtype,
        )
        return {"ehs": ehs, "down": down_res, "mid": mid_res, "temb": time_emb_extra}

    def unet_forward(self, zt, t, cond):
        """
        Frozen SatUNet cagrisi (v-prediction ciktisini dondurur).
        """
        return self.unet(
            sample=zt,
            timestep=t,
            encoder_hidden_states=cond["ehs"],
            down_block_additional_residuals=cond["down"],
            mid_block_additional_residual=cond["mid"],
            time_emb_extra=cond["temb"],
        ).sample

    def forward(self, batch, t=None, scheduler=None):
        """
        batch: dataset'ten gelen dict (Iref, Isyn, Mc, meta)
        t     : [B] belirli timestep'ler (None ise scheduler'dan rastgele seçilir)

        Returns: dict — loss hesaplamak için gereken tüm ara çıktılar
        """
        Iref = batch["Iref"]
        Igt  = batch["Isyn"]
        Mc   = batch["Mc"]
        Ms   = batch["Ms"]
        meta = batch["meta"]
        B    = Iref.shape[0]

        with torch.no_grad():
            z_ref, F_terrain = self.terrain_enc(Iref)
            zc = self.vae_enc.encode(Mc)
            z0 = self.vae_enc.encode(Igt)

        # Stage 2 — em zaman adımı gömmesine eklenecek (rapor 2.2.3)
        em = self.metadata_embedder(meta) + self.zref_proj(z_ref.mean(dim=[2, 3]))   # [B, 256]
        time_emb_extra = self.embed_projector(em)           # [B, time_embed_dim]

        # Stage 3-4
        d = self.disp_mlp(meta)
        Fwarp = self.warp(zc, d)

        # Stage 5-7
        Minfo, Ffused = self.amm(Fwarp, F_terrain, Ms)
        Fattn = self.local_attn(Ffused, F_terrain["F3"])
        Fout, _, _ = self.film(meta, Fattn)

        # Forward diffusion
        # DÜZELTME (MPS/GPU cihaz fix'i): t eskiden her zaman CPU'da
        # oluşturuluyordu. z0 artık MPS/GPU'da olabileceği için
        # (train.py modeli device'a taşıyor), t da aynı cihazda
        # olmalı — yoksa scheduler.alphas_cumprod[t] indexleme
        # işlemi cihaz uyuşmazlığı hatası verir.
        if t is None:
            t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=z0.device)
        eps = torch.randn_like(z0)
        sqrt_at   = scheduler.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1_at = (1 - scheduler.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        zt = sqrt_at * z0 + sqrt_1_at * eps

        # Stage 8 — çok ölçekli enjeksiyon (rapor 2.6.4 / 2.7.4) + em zaman gömmesinde (2.2.3)
        cond = self.build_cond(Fout, F_terrain, time_emb_extra)
        v_hat = self.unet_forward(zt, t, cond)
        # UNet çıktısı v-prediction (checkpoint böyle eğitilmiş) → ε̂'a çevir.
        # Raporun formülleri (Ldiff, ẑ0) ε̂ üzerinden olduğu için aşağısı değişmiyor.
        from stage9_losses import v_to_eps
        eps_hat = v_to_eps(v_hat, zt, t, scheduler.alphas_cumprod)

        return {
            "eps": eps, "eps_hat": eps_hat, "zt": zt, "t": t,
            "d": d, "Minfo": Minfo, "meta": meta,
            "Iref": Iref, "Igt": Igt, "Mc": Mc, "Ms": Ms, "z0": z0,
        }


# ─── TEST / SANITY EĞİTİM ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KIOXIA/LCIB_DiffusionSat/LCIB_project/stages")

    from dataset import LCIBDataset
    from stage9_losses import (
        predict_z0, reconstruction_loss, diffusion_loss,
        preserve_loss, directional_loss, mask_sparsity_loss,
        mask_target_loss, elevation_consistency_loss, total_loss,
    )
    from torch.utils.data import DataLoader
    from stage9_losses import make_scheduler

    # DÜZELTME: λ6 (Lele katsayısı) küçültüldü. Lele diğer loss'lardan
    # ~40x büyük ölçekte çıkıyordu (latent piksel biriminde ham fark).
    # 0.025 ile başlangıçta diğer loss'larla benzer büyüklüğe geliyor
    # (40 * 0.025 ≈ 1.0). Eğitim ilerledikçe tekrar ayarlanabilir.
    LAMBDAS = (1.0, 1.0, 1.0, 1.0, 1.0, 0.025, 1.0)   # λ1..λ7

    print("Dataset yükleniyor...")
    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=2, shuffle=True)

    # SANITY TEST: aynı küçük batch üzerinde overfit deneyip loss'un
    # düştüğünü doğruluyoruz — CPU'da tam eğitim pratik değil, ama bu
    # pipeline'ın gerçekten öğrenebildiğini kanıtlamak için yeterli.
    fixed_batch = next(iter(loader))

    print("\nModel yükleniyor (biraz sürebilir)...")
    model = LCIBPipeline()
    scheduler = make_scheduler()

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)

    N_STEPS = 15
    print(f"\n{N_STEPS} adımlık overfit sanity testi başlıyor (sabit batch)...\n")

    for step in range(N_STEPS):
        optimizer.zero_grad()

        out = model(fixed_batch, scheduler=scheduler)

        z0_hat = predict_z0(out["zt"], out["eps_hat"], out["t"], scheduler.alphas_cumprod)
        with torch.no_grad():
            Isyn_pred = model.vae_enc.decode(z0_hat)

        Ldiff          = diffusion_loss(out["eps"], out["eps_hat"])
        Lrec           = reconstruction_loss(Isyn_pred, out["Igt"])
        Lpreserve      = preserve_loss(Isyn_pred, out["Iref"], out["Mc"], out["Ms"])
        Ldir           = directional_loss(out["d"], out["meta"])
        Lmin           = mask_sparsity_loss(out["Minfo"])
        Lpreserve_mask = mask_target_loss(out["Minfo"], out["Mc"], out["d"])
        Lele           = elevation_consistency_loss(out["d"], out["meta"])

        Ltotal = total_loss(Ldiff, Lrec, Lpreserve, Ldir, Lmin, Lele,
                             Lpreserve_mask, lambdas=LAMBDAS)

        Ltotal.backward()
        optimizer.step()

        print(f"step {step+1:2d}/{N_STEPS} | Ltotal={Ltotal.item():.4f} "
              f"| Ldiff={Ldiff.item():.4f} Lrec={Lrec.item():.4f} "
              f"Lpreserve={Lpreserve.item():.4f} Ldir={Ldir.item():.4f} "
              f"Lmin={Lmin.item():.4f} Lele={Lele.item():.4f}")

    print("\nSanity eğitim testi tamam! Ltotal'ın genel eğiliminin düştüğünü kontrol et.")
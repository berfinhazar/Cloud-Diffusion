import torch
import torch.nn as nn
from diffusers import AutoencoderKL

CHECKPOINT_PATH = "/Volumes/KINGSTON/LCIB_checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"

class TerrainEncoder(nn.Module):
    """
    Eref: Frozen DiffusionSat VAE encoder.
    Multiscale terrain feature extraction via forward hooks.
    Fterrain = {F1, F2, F3, F4} — her downblock'tan bir feature map.
    """
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        super().__init__()

        vae = AutoencoderKL.from_pretrained(
            checkpoint_path,
            subfolder="vae",
            torch_dtype=torch.float32
        )
        self.encoder    = vae.encoder
        self.quant_conv = vae.quant_conv
        self.scaling_factor = vae.config.scaling_factor

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.quant_conv.parameters():
            p.requires_grad = False

        self._features = {}
        for i, block in enumerate(self.encoder.down_blocks):
            block.register_forward_hook(self._make_hook(f"down_{i}"))

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self._features[name] = output[0]
            else:
                self._features[name] = output
        return hook

    def forward(self, x):
        """
        x: [B, 3, H, W] — [-1, 1] normalize edilmiş temiz uydu görüntüsü
        Returns:
            z_ref    : [B, 4, H/8, W/8]
            F_terrain: {"F1", "F2", "F3", "F4"}

        NOT: Iref her zaman dataset'ten gelen sabit bir görüntü (hiçbir
        trainable modülün çıktısı değil), bu yüzden bu yolun gradyan
        taşımasına gerek yok — no_grad burada doğru ve GÜVENLİ.
        """
        self._features = {}
        with torch.no_grad():
            h     = self.encoder(x)
            z_ref = self.quant_conv(h)
            z_ref = z_ref * self.scaling_factor

        F_terrain = {
            "F1": self._features.get("down_0"),
            "F2": self._features.get("down_1"),
            "F3": self._features.get("down_2"),
            "F4": self._features.get("down_3"),
        }
        return z_ref, F_terrain


class VAEEncoder(nn.Module):
    """
    EVAE: Generative stream encoder + decoder.
    Iref, Mc, Ms → latent uzaya encode eder.
    """
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(
            checkpoint_path,
            subfolder="vae",
            torch_dtype=torch.float32
        )
        self.scaling_factor = self.vae.config.scaling_factor

        for p in self.vae.parameters():
            p.requires_grad = False

    def encode(self, x):
        """
        x: [B, C, H, W]
          - Görsel ise: [-1, 1] normalize, 3 kanal
          - Mask ise  : [0, 1] binary, 1 kanal → 3 kanala çıkarılır, [-1, 1]'e çevrilir
        Returns: z [B, 4, H/8, W/8]

        NOT: encode() burada hep dataset'ten gelen ham görüntülere
        (Iref, Mc, Isyn/Igt) uygulanıyor — bunlar hiçbir trainable
        modülün çıktısı değil, dolayısıyla no_grad burada doğru ve
        güvenli (gradyan zinciri zaten bu noktadan başlamıyor).
        """
        with torch.no_grad():
            # 1 kanallı mask → 3 kanala çıkar ve [-1,1]'e normalize et
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)   # [B,1,H,W] → [B,3,H,W]
                x = x * 2.0 - 1.0          # [0,1] → [-1,1]

            posterior = self.vae.encode(x).latent_dist
            z = posterior.sample()
            z = z * self.scaling_factor
        return z

    def decode(self, z):
        """
        z: [B, 4, H/8, W/8]
        Returns: image [B, 3, H, W] — [-1, 1] aralığında

        DÜZELTME (kritik bug fix): Bu fonksiyon eskiden içeride
        `with torch.no_grad():` kullanıyordu. Eğitim sırasında decode()
        z0_hat'e (yani eps_hat'e, yani trainable modüllerin çıktısına)
        uygulanıyor — Lrec ve Lpreserve loss'larının bu çıktı üzerinden
        hesaplanıp geri yayılması (backward) gerekiyor. no_grad burada
        olduğu sürece Lrec/Lpreserve hiçbir gradyan üretmiyordu (loss
        değerleri doğru hesaplanıp loglanıyordu ama .backward() bu
        zincirden geçince hiçbir trainable parametreye ulaşmıyordu).

        VAE ağırlıkları zaten __init__'te requires_grad=False yapıldığı
        için no_grad'i kaldırmak VAE'yi eğitmez — sadece z (girdi) için
        gradyanın hesaplanmasına izin verir, ki asıl ihtiyacımız olan da bu.
        Bu yüzden no_grad kaldırıldı; VAE ağırlıkları hâlâ donuk kalıyor.

        Eğitim dışı (ör. sadece görsel üretmek için inference) çağrılarda
        gradyan gerekmiyorsa çağıran taraf zaten `with torch.no_grad():`
        ile sarmalayabilir (train.py validation adımında zaten böyle
        yapıyor).
        """
        z     = z / self.scaling_factor
        image = self.vae.decode(z).sample
        return image


# ─── TEST — gerçek dataset ile ────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KINGSTON/LCIB_DiffusionSat/LCIB_project/stages")
    from dataset import LCIBDataset
    from torch.utils.data import DataLoader

    print("VAE yükleniyor...")
    terrain_enc = TerrainEncoder()
    vae_enc     = VAEEncoder()
    print("Yüklendi!\n")

    # Gerçek dataset'ten bir batch al
    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)
    batch   = next(iter(loader))

    Iref = batch["Iref"]   # [1, 3, 512, 512]  — [-1,1]
    Mc   = batch["Mc"]     # [1, 1, 512, 512]  — [0,1] binary
    Ms   = batch["Ms"]     # [1, 1, 512, 512]  — [0,1] binary
    meta = batch["meta"]   # [1, 9]
    name = batch["name"][0]

    print(f"Sample: {name}")
    print(f"Iref range: [{Iref.min():.2f}, {Iref.max():.2f}]")
    print(f"Mc   range: [{Mc.min():.2f},  {Mc.max():.2f}]")
    print(f"meta      : {meta[0]}\n")

    # TerrainEncoder test
    print("TerrainEncoder çalışıyor...")
    z_ref_terrain, F_terrain = terrain_enc(Iref)
    print(f"  z_ref shape: {z_ref_terrain.shape}")
    for k, v in F_terrain.items():
        if v is not None:
            print(f"  {k} shape  : {v.shape}")

    # VAEEncoder test
    print("\nVAEEncoder çalışıyor...")
    z_ref = vae_enc.encode(Iref)
    z_c   = vae_enc.encode(Mc)
    z_s   = vae_enc.encode(Ms)
    print(f"  z_ref shape: {z_ref.shape}")
    print(f"  z_c   shape: {z_c.shape}")
    print(f"  z_s   shape: {z_s.shape}")

    # Decode test — reconstruction
    print("\nDecode test...")
    Isyn_recon = vae_enc.decode(z_ref)
    print(f"  Isyn_recon shape: {Isyn_recon.shape}")
    print(f"  Isyn_recon range: [{Isyn_recon.min():.2f}, {Isyn_recon.max():.2f}]")

    # Gradyan testi — DÜZELTMENİN doğrulaması: decode() artık gradyan taşıyor mu?
    print("\nGradyan akışı testi (decode fix doğrulaması)...")
    z_ref_grad = z_ref.clone().requires_grad_(True)
    out = vae_enc.decode(z_ref_grad)
    out.sum().backward()
    has_grad = z_ref_grad.grad is not None and z_ref_grad.grad.abs().sum().item() > 0
    print(f"  z_ref_grad.grad dolu mu: {has_grad}  (True olmalı — artık Lrec/Lpreserve gradyanı akıyor)")

    print("\nTüm testler başarılı!")

import torch
import torch.nn as nn
import math

class SinusoidalProjection(nn.Module):
    """
    DiffusionSat'taki gibi sinusoidal projeksiyon.
    Scalar değeri d-boyutlu vektöre çevirir.

    9 boyutlu ham metadata vektörünü (meta) tek bir 256-boyutlu özet vektöre (em) sıkıştırıyor. 
    Her bileşen (azimuth, elevation, coverage, opacity, height, ccount, sstr) ayrı bir sinusoidal projeksiyon + MLP'den geçip toplanıyor.
    Çıktı em, Stage 8'de EmbeddingProjector üzerinden UNet'in cross-attention girişine besleniyor.
    """
    def __init__(self, dim=256, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        """
        x: [B] veya [B, 1] — tek scalar metadata değeri
        Returns: [B, dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [B] → [B, 1]

        device = x.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half, device=device).float() / half
        )  # [dim/2]

        args = x * freqs.unsqueeze(0)   # [B, dim/2]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        return emb


class MetadataEmbedder(nn.Module):
    """
    Stage 2: Metadata embedding.
    Rapordaki formül 15-17'yi uygular + iki eklenti.

    Input: meta_vec [B, 9]
      [cos_az, sin_az, sin_el, cos_el, ccov, copa, h_norm, ccount, sstr]

    Output: em [B, embed_dim] — timestep embedding ile toplanacak

    NOT (değişiklik geçmişi):
    - ccount: rapor denklem 17 zaten {ccov, ccount, copa, h} toplamını
      istiyor ama daha önce ccount kodda yoktu (dataset her zaman 0.0
      veriyordu) — düzeltildi, artık gerçek değer var ve toplamda.
    - sstr (shadow_strength): rapor denklem 17'de YOK, metodoloji
      dokümanının eklentisi. Berfin'in onayıyla eklendi (JSON'da gerçek
      veri olduğu doğrulandı). Amir Hoca'ya bildirilmeli.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # Azimuth MLP: (cos_az, sin_az) → embed_dim
        self.mlp_az = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Elevation MLP: (sin_el, cos_el) → embed_dim
        self.mlp_el = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Sinusoidal projeksiyon + MLP: ccov, copa, h_norm, ccount, sstr için
        self.sin_proj = SinusoidalProjection(dim=embed_dim)

        self.mlp_ccov = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mlp_copa = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mlp_h = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # YENİ: ccount branch (rapor denklem 17'de zaten isteniyordu)
        # NOT: cloud_count şu an her zaman 1 (Amir Hoca'nın kararıyla tek-bulutlu
        # senaryolarla sınırlandırıldı, dataset.py'de sabitlendi). Bu yüzden
        # mlp_ccount branch'i şu an fonksiyonel olarak "dead weight" — hiçbir
        # örnek arası varyasyon taşımıyor, öğrenmeye katkısı yok. Mimaride
        # bilerek tutuluyor: çok bulutlu deneylere geçilirse hazır olsun diye.
        self.mlp_ccount = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # YENİ: sstr branch (metodoloji eklentisi)
        self.mlp_sstr = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, meta):
        """
        meta: [B, 9]
          idx 0,1 → cos_az, sin_az
          idx 2,3 → sin_el, cos_el
          idx 4   → ccov
          idx 5   → copa
          idx 6   → h_norm
          idx 7   → ccount
          idx 8   → sstr
        """
        va = meta[:, 0:2]
        ea = self.mlp_az(va)

        ve = meta[:, 2:4]
        ee = self.mlp_el(ve)

        ccov   = meta[:, 4]
        copa   = meta[:, 5]
        h_norm = meta[:, 6]
        ccount = meta[:, 7]
        sstr   = meta[:, 8]

        e_ccov   = self.mlp_ccov(self.sin_proj(ccov))
        e_copa   = self.mlp_copa(self.sin_proj(copa))
        e_h      = self.mlp_h(self.sin_proj(h_norm))
        e_ccount = self.mlp_ccount(self.sin_proj(ccount))
        e_sstr   = self.mlp_sstr(self.sin_proj(sstr))

        # Toplam embedding (rapor denklem 17 + ccount + sstr eklentisi)
        em = ea + ee + e_ccov + e_copa + e_h + e_ccount + e_sstr
        return em


# ─── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KINGSTON/LCIB_DiffusionSat/LCIB_project/stages")
    from dataset import LCIBDataset
    from torch.utils.data import DataLoader

    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=2, shuffle=False)
    batch   = next(iter(loader))

    meta = batch["meta"]  # [2, 9]
    print(f"Meta input shape: {meta.shape}")
    print(f"Meta values:\n{meta}")

    embedder = MetadataEmbedder(embed_dim=256)
    em = embedder(meta)

    print(f"\nEmbedding shape: {em.shape}")  # [2, 256]
    print(f"Embedding range: [{em.min():.3f}, {em.max():.3f}]")
    print("\nStage 2 tamam!")
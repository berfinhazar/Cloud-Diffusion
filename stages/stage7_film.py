import torch
import torch.nn as nn


class FiLMModulation(nn.Module):
    """
    Stage 7: Opacity-Based FiLM Conditioning.
    Rapor denklem 30-31 + sstr eklentisi.

    γ, β = MLPfilm(copa, sstr)     (30 — genişletilmiş)
    Fout = γ ⊙ Fattn + β           (31)
    copa (opacity) ve sstr (shadow strength) metadata'sından γ, β üretip Fattn'i FiLM (Feature-wise Linear Modulation)
      ile ölçekliyor: Fout = γ ⊙ Fattn + β.
    Not: Rapor sadece copa diyordu, sstr metodoloji eklentisi olarak eklendi.
    Girdi: copa ∈ [0.85, 1.00], sstr ∈ [0, 1] (dataset.py'de meta[:,5] ve meta[:,8]).
    """
    def __init__(self, fused_channels=4, hidden_dim=64):
        super().__init__()
        self.fused_channels = fused_channels

        # Girdi artık 2 boyutlu: (copa, sstr)
        self.mlp_film = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * fused_channels),
        )

    def forward(self, meta, Fattn):
        """
        meta : [B, 9] — copa = meta[:,5], sstr = meta[:,8] buradan çekilir
        Fattn: [B, C, H, W] — Stage 6 çıktısı

        Returns:
            Fout : [B, C, H, W]
            gamma: [B, C]
            beta : [B, C]
        """
        B, C, H, W = Fattn.shape
        assert C == self.fused_channels, "Fattn kanal sayısı fused_channels ile uyuşmuyor"

        copa = meta[:, 5]
        sstr = meta[:, 8]
        film_input = torch.stack([copa, sstr], dim=-1)   # [B, 2]

        gamma_beta = self.mlp_film(film_input)   # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma_b = gamma.view(B, C, 1, 1)
        beta_b  = beta.view(B, C, 1, 1)

        Fout = gamma_b * Fattn + beta_b
        return Fout, gamma, beta


# ─── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KIOXIA/LCIB_DiffusionSat/LCIB_project/stages")
    from dataset import LCIBDataset
    from stage1_encoder import TerrainEncoder, VAEEncoder
    from stage3_displacement import DisplacementMLP, SpatialWarp
    from stage5_automask import AutoMaskModule
    from stage6_local_attention import WindowLocalAttention
    from torch.utils.data import DataLoader

    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=2, shuffle=False)
    batch   = next(iter(loader))

    Iref = batch["Iref"]
    Mc   = batch["Mc"]
    Ms   = batch["Ms"]
    meta = batch["meta"]   # artık [2, 9]

    print("Encoderlar yükleniyor...")
    terrain_enc = TerrainEncoder()
    vae_enc     = VAEEncoder()

    _, F_terrain = terrain_enc(Iref)
    zc = vae_enc.encode(Mc)

    disp_mlp = DisplacementMLP(hidden_dim=128)
    d = disp_mlp(meta)
    warp = SpatialWarp()
    Fwarp = warp(zc, d)

    # DÜZELTME: AutoMaskModule artık Ms (gölge maskesi) zorunlu 3. parametre
    # istiyor (bkz. stage5_automask.py, Ms entegrasyonu) — bu test bloğu
    # güncellenmemiş kalmıştı, eksik parametre hatası veriyordu.
    amm = AutoMaskModule()
    Minfo, Ffused = amm(Fwarp, F_terrain, Ms)

    local_attn = WindowLocalAttention(
        fused_channels=4, terrain_channels=512,
        d_model=64, num_heads=4, window_size=8
    )
    Fattn = local_attn(Ffused, F_terrain["F3"])
    print(f"Fattn shape: {Fattn.shape}")

    film = FiLMModulation(fused_channels=4, hidden_dim=64)
    Fout, gamma, beta = film(meta, Fattn)

    print(f"\nFout shape : {Fout.shape}")
    print(f"Fout range : [{Fout.min():.3f}, {Fout.max():.3f}]")
    print(f"gamma      : {gamma}")
    print(f"beta       : {beta}")

    print("\nStage 7 tamam!")
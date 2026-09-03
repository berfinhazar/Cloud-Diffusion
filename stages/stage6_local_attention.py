import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowLocalAttention(nn.Module):
    """
    Stage 6: Window-Based Local Texture Attention.
    Rapor denklem 26-29.

    Ffused (Stage 5) ve Fterrain (F3, Stage 1) 8x8'lik non-overlapping
    window'lara bölünür (denklem 26). Her window içinde cross-attention
    uygulanır: Q = Ffused, K = V = Fterrain (denklem 27). Window'lar
    birleştirilerek Fattn üretilir (denklem 28-29).

    Kanal uyuşmazlığı (Ffused: 4, Fterrain: 512) ortak bir d_model
    boyutuna projekte edilerek çözülür; çıkış tekrar Ffused'in kanal
    sayısına (4) geri projekte edilir.
    """
    def __init__(self, fused_channels=4, terrain_channels=512,
                 d_model=64, num_heads=4, window_size=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model, num_heads'e tam bölünmeli"

        self.window_size = window_size
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.head_dim      = d_model // num_heads

        # Q: Ffused → d_model
        self.q_proj = nn.Conv2d(fused_channels, d_model, kernel_size=1)
        # K, V: Fterrain → d_model
        self.k_proj = nn.Conv2d(terrain_channels, d_model, kernel_size=1)
        self.v_proj = nn.Conv2d(terrain_channels, d_model, kernel_size=1)
        # Çıkış: d_model → fused_channels (Fattn, Ffused ile aynı shape)
        self.out_proj = nn.Conv2d(d_model, fused_channels, kernel_size=1)

        # Sayısal stabilite: attention çıktısını (d_model kanalı) out_proj'dan
        # önce normalize et. Rapor bunu belirtmiyor ama diffusion UNet
        # mimarilerinde attention çıktısını projeksiyondan önce normalize
        # etmek standart pratik — ölçeği kontrolsüz büyüyen değerlerin
        # (bkz. test: Fattn range [-52, 41]) sonraki UNet enjeksiyonunu
        # dengesizleştirmesini önler. num_groups=8, d_model=64'ü tam böler.
        self.norm = nn.GroupNorm(num_groups=8, num_channels=d_model)

        self.scale = self.head_dim ** -0.5

    def _partition_windows(self, x):
        """
        x: [B, C, H, W] → windows: [B * num_windows, ws*ws, C]
        """
        B, C, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "H, W window_size'a tam bölünmeli"

        # [B, C, H, W] → [B, C, H/ws, ws, W/ws, ws]
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        # → [B, H/ws, W/ws, ws, ws, C]
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        # → [B * num_windows, ws*ws, C]
        x = x.view(B * (H // ws) * (W // ws), ws * ws, C)
        return x

    def _merge_windows(self, x, B, H, W, C):
        """
        x: [B * num_windows, ws*ws, C] → [B, C, H, W]
        """
        ws = self.window_size
        nH, nW = H // ws, W // ws
        x = x.view(B, nH, nW, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        return x

    def forward(self, Ffused, F_terrain_local):
        """
        Ffused         : [B, 4, 64, 64]    — Stage 5 çıktısı
        F_terrain_local: [B, 512, 64, 64]  — F3 (Stage 1 TerrainEncoder)

        Returns:
            Fattn: [B, 4, 64, 64] — terrain-aware shadow features
        """
        B, _, H, W = Ffused.shape
        ws = self.window_size
        nW = (H // ws) * (W // ws)   # window sayısı
        tok = ws * ws                 # window başına token sayısı

        # Q, K, V projeksiyonu (hâlâ [B, d_model, H, W])
        Q = self.q_proj(Ffused)
        K = self.k_proj(F_terrain_local)
        V = self.v_proj(F_terrain_local)

        # Window partition (denklem 26): [B*nW, tok, d_model]
        Qw = self._partition_windows(Q)
        Kw = self._partition_windows(K)
        Vw = self._partition_windows(V)

        # Multi-head reshape: [B*nW, tok, d_model] → [B*nW, heads, tok, head_dim]
        def to_heads(t):
            Bn, T, _ = t.shape
            t = t.view(Bn, T, self.num_heads, self.head_dim)
            return t.permute(0, 2, 1, 3)  # [Bn, heads, T, head_dim]

        Qh = to_heads(Qw)
        Kh = to_heads(Kw)
        Vh = to_heads(Vw)

        # Scaled dot-product cross-attention (denklem 27)
        attn = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.scale  # [Bn, heads, tok, tok]
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, Vh)   # [Bn, heads, tok, head_dim]

        # Head'leri birleştir: [Bn, tok, d_model]
        Bn = out.shape[0]
        out = out.permute(0, 2, 1, 3).contiguous().view(Bn, tok, self.d_model)

        # Window merge (denklem 28): [B, d_model, H, W]
        out = self._merge_windows(out, B, H, W, self.d_model)

        # d_model kanalını normalize et (stabilite), sonra fused_channels'a projekte et
        out = self.norm(out)
        Fattn = self.out_proj(out)   # [B, 4, H, W]
        return Fattn


# ─── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("/Volumes/KIOXIA/LCIB_DiffusionSat/LCIB_project/stages")
    from dataset import LCIBDataset
    from stage1_encoder import TerrainEncoder, VAEEncoder
    from stage3_displacement import DisplacementMLP, SpatialWarp
    from stage5_automask import AutoMaskModule
    from torch.utils.data import DataLoader

    dataset = LCIBDataset()
    loader  = DataLoader(dataset, batch_size=2, shuffle=False)
    batch   = next(iter(loader))

    Iref = batch["Iref"]
    Mc   = batch["Mc"]
    meta = batch["meta"]

    print("Encoderlar yükleniyor...")
    terrain_enc = TerrainEncoder()
    vae_enc     = VAEEncoder()

    _, F_terrain = terrain_enc(Iref)
    zc = vae_enc.encode(Mc)

    disp_mlp = DisplacementMLP(hidden_dim=128)
    d = disp_mlp(meta)
    warp = SpatialWarp()
    Fwarp = warp(zc, d)

    amm = AutoMaskModule()
    Minfo, Ffused = amm(Fwarp, F_terrain)
    print(f"Ffused shape: {Ffused.shape}")

    # Stage 6: Window-Based Local Texture Attention
    local_attn = WindowLocalAttention(
        fused_channels=4, terrain_channels=512,
        d_model=64, num_heads=4, window_size=8
    )
    Fattn = local_attn(Ffused, F_terrain["F3"])

    print(f"\nFattn shape: {Fattn.shape}")   # [2, 4, 64, 64]
    print(f"Fattn range: [{Fattn.min():.3f}, {Fattn.max():.3f}]")

    print("\nStage 6 tamam!")
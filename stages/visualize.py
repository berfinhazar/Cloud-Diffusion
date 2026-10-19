"""
visualize.py — eğitilmiş modeli train/val örnekleri üzerinde görsel olarak test eder.

Her örnek için GERÇEK örnekleme (saf gürültüden başlayıp DDIM ile adım adım
denoise) yapar; eğitim loss'larındaki "gürültülü hedefi temizleme" hesabından
farklı olarak modelin Iref + Mc + metadata'dan görüntü ÜRETEBİLDİĞİNİ test eder.

Kolonlar (soldan sağa):
  Iref | Mc | Ms | Minfo (Ms açık) | Minfo (Ms=0) |
  Üretim (Ms açık) | Üretim (Ms=0) | Hedef (Isyn) | 1-adım (t=500, eğitim tarzı)

Ms=0 kolonları, modelin gölgeyi Ms maskesinden "kopyalayıp kopyalamadığını"
görmek içindir (--skip-ms-test ile kapatılabilir).

Kullanım (stages/ klasöründen):
    python visualize.py
    python visualize.py --n-train 5 --n-val 5 --steps 25
    python visualize.py --checkpoint /yol/epoch_1.pt --skip-ms-test

Bu script mevcut hiçbir dosyayı DEĞİŞTİRMEZ; sadece okur ve PNG/CSV yazar.
"""

import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import LCIBDataset                                   # noqa: E402
from stage10_train import LCIBPipeline                            # noqa: E402
from stage9_losses import predict_z0, v_to_eps, make_scheduler                              # noqa: E402
from train import CHECKPOINT_PATH, TRAINABLE_SUBMODULES, get_device  # noqa: E402

DEFAULT_CKPT = "/Volumes/KIOXIA/LCIB_DiffusionSat/LCIB_project/checkpoints/best.pt"
DEFAULT_OUT = "/Volumes/KIOXIA/LCIB_DiffusionSat/LCIB_project/visualizations"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    ap.add_argument("--n-train", type=int, default=5)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--steps", type=int, default=25, help="DDIM adım sayısı")
    ap.add_argument("--seed", type=int, default=42, help="train.py'deki split seed'i ile AYNI olmalı")
    ap.add_argument("--val-split", type=float, default=0.1, help="train.py ile AYNI olmalı")
    ap.add_argument("--pick-seed", type=int, default=0, help="örnek seçimi + gürültü seed'i")
    ap.add_argument("--skip-ms-test", action="store_true", help="Ms=0 karşılaştırmasını atla")
    ap.add_argument(
        "--sdedit-test", action="store_true",
        help="Örnekleyici doğrulama testi: gerçek hedefi t_start'a kadar gürültüleyip "
             "aynı DDIM ile geri temizler (normal grid üretilmez).",
    )
    ap.add_argument("--sdedit-ts", type=str, default="250,500,750,999",
                    help="virgülle ayrılmış t_start değerleri")
    ap.add_argument(
        "--iref-test", action="store_true",
        help="Asıl kullanım testi: TEMİZ görüntüyü (Iref) t_start'a kadar gürültüleyip, "
             "Mc/Ms/metadata conditioning ile geri temizler; bulut+gölge ekleniyor mu bakar.",
    )
    ap.add_argument("--iref-ts", type=str, default="250,500",
                    help="--iref-test için virgülle ayrılmış t_start değerleri")
    return ap.parse_args()


def denorm(x):
    """[-1,1] -> [0,1] HxWx3 numpy"""
    x = (x.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1)
    return x[0].permute(1, 2, 0).numpy()


def to_gray(m):
    return m.detach().float().cpu()[0, 0].numpy()


@torch.no_grad()
def build_conditioning(model, batch, Ms_override=None):
    """
    LCIBPipeline.forward'taki conditioning zincirinin (noise ve UNet çağrısı
    hariç) birebir aynısı. Returns: encoder_hidden_states, mid_residual, Minfo.
    """
    Iref, Mc, Ms, meta = batch["Iref"], batch["Mc"], batch["Ms"], batch["meta"]
    if Ms_override is not None:
        Ms = Ms_override

    _, F_terrain = model.terrain_enc(Iref)
    zc = model.vae_enc.encode(Mc)

    em = model.metadata_embedder(meta)
    ehs = model.embed_projector(em)

    d = model.disp_mlp(meta)
    Fwarp = model.warp(zc, d)

    Minfo, Ffused = model.amm(Fwarp, F_terrain, Ms)
    Fattn = model.local_attn(Ffused, F_terrain["F3"])
    Fout, _, _ = model.film(meta, Fattn)
    mid = model.cond_projector(Fout)
    return ehs, mid, Minfo


@torch.no_grad()
def ddim_sample(model, ehs, mid, alphas_cumprod, z_init, steps, device, t_start=None):
    """
    Deterministik DDIM (eta=0). UNet çıktısı v-prediction; ε'a çevrilip DDIM adımı uygulanır.
    t_start=None: saf gürültüden (t=999) başlar. Değilse z_init'in t_start
    seviyesinde gürültülü bir latent olduğu varsayılır.
    """
    ac = alphas_cumprod.cpu().numpy().astype(np.float64)
    if t_start is None:
        t_start = len(ac) - 1
    ts = np.linspace(0, t_start, steps).round().astype(int)[::-1]
    z = z_init.clone()
    B = z.shape[0]
    dummy_md = torch.zeros(B, getattr(model.unet, "num_metadata", 7), device=device)

    for i, t in enumerate(ts):
        t_tensor = torch.full((B,), int(t), device=device, dtype=torch.long)
        v = model.unet(
            sample=z, timestep=t_tensor, encoder_hidden_states=ehs,
            metadata=dummy_md, mid_block_additional_residual=mid,
        ).sample
        a_t = float(ac[t])
        eps = (a_t ** 0.5) * v + ((1 - a_t) ** 0.5) * z   # v → ε
        a_prev = float(ac[ts[i + 1]]) if i + 1 < len(ts) else 1.0
        z0 = (z - (1 - a_t) ** 0.5 * eps) / (a_t ** 0.5)
        z = (a_prev ** 0.5) * z0 + ((1 - a_prev) ** 0.5) * eps
    return z


@torch.no_grad()
def one_step_train_style(model, batch, ehs, mid, scheduler, device, t_val=500):
    """Eğitimdeki Lrec hesabının görsel karşılığı: gürültülü HEDEFTEN tek adımda tahmin."""
    z0 = model.vae_enc.encode(batch["Isyn"])
    eps = torch.randn_like(z0)
    t = torch.full((z0.shape[0],), t_val, device=device, dtype=torch.long)
    ac = scheduler.alphas_cumprod
    zt = ac[t].sqrt().view(-1, 1, 1, 1) * z0 + (1 - ac[t]).sqrt().view(-1, 1, 1, 1) * eps
    dummy_md = torch.zeros(z0.shape[0], getattr(model.unet, "num_metadata", 7), device=device)
    v_hat = model.unet(
        sample=zt, timestep=t, encoder_hidden_states=ehs,
        metadata=dummy_md, mid_block_additional_residual=mid,
    ).sample
    eps_hat = v_to_eps(v_hat, zt, t, ac)
    z0_hat = predict_z0(zt, eps_hat, t, ac)
    return model.vae_enc.decode(z0_hat)


@torch.no_grad()
def sdedit_denoise(model, batch, ehs, mid, alphas_cumprod, t_start, steps, device, gen):
    """Gerçek hedefi t_start'a kadar gürültüle, sonra aynı DDIM ile geri temizle."""
    z0 = model.vae_enc.encode(batch["Isyn"])
    eps = torch.randn(z0.shape, generator=gen).to(device)
    a = float(alphas_cumprod[t_start])
    zt = (a ** 0.5) * z0 + ((1 - a) ** 0.5) * eps
    z = ddim_sample(model, ehs, mid, alphas_cumprod, zt, steps, device, t_start=t_start)
    return model.vae_enc.decode(z)


def run_sdedit_test(args, model, scheduler, chosen, device, ckpt, plt):
    """
    Örnekleyici doğrulama testi. Beklenti: küçük t_start'ta çıktı hedefe çok
    yakın olmalı; t_start büyüdükçe giderek bozulur. t=250/500 bile bozuksa
    örnekleyici kodunda hata vardır; sadece t=999 bozuksa sorun saf gürültüden
    üretmededir (model/tasarım).
    """
    t_list = [int(x) for x in args.sdedit_ts.split(",") if x.strip()]
    items = []
    for split_name, (subset, idxs) in chosen.items():
        for idx in idxs:
            items.append((split_name, subset[idx]))

    columns = ["Iref (girdi)", "Mc (bulut)", "Hedef (Isyn)"] + [f"t={t} → geri" for t in t_list]
    fig, axes = plt.subplots(len(items), len(columns),
                             figsize=(1.9 * len(columns), 2.1 * len(items) + 0.5))
    axes = np.atleast_2d(axes)
    rows_csv = []

    for r, (split_name, sample) in enumerate(items):
        t0 = time.time()
        batch = {
            k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
            for k, v in sample.items()
        }
        torch.manual_seed(args.pick_seed * 1000 + r)
        ehs, mid, _ = build_conditioning(model, batch)
        gt = batch["Isyn"]

        panels = [(denorm(batch["Iref"]), None), (to_gray(batch["Mc"]), "gray"), (denorm(gt), None)]
        row = {"split": split_name, "name": sample["name"]}
        for t in t_list:
            steps_t = max(5, int(round(args.steps * t / 999.0)))
            gen = torch.Generator().manual_seed(args.pick_seed * 1000 + r)
            rec = sdedit_denoise(model, batch, ehs, mid, scheduler.alphas_cumprod,
                                 t, steps_t, device, gen)
            l1 = (rec - gt).abs().mean().item()
            row[f"L1_t{t}"] = round(l1, 4)
            panels.append((denorm(rec), None))
        rows_csv.append(row)

        for c, (img, cmap) in enumerate(panels):
            ax = axes[r, c]
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(columns[c], fontsize=8)
        label = " ".join(f"t{t}:{row[f'L1_t{t}']:.2f}" for t in t_list)
        axes[r, 0].set_ylabel(f"{split_name} {sample['name'][:14]}\n{label}", fontsize=6)
        print(f"  [{split_name} {r+1}/{len(items)}] {sample['name']} | {label} | {time.time()-t0:.0f}s")

    fig.suptitle(f"SDEdit-tarzı örnekleyici testi — ckpt step={ckpt.get('global_step')}", fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "sdedit_grid.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    out_csv = os.path.join(args.out_dir, "sdedit_metrics.csv")
    keys = sorted({k for r in rows_csv for k in r}, key=lambda k: (k not in ("split", "name"), k))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_csv)
    print(f"[+] Kaydedildi: {out_png}")
    print(f"[+] Kaydedildi: {out_csv}")


def _region_mean(img, mask):
    """img [1,3,H,W] (-1..1), mask [1,1,H,W] (bool/0-1) -> bölgedeki ortalama (nan: bölge boş)."""
    m = mask.float().expand_as(img)
    if m.sum().item() < 1:
        return float("nan")
    return ((img * m).sum() / m.sum()).item()


def _region_l1(a, b, mask):
    m = mask.float().expand_as(a)
    if m.sum().item() < 1:
        return float("nan")
    return (((a - b).abs() * m).sum() / m.sum()).item()


def run_iref_test(args, model, scheduler, chosen, device, ckpt, plt):
    """
    Asıl kullanım testi (cloud/shadow synthesis): temiz Iref -> gürültüle ->
    Mc/Ms/metadata conditioning ile geri temizle. Model bulut+gölge ekliyorsa:
      - bulut bölgesinde parlaklık Iref'e göre artmalı (hedefteki artışa yakın),
      - gölge bölgesinde (bulut dışı kısım) koyulaşma olmalı.
    Not: bu görevle model eğitilmedi, iyi çıkması garanti değil.
    """
    t_list = [int(x) for x in args.iref_ts.split(",") if x.strip()]
    items = []
    for split_name, (subset, idxs) in chosen.items():
        for idx in idxs:
            items.append((split_name, subset[idx]))

    columns = ["Iref (girdi)", "Mc (bulut)", "Hedef (Isyn)"]
    columns += [f"Iref→t={t}" for t in t_list]
    columns += [f"fark t={t} (sonuç−Iref)" for t in t_list]
    fig, axes = plt.subplots(len(items), len(columns),
                             figsize=(1.9 * len(columns), 2.1 * len(items) + 0.5))
    axes = np.atleast_2d(axes)
    rows_csv = []

    for r, (split_name, sample) in enumerate(items):
        t0 = time.time()
        batch = {
            k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
            for k, v in sample.items()
        }
        torch.manual_seed(args.pick_seed * 1000 + r)
        ehs, mid, _ = build_conditioning(model, batch)
        iref, gt, Mc, Ms = batch["Iref"], batch["Isyn"], batch["Mc"], batch["Ms"]
        cloud = Mc > 0.5
        shadow_only = (Ms > 0.5) & (~cloud)

        with torch.no_grad():
            z_ref = model.vae_enc.encode(iref)

        row = {"split": split_name, "name": sample["name"]}
        # Hedefin bulut/gölge bölgelerindeki değişimi (Iref'e göre) = "olması gereken"
        row["hedef_bulut_parlama"] = round(_region_mean(gt, cloud) - _region_mean(iref, cloud), 4)
        row["hedef_golge_koyulasma"] = round(
            _region_mean(gt, shadow_only) - _region_mean(iref, shadow_only), 4)
        row["L1_bulut_bolgesi_Iref_vs_hedef"] = round(_region_l1(iref, gt, cloud), 4)

        recs = []
        for t in t_list:
            steps_t = max(5, int(round(args.steps * t / 999.0)))
            gen = torch.Generator().manual_seed(args.pick_seed * 1000 + r)
            eps = torch.randn(z_ref.shape, generator=gen).to(device)
            a = float(scheduler.alphas_cumprod[t])
            zt = (a ** 0.5) * z_ref + ((1 - a) ** 0.5) * eps
            z = ddim_sample(model, ehs, mid, scheduler.alphas_cumprod, zt, steps_t, device, t_start=t)
            with torch.no_grad():
                rec = model.vae_enc.decode(z)
            recs.append(rec)
            row[f"t{t}_bulut_parlama"] = round(_region_mean(rec, cloud) - _region_mean(iref, cloud), 4)
            row[f"t{t}_golge_koyulasma"] = round(
                _region_mean(rec, shadow_only) - _region_mean(iref, shadow_only), 4)
            row[f"t{t}_L1_bulut_bolgesi_sonuc_vs_hedef"] = round(_region_l1(rec, gt, cloud), 4)
        rows_csv.append(row)

        panels = [(denorm(iref), None), (to_gray(Mc), "gray"), (denorm(gt), None)]
        panels += [(denorm(rec), None) for rec in recs]
        for rec in recs:
            diff = (rec - iref).mean(dim=1, keepdim=True)
            panels.append((to_gray(diff), "coolwarm"))

        for c, (img, cmap) in enumerate(panels):
            ax = axes[r, c]
            if cmap == "coolwarm":
                ax.imshow(img, cmap=cmap, vmin=-0.6, vmax=0.6)
            elif cmap:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(columns[c], fontsize=8)
        label = " ".join(f"t{t}:{row[f't{t}_bulut_parlama']:+.2f}" for t in t_list)
        axes[r, 0].set_ylabel(
            f"{split_name} {sample['name'][:14]}\nhedef:{row['hedef_bulut_parlama']:+.2f} {label}",
            fontsize=6)
        print(f"  [{split_name} {r+1}/{len(items)}] {sample['name']} | bulut parlama hedef="
              f"{row['hedef_bulut_parlama']:+.3f} model: {label} | {time.time()-t0:.0f}s")

    fig.suptitle(
        f"Iref testi (temiz görüntüden bulut+gölge ekleme) — ckpt step={ckpt.get('global_step')}\n"
        f"fark haritası: kırmızı=parlaklaştı, mavi=koyulaştı", fontsize=9)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "iref_test_grid.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    out_csv = os.path.join(args.out_dir, "iref_test_metrics.csv")
    keys = sorted({k for r in rows_csv for k in r}, key=lambda k: (k not in ("split", "name"), k))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_csv)
    print(f"[+] Kaydedildi: {out_png}")
    print(f"[+] Kaydedildi: {out_csv}")


def minfo_up(Minfo):
    return to_gray(F.interpolate(Minfo, size=(512, 512), mode="nearest"))


def in_out_ratio(Minfo, Mc):
    Mc_down = F.adaptive_avg_pool2d(Mc, output_size=Minfo.shape[-2:])
    cloud = (Mc_down > 0.1).float()
    non = 1.0 - cloud
    inc = (Minfo * cloud).sum() / cloud.sum().clamp_min(1)
    out = (Minfo * non).sum() / non.sum().clamp_min(1)
    return inc.item(), out.item()


def pick_indices(n_total, k, rng):
    k = min(k, n_total)
    return sorted(rng.sample(range(n_total), k))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"[+] Cihaz: {device}")
    print(f"[+] Checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {args.checkpoint}")

    # --- Dataset + train.py ile AYNI split ---
    full = LCIBDataset()
    n_val = max(1, int(len(full) * args.val_split))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"[+] Train: {n_train} | Val: {n_val}")

    rng = random.Random(args.pick_seed)
    chosen = {
        "train": (train_set, pick_indices(len(train_set), args.n_train, rng)),
        "val": (val_set, pick_indices(len(val_set), args.n_val, rng)),
    }

    # --- Model + ağırlıklar ---
    print("[+] Model yükleniyor (biraz sürebilir)...")
    model = LCIBPipeline(checkpoint_path=CHECKPOINT_PATH)
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    for name in TRAINABLE_SUBMODULES:
        getattr(model, name).load_state_dict(ckpt[name])
    print(f"[+] Ağırlıklar yüklendi (epoch={ckpt.get('epoch')}, global_step={ckpt.get('global_step')})")
    model = model.to(device).eval()

    scheduler = make_scheduler()
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.iref_test:
        run_iref_test(args, model, scheduler, chosen, device, ckpt, plt)
        print("[+] Bitti. iref_test_grid.png ve iref_test_metrics.csv dosyalarını paylaşabilirsin.")
        return

    if args.sdedit_test:
        run_sdedit_test(args, model, scheduler, chosen, device, ckpt, plt)
        print("[+] Bitti. sdedit_grid.png ve sdedit_metrics.csv dosyalarını paylaşabilirsin.")
        return

    do_ms = not args.skip_ms_test
    columns =["Iref (girdi)", "Mc (bulut)", "Ms (gölge)", "Minfo (Ms açık)"]
    if do_ms:
        columns.append("Minfo (Ms=0)")
    columns.append("Üretim (Ms açık)")
    if do_ms:
        columns.append("Üretim (Ms=0)")
    columns += ["Hedef (Isyn)", "1-adım t=500"]

    rows_csv = []
    for split_name, (subset, idxs) in chosen.items():
        n = len(idxs)
        fig, axes = plt.subplots(n, len(columns), figsize=(1.9 * len(columns), 2.1 * n + 0.5))
        axes = np.atleast_2d(axes)

        for r, idx in enumerate(idxs):
            t0 = time.time()
            sample = subset[idx]
            batch = {
                k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
                for k, v in sample.items()
            }
            name = sample["name"]

            torch.manual_seed(args.pick_seed * 1000 + r)
            ehs, mid, Minfo_on = build_conditioning(model, batch)
            gen = torch.Generator().manual_seed(args.pick_seed * 1000 + r)
            z_init = torch.randn(1, 4, 64, 64, generator=gen).to(device)

            z_on = ddim_sample(model, ehs, mid, scheduler.alphas_cumprod, z_init, args.steps, device)
            with torch.no_grad():
                gen_on = model.vae_enc.decode(z_on)

            if do_ms:
                ehs0, mid0, Minfo_off = build_conditioning(
                    model, batch, Ms_override=torch.zeros_like(batch["Ms"])
                )
                z_off = ddim_sample(model, ehs0, mid0, scheduler.alphas_cumprod, z_init, args.steps, device)
                with torch.no_grad():
                    gen_off = model.vae_enc.decode(z_off)

            one = one_step_train_style(model, batch, ehs, mid, scheduler, device)

            gt = batch["Isyn"]
            l1_on = (gen_on - gt).abs().mean().item()
            l1_one = (one - gt).abs().mean().item()
            inc, out = in_out_ratio(Minfo_on, batch["Mc"])
            row = {
                "split": split_name, "name": name,
                "L1_uretim_Ms_acik": round(l1_on, 4),
                "L1_1adim": round(l1_one, 4),
                "Minfo_in": round(inc, 4), "Minfo_out": round(out, 4),
                "Minfo_oran": round(inc / max(out, 1e-8), 3),
            }
            if do_ms:
                row["L1_uretim_Ms0"] = round((gen_off - gt).abs().mean().item(), 4)
            rows_csv.append(row)

            panels = [
                (denorm(batch["Iref"]), None),
                (to_gray(batch["Mc"]), "gray"),
                (to_gray(batch["Ms"]), "gray"),
                (minfo_up(Minfo_on), "viridis"),
            ]
            if do_ms:
                panels.append((minfo_up(Minfo_off), "viridis"))
            panels.append((denorm(gen_on), None))
            if do_ms:
                panels.append((denorm(gen_off), None))
            panels += [(denorm(gt), None), (denorm(one), None)]

            for c, (img, cmap) in enumerate(panels):
                ax = axes[r, c]
                if cmap == "viridis":
                    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                elif cmap:
                    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                else:
                    ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(columns[c], fontsize=8)
            axes[r, 0].set_ylabel(f"{name[:18]}\nL1={l1_on:.3f}", fontsize=7)
            print(f"  [{split_name} {r+1}/{n}] {name} | L1(üretim)={l1_on:.4f} "
                  f"| Minfo in/out={inc:.3f}/{out:.3f} | {time.time()-t0:.0f}s")

        fig.suptitle(
            f"{split_name.upper()} örnekleri — {args.steps}-adım DDIM, "
            f"ckpt step={ckpt.get('global_step')}", fontsize=10
        )
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, f"{split_name}_grid.png")
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"[+] Kaydedildi: {out_png}")

    out_csv = os.path.join(args.out_dir, "metrics.csv")
    keys = sorted({k for r in rows_csv for k in r}, key=lambda k: (k not in ("split", "name"), k))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_csv)
    print(f"[+] Kaydedildi: {out_csv}")
    print("[+] Bitti. train_grid.png, val_grid.png ve metrics.csv dosyalarını paylaşabilirsin.")


if __name__ == "__main__":
    main()

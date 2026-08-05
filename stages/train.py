"""
train.py — LCIB_DiffusionSat eğitim scripti.

Kullanım:
    python train.py --epochs 5 --batch-size 2
    python train.py --resume checkpoints/epoch_3.pt

Sadece trainable modülleri kaydeder/yükler (MetadataEmbedder, DisplacementMLP,
AutoMaskModule, WindowLocalAttention, FiLMModulation, ConditioningProjector,
EmbeddingProjector) — frozen VAE/TerrainEncoder/SatUNet checkpoint'ten her
seferinde taze yükleniyor, tekrar kaydetmeye gerek yok (checkpoint boyutunu
küçük tutar).

DÜZELTME (tanı amaçlı): compute_losses() Minfo'nun ortalama/maksimum
değerini de hesaplayıp loglara yazıyor. Bu, AutoMaskModule'ün ürettiği
Minfo maskesinin sıfıra kilitlenip kilitlenmediğini doğrudan gözlemlemek
için eklendi.

DÜZELTME (kod seviyesinde AMM çökme fix'i — Lmin warm-up):
Gözlemlenen kök neden: ConditioningProjector (Stage 8) ControlNet
konvansiyonuyla SIFIR init ediliyor (bkz. stage8_diffusion.py). Eğitimin
ilk adımlarında bu yüzden Ldiff, Minfo'yu "anlamlı tut" diye hiçbir gradyan
baskısı üretmiyor — ama Lmin (L1 sparsity, denklem 44) daha ilk adımdan
itibaren Minfo'yu sıfıra çekmeye çalışıyor. Karşı baskı olmadığı için
sigmoid çıkışı birkaç adımda 0'da doyuma ulaşıp kilitleniyor (vanishing
gradient) — bir kere kilitlenince de zero-conv hiç ısınamıyor, kısır döngü.

Çözüm: Lmin'in katsayısını (λ5) eğitimin ilk `--lmin-warmup-steps` adımında
0'dan hedef değerine DOĞRUSAL olarak rampalıyoruz (bkz. compute_losses()).
Böylece zero-conv ısınıp Ldiff gerçek bir sinyal üretmeye başlayana kadar
Minfo'ya sparsity baskısı uygulanmıyor; λ5 daha sonra kademeli devreye
giriyor. Bu, sadece --lambdas ile λ5'i sabit küçültmekten daha sağlam bir
çözüm çünkü kalıcı olarak zayıf bir sparsity yerine, mekanizmanın kendisini
(erken kilitlenmeyi) hedefliyor.

Varsayılan warm-up: 300 adım. Değiştirmek için:
    python train.py --epochs 1 --batch-size 2 --log-every 1 \\
        --lmin-warmup-steps 500
Warm-up'ı tamamen kapatmak (eski davranış) için --lmin-warmup-steps 0.
"""
import argparse
import csv
import os
import time
import sys

import torch
from torch.utils.data import DataLoader, random_split
from diffusers import DDPMScheduler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import LCIBDataset
from stage10_train import LCIBPipeline
from stage9_losses import (
    predict_z0, reconstruction_loss, diffusion_loss,
    preserve_loss, directional_loss, mask_sparsity_loss,
    mask_target_loss, elevation_consistency_loss, total_loss,
)

CHECKPOINT_PATH = "/Volumes/KINGSTON/LCIB_checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"
DEFAULT_SAVE_DIR = "/Volumes/KINGSTON/LCIB_DiffusionSat/LCIB_project/checkpoints"

TRAINABLE_SUBMODULES = [
    "metadata_embedder", "disp_mlp", "amm",
    "local_attn", "film", "cond_projector", "embed_projector",
]

# CSV/log'daki metrik sırası — hem train hem val satırlarının aynı sütun
# sayısında kalması için tek yerden yönetiliyor.
METRIC_KEYS = ["Ldiff", "Lrec", "Lpreserve", "Ldir", "Lmin", "Lpreserve_mask",
               "Lele", "Ltotal", "Minfo_mean", "Minfo_max", "Lmin_weight"]


def lmin_warmup_factor(global_step, warmup_steps):
    """
    λ5 (Lmin katsayısı) için 0→1 doğrusal warm-up çarpanı.
    global_step, warmup_steps: int. warmup_steps<=0 ise her zaman 1.0
    (warm-up kapalı, eski davranış).
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, global_step / warmup_steps)


def parse_args():
    ap = argparse.ArgumentParser(description="LCIB_DiffusionSat eğitim scripti")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--checkpoint-dir", type=str, default=DEFAULT_SAVE_DIR)
    ap.add_argument("--resume", type=str, default=None, help="Devam edilecek checkpoint dosyası")
    ap.add_argument("--log-every", type=int, default=5, help="Kaç step'te bir log basılsın")
    ap.add_argument("--save-every", type=int, default=1, help="Kaç epoch'ta bir checkpoint kaydedilsin")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambdas", type=float, nargs=7,
                     default=[1.0, 1.0, 1.0, 1.0, 1.0, 0.025, 1.0],
                     help="λ1..λ7: Ldiff Lrec Lpreserve Ldir Lmin Lele "
                          "Lpreserve_mask(Minfo — Lmin'e karşı-kuvvet)")
    ap.add_argument("--lmin-warmup-steps", type=int, default=300,
                     help="λ5 (Lmin/AMM sparsity katsayısı) 0'dan hedef "
                          "değerine kaç global step'te doğrusal rampalanacak "
                          "(AMM/Minfo erken çökme fix'i, bkz. dosya başı not). "
                          "0 = warm-up kapalı, λ5 baştan tam güçte.")
    ap.add_argument("--num-workers", type=int, default=0)
    return ap.parse_args()


def save_checkpoint(path, model, optimizer, lr_sched, epoch, global_step, best_val):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_sched.state_dict(),
    }
    for name in TRAINABLE_SUBMODULES:
        ckpt[name] = getattr(model, name).state_dict()
    torch.save(ckpt, path)
    print(f"  [checkpoint kaydedildi → {path}]")


def load_checkpoint(path, model, optimizer, lr_sched):
    ckpt = torch.load(path, map_location="cpu")
    for name in TRAINABLE_SUBMODULES:
        getattr(model, name).load_state_dict(ckpt[name])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_sched.load_state_dict(ckpt["lr_scheduler"])
    print(f"  [checkpoint yüklendi ← {path}, epoch {ckpt['epoch']}'ten devam]")
    return ckpt["epoch"], ckpt["global_step"], ckpt.get("best_val", float("inf"))


def compute_losses(out, model, lambdas, scheduler, global_step=0, warmup_steps=0):
    """
    NOT: decode() burada BİLEREK no_grad içinde DEĞİL — Lrec ve Lpreserve'in
    gradyanının eps_hat'e (ve oradan trainable modüllere) geri akabilmesi
    için z0_hat → decode zincirinin hesap grafiğinde kalması şart. Eğitim
    dışı (validation) çağrılarda bu fonksiyon zaten dışarıdan torch.no_grad()
    ile sarmalanıyor, orada ekstra bir şey yapmaya gerek yok.

    global_step, warmup_steps: λ5 (Lmin katsayısı) warm-up'ı için — bkz.
    dosya başındaki "Lmin warm-up" notu ve lmin_warmup_factor().
    """
    z0_hat = predict_z0(out["zt"], out["eps_hat"], out["t"], scheduler.alphas_cumprod)
    Isyn_pred = model.vae_enc.decode(z0_hat)

    Ldiff          = diffusion_loss(out["eps"], out["eps_hat"])
    Lrec           = reconstruction_loss(Isyn_pred, out["Igt"])
    Lpreserve      = preserve_loss(Isyn_pred, out["Iref"], out["Mc"])
    Ldir           = directional_loss(out["d"], out["meta"])
    Lmin           = mask_sparsity_loss(out["Minfo"])
    Lpreserve_mask = mask_target_loss(out["Minfo"], out["Mc"], out["d"])
    Lele           = elevation_consistency_loss(out["d"], out["meta"])

    # DÜZELTME (AMM çökme fix'i): λ5'i (Lmin katsayısı) warm-up çarpanıyla
    # ölçekliyoruz. warmup_steps boyunca 0→λ5 rampalanır; bu sürede AMM
    # sadece Ldiff/Lrec/Lpreserve gibi diğer loss'ların dolaylı baskısıyla
    # şekillenir, erken ve karşılıksız bir L1 cezasıyla sıfıra kilitlenmez.


    l1, l2, l3, l4, l5, l6, l7 = lambdas
    warmup_factor = lmin_warmup_factor(global_step, warmup_steps)
    # NOT: l7 (Lpreserve_mask) warm-up'a TABİ DEĞİL. Lmin zaten warm-up
    # sırasında zayıf/kapalı olduğundan l7'nin baştan tam güçte olması bir
    # dengesizlik yaratmaz — tersine, AMM'in en baştan "hangi bölgeler
    # önemli" sinyalini almasını sağlar. Warm-up bitip Lmin tam güce
    # ulaştığında da artık karşısında sabit bir direnç var.
    lambdas_eff = (l1, l2, l3, l4, l5 * warmup_factor, l6, l7)

    Ltotal = total_loss(Ldiff, Lrec, Lpreserve, Ldir, Lmin, Lele,
                         Lpreserve_mask, lambdas=lambdas_eff)

    # TANI: Minfo'nun (AMM soft mask) ortalama/maksimum değeri + o anki
    # efektif Lmin katsayısı. Minfo_mean birkaç düzine adım boyunca ~0.00
    # civarında kilitli kalırsa (warm-up tamamlandıktan SONRA da), AMM
    # maskesi hâlâ öğrenmeyi bırakmış demektir — bu durumda
    # --lmin-warmup-steps'i artırıp tekrar deneyin.
    with torch.no_grad():
        Minfo_mean = out["Minfo"].mean().item()
        Minfo_max  = out["Minfo"].max().item()

    parts = {
        "Ldiff": Ldiff.item(), "Lrec": Lrec.item(), "Lpreserve": Lpreserve.item(),
        "Ldir": Ldir.item(), "Lmin": Lmin.item(),
        "Lpreserve_mask": Lpreserve_mask.item(), "Lele": Lele.item(),
        "Ltotal": Ltotal.item(),
        "Minfo_mean": Minfo_mean, "Minfo_max": Minfo_max,
        "Lmin_weight": l5 * warmup_factor,
    }
    return Ltotal, parts


def run_epoch(model, loader, scheduler, optimizer, lr_sched, lambdas,
              train, log_every, global_step, csv_writer, epoch_idx,
              lmin_warmup_steps=0):
    total_loss_sum = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        if train:
            optimizer.zero_grad()
            out = model(batch, scheduler=scheduler)
            # NOT: warm-up hesabı için global_step, bu adımın BAŞINDAKİ
            # değeriyle veriliyor (ilk adım = step 0 → Lmin ağırlığı 0'dan
            # başlar). Increment aşağıda backward/step'ten sonra oluyor.
            Ltotal, parts = compute_losses(
                out, model, lambdas, scheduler,
                global_step=global_step, warmup_steps=lmin_warmup_steps,
            )
            Ltotal.backward()
            optimizer.step()
            lr_sched.step()
            global_step += 1
        else:
            with torch.no_grad():
                out = model(batch, scheduler=scheduler)
                Ltotal, parts = compute_losses(
                    out, model, lambdas, scheduler,
                    global_step=global_step, warmup_steps=lmin_warmup_steps,
                )

        total_loss_sum += parts["Ltotal"]
        n_batches += 1

        if train and csv_writer is not None:
            csv_writer.writerow([epoch_idx, global_step, "train"]
                                 + [parts[k] for k in METRIC_KEYS])

        if train and (i % log_every == 0):
            lr_now = lr_sched.get_last_lr()[0]
            print(f"  epoch {epoch_idx} step {i+1}/{len(loader)} "
                  f"(global {global_step}) lr={lr_now:.2e} | "
                  + " ".join(f"{k}={parts[k]:.4f}" for k in METRIC_KEYS))

    avg_loss = total_loss_sum / max(n_batches, 1)
    return avg_loss, global_step


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Dataset yükleniyor...")
    full_dataset = LCIBDataset()
    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {n_train} sample | Val: {n_val} sample")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

    print("Model yükleniyor (SatUNet dahil, biraz sürebilir)...")
    model = LCIBPipeline(checkpoint_path=CHECKPOINT_PATH)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    if args.resume is not None and os.path.exists(args.resume):
        start_epoch, global_step, best_val = load_checkpoint(args.resume, model, optimizer, lr_sched)
        start_epoch += 1   # kaldığı epoch'tan SONRA devam et

    log_path = os.path.join(args.checkpoint_dir, "train_log.csv")
    log_is_new = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    csv_writer = csv.writer(log_file)
    if log_is_new:
        csv_writer.writerow(["epoch", "global_step", "split"] + METRIC_KEYS)

    print(f"\nEğitim başlıyor: epoch {start_epoch} → {args.epochs - 1}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"── Epoch {epoch} ──")

        train_loss, global_step = run_epoch(
            model, train_loader, scheduler, optimizer, lr_sched, args.lambdas,
            train=True, log_every=args.log_every, global_step=global_step,
            csv_writer=csv_writer, epoch_idx=epoch,
            lmin_warmup_steps=args.lmin_warmup_steps,
        )

        val_loss, _ = run_epoch(
            model, val_loader, scheduler, optimizer, lr_sched, args.lambdas,
            train=False, log_every=args.log_every, global_step=global_step,
            csv_writer=None, epoch_idx=epoch,
            lmin_warmup_steps=args.lmin_warmup_steps,
        )
        # val satırı da METRIC_KEYS ile aynı sütun sayısında olmalı — sadece
        # Ltotal biliniyor, geri kalanı boş bırakılıyor (Ltotal, METRIC_KEYS
        # içinde 7. sırada).
        val_row = ["" for _ in METRIC_KEYS]
        val_row[METRIC_KEYS.index("Ltotal")] = val_loss
        csv_writer.writerow([epoch, global_step, "val"] + val_row)
        log_file.flush()

        dt = time.time() - t0
        print(f"  epoch {epoch} bitti ({dt:.1f}s) | train Ltotal={train_loss:.4f} "
              f"| val Ltotal={val_loss:.4f}")

        if (epoch % args.save_every == 0) or (epoch == args.epochs - 1):
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, lr_sched, epoch, global_step, best_val)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.checkpoint_dir, "best.pt")
            save_checkpoint(best_path, model, optimizer, lr_sched, epoch, global_step, best_val)

    log_file.close()
    print("\nEğitim tamamlandı!")


if __name__ == "__main__":
    main()

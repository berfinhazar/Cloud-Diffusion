"""
train.py — LCIB_DiffusionSat eğitim scripti.

Kullanım:
    Yeni eğitim:
        python train.py --epochs 5 --batch-size 2

    Checkpoint'ten devam:
        python train.py --resume checkpoints/latest.pt --epochs 5

Checkpoint sistemi:
    - latest.pt
        En son güvenli devam noktası.
        Her N global step'te bir üzerine yazılır.

    - step_XXXXXX.pt
        Her N global step'te oluşturulan kalıcı yedek.

    - epoch_X.pt
        Epoch sonunda oluşturulan checkpoint.

    - best.pt
        En iyi validation loss'a sahip checkpoint.

Resume sistemi:
    Checkpoint'te aşağıdakiler saklanır:

        epoch
        batch_in_epoch
        global_step
        epoch_loss_sum
        epoch_loss_count
        optimizer
        scheduler
        trainable model ağırlıkları
        Python RNG
        NumPy RNG
        PyTorch RNG
        MPS RNG (varsa)

    Böylece eğitim epoch ortasında kesilirse aynı epoch'un
    aynı batch sırasından devam edebilir.

ÖNEMLİ:
    Frozen VAE / TerrainEncoder / SatUNet checkpoint'ten
    tekrar yüklenir ve checkpoint içine kaydedilmez.

Lmin warm-up:
    Varsayılan 300 global step.

    --lmin-warmup-steps 0
        warm-up kapalı.

    Örneğin:
        python train.py --epochs 5 --batch-size 2 \
            --lmin-warmup-steps 500
"""

import argparse
import csv
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split, Sampler
from diffusers import DDPMScheduler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import LCIBDataset
from stage10_train import LCIBPipeline
from stage9_losses import (
    predict_z0,
    reconstruction_loss,
    diffusion_loss,
    preserve_loss,
    directional_loss,
    mask_sparsity_loss,
    mask_target_loss,
    elevation_consistency_loss,
    total_loss,
)


CHECKPOINT_PATH = (
    "/Volumes/KIOXIA/LCIB_checkpoints/"
    "finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"
)

DEFAULT_SAVE_DIR = (
    "/Volumes/KIOXIA/LCIB_DiffusionSat/"
    "LCIB_project/checkpoints"
)

TRAINABLE_SUBMODULES = [
    "metadata_embedder",
    "disp_mlp",
    "amm",
    "local_attn",
    "film",
    "cond_projector",
    "embed_projector",
]

METRIC_KEYS = [
    "Ldiff", "Lrec", "Lpreserve", "Ldir", "Lmin", "Lpreserve_mask",
    "Lele", "Ltotal", "Minfo_mean", "Minfo_max", "Minfo_in_cloud",
    "Minfo_out_cloud", "Lmin_weight",
]


# ============================================================
# LMIN WARM-UP
# ============================================================

def lmin_warmup_factor(global_step, warmup_steps):
    """
    λ5 için 0 → 1 doğrusal warm-up.

    Örnek:
        warmup_steps = 300

        step 0    -> 0.00
        step 75   -> 0.25
        step 150  -> 0.50
        step 225  -> 0.75
        step 300+ -> 1.00

    warmup_steps <= 0:
        warm-up kapalı.
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, global_step / warmup_steps)


# ============================================================
# DETERMINISTIC EPOCH SAMPLER
# ============================================================

class EpochRandomSampler(Sampler):
    """
    Her epoch için deterministic permutation üretir.

    Bunun amacı resume sırasında aynı epoch'un
    aynı batch sırasını tekrar elde etmektir.

    Örneğin:

        Epoch 2
        Batch 1243

    checkpoint'ten sonra tekrar başlatıldığında
    Epoch 2'nin aynı permutation'ı oluşturulur ve
    ilk 1243 batch atlanarak 1244. batch'ten devam edilir.
    """

    def __init__(self, data_source, seed=42, epoch=0):
        self.data_source = data_source
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


# ============================================================
# DEVICE (MPS/CPU)
# ============================================================

def get_device():
    """
    DÜZELTME (MPS/GPU fix'i): eskiden hiçbir yerde model/tensor
    açıkça bir cihaza taşınmıyordu — her şey varsayılan olarak
    CPU'da çalışıyordu. SatUNet gibi büyük bir model için bu,
    Apple Silicon Mac'lerde MPS'e göre ~10-50x daha yavaş demek.

    Mevcutsa MPS (Apple Silicon GPU) kullanılır, yoksa CPU'ya
    düşülür.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch, device):
    """
    batch (dataset'ten gelen dict) içindeki tüm tensor değerlerini
    (Iref, Isyn, Mc, Ms, meta) verilen cihaza taşır. "name" gibi
    tensor olmayan alanlara dokunmaz.
    """
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


# ============================================================
# RNG
# ============================================================

def get_rng_state():
    """
    Python / NumPy / PyTorch / CUDA / MPS RNG state'lerini alır.

    MPS, Apple Silicon Mac'lerde kullanılan backend'dir.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    # CUDA
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    # MPS
    if (
        hasattr(torch, "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "get_rng_state")
    ):
        try:
            state["mps"] = torch.mps.get_rng_state()
        except Exception:
            pass

    return state


def restore_rng_state(state):
    """
    Daha önce kaydedilmiş RNG state'lerini geri yükler.
    """
    if state is None:
        return

    if "python" in state:
        random.setstate(state["python"])

    if "numpy" in state:
        np.random.set_state(state["numpy"])

    if "torch" in state:
        torch.set_rng_state(state["torch"])

    # CUDA
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])

    # MPS
    if (
        "mps" in state
        and hasattr(torch, "mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        try:
            torch.mps.set_rng_state(state["mps"])
        except Exception:
            pass


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="LCIB_DiffusionSat eğitim scripti")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--checkpoint-dir", type=str, default=DEFAULT_SAVE_DIR)
    ap.add_argument(
        "--resume", type=str, default=None,
        help="Devam edilecek checkpoint dosyası",
    )
    ap.add_argument(
        "--log-every", type=int, default=5,
        help="Kaç step'te bir log basılsın",
    )
    ap.add_argument(
        "--save-every", type=int, default=1,
        help="Kaç epoch'ta bir epoch checkpoint kaydedilsin",
    )
    ap.add_argument(
        "--save-every-steps", type=int, default=200,
        help=(
            "Kaç global step'te bir checkpoint alınsın. "
            "DÜZELTME (disk doluşu fix'i): eskiden 50'ydi — çok sık "
            "checkpoint almak hem gereksiz IO yükü hem de (aşağıdaki "
            "--keep-last-steps olmasa) hızlı disk doluşu demekti. "
            "200 ile hem yeterince sık güvenlik kopyası alınıyor hem "
            "de dosya sayısı/yazma sıklığı azalıyor."
        ),
    )
    ap.add_argument(
        "--keep-last-steps", type=int, default=5,
        help=(
            "DÜZELTME (disk doluşu fix'i): step_*.pt dosyaları eskiden "
            "hiç silinmiyordu — uzun bir eğitimde (örn. 10 epoch, "
            "~90.000 step) bu yüzlerce GB'a kadar çıkabiliyordu "
            "(checkpoint başına ~150-200MB x binlerce dosya), dataset'in "
            "kendisinden (~66GB) bile büyük. Bu parametre sadece en yeni "
            "N step_*.pt dosyasını tutup eskilerini otomatik siliyor. "
            "epoch_X.pt ve best.pt HİÇ etkilenmiyor, hep saklanıyor — "
            "onlar zaten epoch sayısıyla sınırlı, disk doldurma riski "
            "taşımıyorlar. 0 = temizlik kapalı (eski davranış, hepsi "
            "saklanır)."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--lambdas", type=float, nargs=7,
        default=[1.0, 1.0, 1.0, 1.0, 0.25, 0.025, 1.0],
        help="λ1..λ7: Ldiff Lrec Lpreserve Ldir Lmin Lele Lpreserve_mask",
    )
    ap.add_argument(
        "--lmin-warmup-steps", type=int, default=300,
        help=(
            "λ5'in 0'dan hedef değerine kaç global step'te "
            "rampalanacağı. 0 = warm-up kapalı."
        ),
    )
    ap.add_argument("--num-workers", type=int, default=0)

    return ap.parse_args()


# ============================================================
# ATOMIC CHECKPOINT SAVE
# ============================================================

def atomic_torch_save(obj, path):
    """
    Checkpoint'i önce geçici dosyaya yazar,
    ardından os.replace() ile gerçek dosyanın yerine koyar.

    Böylece özellikle latest.pt yazılırken bilgisayar kapanırsa
    yarım/corrupt latest.pt bırakma riski azaltılır.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    temp_path = path + ".tmp." + str(os.getpid())

    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# STEP CHECKPOINT CLEANUP
# ============================================================

def cleanup_old_step_checkpoints(checkpoint_dir, keep_last_steps):
    """
    DÜZELTME (disk doluşu fix'i): step_XXXXXX.pt dosyaları her
    --save-every-steps adımda bir oluşturuluyor ve eskiden hiç
    silinmiyordu. Uzun bir eğitimde (örn. 10 epoch, ~90.000 step)
    bu, checkpoint boyutuna göre yüzlerce GB'a kadar çıkabiliyor
    (~150-200MB/dosya x binlerce dosya) — dataset'in kendisinden
    (~66GB) bile büyük, SSD'yi tamamen doldurabilir.

    step_*.pt dosyaları sadece "eğitim yarıda kesilirse kaldığı
    yerden devam edebilme" amaçlı GEÇİCİ güvenlik kopyaları — nihai
    sonuç/sunum için hep epoch_X.pt ve best.pt kullanılıyor. Bu
    fonksiyon SADECE step_*.pt dosyalarına dokunur; epoch_X.pt ve
    best.pt hiçbir zaman silinmez (zaten epoch sayısıyla sınırlı
    oldukları için disk doldurma riski taşımıyorlar).

    En yeni `keep_last_steps` tane step_*.pt dosyası tutulur, daha
    eskileri silinir. Böylece kesinti durumunda hâlâ yakın bir
    noktadan devam edilebilir, ama disk kullanımı sabit bir üst
    sınırda kalır.

    keep_last_steps <= 0 ise temizlik yapılmaz (eski davranış).
    """
    if keep_last_steps <= 0:
        return

    pattern = re.compile(r"^step_(\d+)\.pt$")
    step_files = []

    for name in os.listdir(checkpoint_dir):
        m = pattern.match(name)
        if m:
            step_files.append((int(m.group(1)), name))

    # Step numarasına göre eskiden yeniye sırala.
    step_files.sort(key=lambda x: x[0])

    if len(step_files) <= keep_last_steps:
        return

    # En yeni keep_last_steps tanesi hariç hepsi silinecek.
    to_delete = step_files[:-keep_last_steps]

    for _, name in to_delete:
        path = os.path.join(checkpoint_dir, name)
        try:
            os.remove(path)
        except OSError:
            pass


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    path, model, optimizer, lr_sched,
    epoch, batch_in_epoch, global_step, best_val,
    epoch_loss_sum, epoch_loss_count, args,
    rng_state=None,
):
    """
    Tam eğitim state'ini kaydeder.

    epoch:
        Şu anda çalışılan epoch.

        Örneğin:
            epoch = 2

        -> 3. epoch (0-indexed)

    batch_in_epoch:
        Tamamlanmış batch sayısı.

        Örneğin:
            batch_in_epoch = 2500

        -> bu epoch'ta ilk 2500 batch tamamlandı.
        Resume'da 2501. batch'ten devam edilir.

    epoch_loss_sum / epoch_loss_count:
        Epoch ortasında resume edilirse train loss hesabının
        da kaldığı yerden devam etmesi için saklanır.

    rng_state:
        Özel olarak verilirse o RNG state kullanılır.

        Bu özellikle Ctrl+C sırasında önemlidir:
        Eğer batch henüz tamamlanmadıysa, o batch'in başındaki
        RNG state kaydedilir.
    """
    if rng_state is None:
        rng_state = get_rng_state()

    ckpt = {
        "checkpoint_version": 2,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "best_val": best_val,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_loss_count": epoch_loss_count,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_sched.state_dict(),
        "rng_state": rng_state,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambdas": args.lambdas,
        "lmin_warmup_steps": args.lmin_warmup_steps,
        "trainable_submodules": TRAINABLE_SUBMODULES,
    }

    for name in TRAINABLE_SUBMODULES:
        ckpt[name] = getattr(model, name).state_dict()

    atomic_torch_save(ckpt, path)


# ============================================================
# CHECKPOINT LOAD
# ============================================================

def load_checkpoint(path, model, optimizer, lr_sched):
    """
    Checkpoint yükler.

    Yeni checkpoint:
        epoch
        batch_in_epoch
        global_step
        epoch_loss_sum
        epoch_loss_count

    içerir.

    Eski checkpoint'te batch_in_epoch yoksa,
    checkpoint epoch sonu checkpoint'i kabul edilir.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Eski PyTorch sürümleri weights_only parametresini
        # desteklemiyorsa normal yükleme.
        ckpt = torch.load(path, map_location="cpu")

    for name in TRAINABLE_SUBMODULES:
        if name not in ckpt:
            raise KeyError(f"Checkpoint içinde '{name}' bulunamadı.")

        getattr(model, name).load_state_dict(ckpt[name])

    optimizer.load_state_dict(ckpt["optimizer"])
    lr_sched.load_state_dict(ckpt["lr_scheduler"])

    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    best_val = ckpt.get("best_val", float("inf"))

    # Yeni checkpoint sistemi
    if "batch_in_epoch" in ckpt:
        batch_in_epoch = ckpt.get("batch_in_epoch", 0)
        epoch_loss_sum = ckpt.get("epoch_loss_sum", 0.0)
        epoch_loss_count = ckpt.get("epoch_loss_count", 0)
    else:
        # Eski checkpoint:
        # Sadece epoch kaydedilmişse bunu epoch SONU
        # checkpoint'i kabul ediyoruz.
        batch_in_epoch = 0
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch += 1

    restore_rng_state(ckpt.get("rng_state"))

    return epoch, batch_in_epoch, global_step, best_val, epoch_loss_sum, epoch_loss_count


# ============================================================
# LOSS
# ============================================================

def compute_losses(out, model, lambdas, scheduler, global_step=0, warmup_steps=0):
    """
    Loss hesaplama.

    decode() BİLEREK no_grad içinde değildir.

    Lrec ve Lpreserve'in gradient'inin eps_hat'e,
    oradan trainable modüllere geri akması gerekir.
    """
    z0_hat = predict_z0(out["zt"], out["eps_hat"], out["t"], scheduler.alphas_cumprod)
    Isyn_pred = model.vae_enc.decode(z0_hat)

    Ldiff = diffusion_loss(out["eps"], out["eps_hat"])
    Lrec = reconstruction_loss(Isyn_pred, out["Igt"])
    Lpreserve = preserve_loss(Isyn_pred, out["Iref"], out["Mc"])
    Ldir = directional_loss(out["d"], out["meta"])
    Lmin = mask_sparsity_loss(out["Minfo"])
    Lpreserve_mask = mask_target_loss(out["Minfo"], out["Mc"], out["d"])
    Lele = elevation_consistency_loss(out["d"], out["meta"])

    l1, l2, l3, l4, l5, l6, l7 = lambdas
    warmup_factor = lmin_warmup_factor(global_step, warmup_steps)
    lambdas_eff = (l1, l2, l3, l4, l5 * warmup_factor, l6, l7)

    Ltotal = total_loss(
        Ldiff, Lrec, Lpreserve, Ldir, Lmin, Lele, Lpreserve_mask,
        lambdas=lambdas_eff,
    )

    with torch.no_grad():
        Minfo_mean = out["Minfo"].mean().item()
        Minfo_max = out["Minfo"].max().item()

        Mc_down = F.adaptive_avg_pool2d(out["Mc"], output_size=out["Minfo"].shape[-2:])
        cloud_region = (Mc_down > 0.1).float()
        noncloud_region = 1.0 - cloud_region

        Minfo_in_cloud = (
            (out["Minfo"] * cloud_region).sum() / cloud_region.sum().clamp_min(1)
        )
        Minfo_out_cloud = (
            (out["Minfo"] * noncloud_region).sum() / noncloud_region.sum().clamp_min(1)
        )

    parts = {
        "Ldiff": Ldiff.item(),
        "Lrec": Lrec.item(),
        "Lpreserve": Lpreserve.item(),
        "Ldir": Ldir.item(),
        "Lmin": Lmin.item(),
        "Lpreserve_mask": Lpreserve_mask.item(),
        "Lele": Lele.item(),
        "Ltotal": Ltotal.item(),
        "Minfo_mean": Minfo_mean,
        "Minfo_max": Minfo_max,
        "Minfo_in_cloud": Minfo_in_cloud.item(),
        "Minfo_out_cloud": Minfo_out_cloud.item(),
        "Lmin_weight": l5 * warmup_factor,
    }

    return Ltotal, parts


# ============================================================
# TRAIN EPOCH
# ============================================================

def run_train_epoch(
    model, loader, sampler, scheduler, optimizer, lr_sched, lambdas,
    epoch_idx, start_batch, global_step, log_every, csv_writer, csv_file,
    lmin_warmup_steps, save_every_steps, checkpoint_dir, best_val, args,
    progress_state, epoch_loss_sum=0.0, epoch_loss_count=0,
    keep_last_steps=5, device=None,
):
    """
    Training epoch.

    start_batch:
        Daha önce tamamlanan batch sayısı.

        Örneğin:

            start_batch = 2500

        ise ilk 2500 batch atlanır ve
        2501. batch'ten devam edilir.

    progress_state:
        Ctrl+C geldiğinde main()'in tam olarak
        nerede olduğumuzu bilmesi için ortak state.
    """
    sampler.set_epoch(epoch_idx)
    total_batches = len(loader)

    if start_batch > 0:
        print(f"  [+] Epoch {epoch_idx}: {start_batch} batch tamamlanmış.")
        print(f"  [+] {start_batch + 1}. batch'ten devam ediliyor.")

    for i, batch in enumerate(loader):
        # Daha önce tamamlanan batch'leri atla.
        if i < start_batch:
            continue

        # DÜZELTME (MPS/GPU fix'i): batch içindeki tensörleri
        # model ile aynı cihaza taşı.
        if device is not None:
            batch = move_batch_to_device(batch, device)

        # ----------------------------------------------------
        # Bu batch'in BAŞLANGIÇ state'i
        # ----------------------------------------------------
        #
        # Eğer Ctrl+C batch tamamlanmadan gelirse,
        # bu RNG state checkpoint'e yazılacak.
        #
        # Böylece resume edildiğinde aynı batch,
        # aynı RNG başlangıcıyla tekrar hesaplanabilir.
        #
        batch_rng_state = get_rng_state()

        progress_state["epoch"] = epoch_idx
        progress_state["batch_in_epoch"] = i
        progress_state["global_step"] = global_step
        progress_state["epoch_loss_sum"] = epoch_loss_sum
        progress_state["epoch_loss_count"] = epoch_loss_count
        progress_state["rng_before_batch"] = batch_rng_state

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------
        optimizer.zero_grad()

        out = model(batch, scheduler=scheduler)

        Ltotal, parts = compute_losses(
            out, model, lambdas, scheduler,
            global_step=global_step, warmup_steps=lmin_warmup_steps,
        )

        Ltotal.backward()
        optimizer.step()

        # Optimizer step tamamlandıktan sonra scheduler.
        lr_sched.step()

        # ----------------------------------------------------
        # BATCH TAMAMLANDI
        # ----------------------------------------------------
        global_step += 1
        completed_batch = i + 1

        epoch_loss_sum += parts["Ltotal"]
        epoch_loss_count += 1

        # Artık batch tamamlandı.
        # Ctrl+C bundan sonra gelirse bu batch replay edilmemeli.
        progress_state["epoch"] = epoch_idx
        progress_state["batch_in_epoch"] = completed_batch
        progress_state["global_step"] = global_step
        progress_state["epoch_loss_sum"] = epoch_loss_sum
        progress_state["epoch_loss_count"] = epoch_loss_count
        progress_state["rng_before_batch"] = None

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------
        if csv_writer is not None:
            csv_writer.writerow(
                [epoch_idx, global_step, "train"] + [parts[k] for k in METRIC_KEYS]
            )

            # Ani kapanmaya karşı her satırı flush et.
            if csv_file is not None:
                csv_file.flush()

        # ----------------------------------------------------
        # TERMINAL LOG
        # ----------------------------------------------------
        if (
            global_step % log_every == 0
            or completed_batch == start_batch + 1
            or completed_batch == total_batches
        ):
            lr_now = lr_sched.get_last_lr()[0]
            metrics_text = " ".join(f"{k}={parts[k]:.4f}" for k in METRIC_KEYS)

            print(
                f"  epoch {epoch_idx} | step {global_step} | "
                f"batch {completed_batch}/{total_batches} | "
                f"lr={lr_now:.2e} | {metrics_text}"
            )

        # ----------------------------------------------------
        # STEP CHECKPOINT
        # ----------------------------------------------------
        if save_every_steps > 0 and global_step % save_every_steps == 0:
            latest_path = os.path.join(checkpoint_dir, "latest.pt")
            step_path = os.path.join(checkpoint_dir, f"step_{global_step:06d}.pt")

            # Batch tamamen tamamlandığı için
            # mevcut RNG state güvenlidir.
            save_checkpoint(
                latest_path, model, optimizer, lr_sched,
                epoch_idx, completed_batch, global_step, best_val,
                epoch_loss_sum, epoch_loss_count, args,
            )

            save_checkpoint(
                step_path, model, optimizer, lr_sched,
                epoch_idx, completed_batch, global_step, best_val,
                epoch_loss_sum, epoch_loss_count, args,
            )

            # DÜZELTME (disk doluşu fix'i): sadece en yeni keep_last_steps
            # tane step_*.pt tutulur, eskileri silinir. latest.pt/best.pt/
            # epoch_X.pt bundan etkilenmez.
            cleanup_old_step_checkpoints(checkpoint_dir, keep_last_steps)

            print(f"  [+] latest.pt güncellendi (global step {global_step})")
            print(f"  [+] Kalıcı step checkpoint: {os.path.basename(step_path)}")

    avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)

    return avg_loss, global_step, epoch_loss_sum, epoch_loss_count


# ============================================================
# VALIDATION
# ============================================================

def run_validation(model, loader, scheduler, lambdas, global_step, lmin_warmup_steps, device=None):
    total_loss_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            # DÜZELTME (MPS/GPU fix'i): batch'i model ile aynı cihaza taşı.
            if device is not None:
                batch = move_batch_to_device(batch, device)

            out = model(batch, scheduler=scheduler)

            Ltotal, parts = compute_losses(
                out, model, lambdas, scheduler,
                global_step=global_step, warmup_steps=lmin_warmup_steps,
            )

            total_loss_sum += parts["Ltotal"]
            n_batches += 1

    return total_loss_sum / max(n_batches, 1)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------
    train_log_path = os.path.join(args.checkpoint_dir, "train.log")
    log_file = open(train_log_path, "a", encoding="utf-8", buffering=1)

    def log(message=""):
        print(message)
        log_file.write(str(message) + "\n")
        log_file.flush()

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------
    csv_path = os.path.join(args.checkpoint_dir, "train_log.csv")
    csv_is_new = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    if csv_is_new:
        csv_writer.writerow(["epoch", "global_step", "split"] + METRIC_KEYS)
        csv_file.flush()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------
    log("")
    log("=" * 70)
    log("LCIB_DiffusionSat TRAINING")
    log("=" * 70)
    log(f"Checkpoint directory: {args.checkpoint_dir}")
    log(f"Step checkpoint frequency: every {args.save_every_steps} steps")
    log(f"Step checkpoints kept: last {args.keep_last_steps} (0 = keep all)")
    log(f"Lmin warm-up: {args.lmin_warmup_steps} steps")
    log("")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------
    log("Dataset yükleniyor...")

    full_dataset = LCIBDataset()
    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val

    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    log(f"Train: {n_train} sample | Val: {n_val} sample")

    # --------------------------------------------------------
    # DETERMINISTIC TRAIN SAMPLER
    # --------------------------------------------------------
    train_sampler = EpochRandomSampler(train_set, seed=args.seed, epoch=0)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    log("")
    log("Model yükleniyor (SatUNet dahil, biraz sürebilir)...")

    model = LCIBPipeline(checkpoint_path=CHECKPOINT_PATH)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # --------------------------------------------------------
    # DEVICE (MPS/CPU)
    # --------------------------------------------------------
    #
    # DÜZELTME (MPS/GPU fix'i): model ve scheduler'ın alphas_cumprod'u
    # burada bir kere cihaza taşınıyor. Batch'ler her adımda
    # run_train_epoch/run_validation içinde taşınıyor (move_batch_to_device).
    #
    device = get_device()
    log(f"[+] Cihaz: {device}")

    model = model.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------
    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_val = float("inf")
    epoch_loss_sum = 0.0
    epoch_loss_count = 0

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------
    if args.resume is not None and os.path.exists(args.resume):
        log("")
        log(f"[+] Resume checkpoint: {args.resume}")

        (
            start_epoch, start_batch, global_step, best_val,
            epoch_loss_sum, epoch_loss_count,
        ) = load_checkpoint(args.resume, model, optimizer, lr_sched)

        log("[+] Checkpoint yüklendi.")
        log(f"[+] Epoch: {start_epoch}")
        log(f"[+] Tamamlanan batch: {start_batch}")
        log(f"[+] Global step: {global_step}")
        log(f"[+] Epoch loss count: {epoch_loss_count}")
        log("[+] Eğitim tam olarak kaldığı noktadan devam edecek.")

    elif args.resume is not None:
        log(f"[!] Resume checkpoint bulunamadı: {args.resume}")
        log("[!] Eğitim sıfırdan başlatılıyor.")

    # --------------------------------------------------------
    # ALREADY COMPLETE?
    # --------------------------------------------------------
    if start_epoch >= args.epochs:
        log("")
        log("[!] Checkpoint zaten hedef epoch sayısına ulaşmış.")
        log(f"[!] Checkpoint epoch: {start_epoch}")
        log(f"[!] Hedef epoch: {args.epochs}")

        csv_file.flush()
        csv_file.close()
        log_file.close()
        return

    # --------------------------------------------------------
    # PROGRESS STATE
    # --------------------------------------------------------
    #
    # Bu dictionary özellikle Ctrl+C için kullanılıyor.
    #
    # main()'in içinde run_train_epoch çalışırken local
    # değişkenlerin güncellenmesini beklemek yerine,
    # her batch tamamlandığında bu state güncelleniyor.
    #
    progress_state = {
        "epoch": start_epoch,
        "batch_in_epoch": start_batch,
        "global_step": global_step,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_loss_count": epoch_loss_count,
        "rng_before_batch": None,
    }

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------
    log("")
    log("[+] Eğitim başlıyor...")
    log(f"[+] Başlangıç epoch: {start_epoch}")
    log(f"[+] Başlangıç batch: {start_batch + 1}")
    log(f"[+] Global step: {global_step}")
    log(f"[+] Hedef epoch: {args.epochs}")
    log("")

    try:
        for epoch in range(start_epoch, args.epochs):
            # ------------------------------------------------
            # Resume edilen ilk epoch
            # ------------------------------------------------
            if epoch == start_epoch:
                epoch_start_batch = start_batch
                current_epoch_loss_sum = epoch_loss_sum
                current_epoch_loss_count = epoch_loss_count
            else:
                epoch_start_batch = 0
                current_epoch_loss_sum = 0.0
                current_epoch_loss_count = 0

            # ------------------------------------------------
            # Progress state
            # ------------------------------------------------
            progress_state["epoch"] = epoch
            progress_state["batch_in_epoch"] = epoch_start_batch
            progress_state["global_step"] = global_step
            progress_state["epoch_loss_sum"] = current_epoch_loss_sum
            progress_state["epoch_loss_count"] = current_epoch_loss_count
            progress_state["rng_before_batch"] = None

            # ------------------------------------------------
            # Epoch
            # ------------------------------------------------
            t0 = time.time()

            log("")
            log(f"── Epoch {epoch} / {args.epochs - 1} ──")

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------
            (
                train_loss, global_step,
                current_epoch_loss_sum, current_epoch_loss_count,
            ) = run_train_epoch(
                model=model,
                loader=train_loader,
                sampler=train_sampler,
                scheduler=scheduler,
                optimizer=optimizer,
                lr_sched=lr_sched,
                lambdas=args.lambdas,
                epoch_idx=epoch,
                start_batch=epoch_start_batch,
                global_step=global_step,
                log_every=args.log_every,
                csv_writer=csv_writer,
                csv_file=csv_file,
                lmin_warmup_steps=args.lmin_warmup_steps,
                save_every_steps=args.save_every_steps,
                checkpoint_dir=args.checkpoint_dir,
                best_val=best_val,
                args=args,
                progress_state=progress_state,
                epoch_loss_sum=current_epoch_loss_sum,
                epoch_loss_count=current_epoch_loss_count,
                keep_last_steps=args.keep_last_steps,
                device=device,
            )

            # ------------------------------------------------
            # Epoch tamamen tamamlandı
            # ------------------------------------------------
            progress_state["batch_in_epoch"] = len(train_loader)
            progress_state["global_step"] = global_step
            progress_state["epoch_loss_sum"] = current_epoch_loss_sum
            progress_state["epoch_loss_count"] = current_epoch_loss_count
            progress_state["rng_before_batch"] = None

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------
            val_loss = run_validation(
                model=model,
                loader=val_loader,
                scheduler=scheduler,
                lambdas=args.lambdas,
                global_step=global_step,
                lmin_warmup_steps=args.lmin_warmup_steps,
                device=device,
            )

            # ------------------------------------------------
            # CSV VALIDATION
            # ------------------------------------------------
            val_row = ["" for _ in METRIC_KEYS]
            val_row[METRIC_KEYS.index("Ltotal")] = val_loss

            csv_writer.writerow([epoch, global_step, "val"] + val_row)
            csv_file.flush()

            # ------------------------------------------------
            # EPOCH LOG
            # ------------------------------------------------
            dt = time.time() - t0

            log(
                f"Epoch {epoch} bitti ({dt:.1f}s) | "
                f"train Ltotal={train_loss:.4f} | val Ltotal={val_loss:.4f}"
            )

            # ------------------------------------------------
            # BEST CHECKPOINT
            # ------------------------------------------------
            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(args.checkpoint_dir, "best.pt")

                save_checkpoint(
                    best_path, model, optimizer, lr_sched,
                    # Epoch tamamen bitti.
                    # Bir sonraki çalıştırılacak epoch:
                    epoch + 1,
                    0,
                    global_step,
                    best_val,
                    # Yeni epoch başlayacağı için:
                    0.0,
                    0,
                    args,
                )

                log(f"[+] Yeni best checkpoint: {best_path}")

            # ------------------------------------------------
            # EPOCH CHECKPOINT
            # ------------------------------------------------
            if epoch % args.save_every == 0 or epoch == args.epochs - 1:
                epoch_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")

                save_checkpoint(
                    epoch_path, model, optimizer, lr_sched,
                    # Epoch tamamlandı.
                    # Resume bir sonraki epoch'tan başlayacak.
                    epoch + 1,
                    0,
                    global_step,
                    best_val,
                    0.0,
                    0,
                    args,
                )

                log(f"[+] Epoch checkpoint: {epoch_path}")

            # ------------------------------------------------
            # LATEST CHECKPOINT
            # ------------------------------------------------
            #
            # Epoch tamamen bittikten sonra latest.pt:
            #
            # epoch = epoch + 1
            # batch = 0
            #
            # şeklinde tutuluyor.
            #
            # Böylece resume doğrudan bir sonraki epoch'a geçer.
            #
            latest_path = os.path.join(args.checkpoint_dir, "latest.pt")

            save_checkpoint(
                latest_path, model, optimizer, lr_sched,
                epoch + 1,
                0,
                global_step,
                best_val,
                0.0,
                0,
                args,
            )

            log("[+] latest.pt güncellendi.")

            # Bir sonraki epoch için state sıfırlanır.
            start_batch = 0
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

    except KeyboardInterrupt:
        log("")
        log("!" * 70)
        log("[!] CTRL+C / durdurma isteği alındı.")
        log("[!] Eğitim güvenli şekilde checkpoint'e yazılıyor...")

        # ----------------------------------------------------
        # Ctrl+C sırasında EXACT STATE
        # ----------------------------------------------------
        current_epoch = progress_state["epoch"]
        current_batch = progress_state["batch_in_epoch"]
        current_global_step = progress_state["global_step"]
        current_epoch_loss_sum = progress_state["epoch_loss_sum"]
        current_epoch_loss_count = progress_state["epoch_loss_count"]

        # Eğer halen bir batch işleniyorsa,
        # o batch'in BAŞLANGIÇ RNG state'ini kullan.
        #
        # Böylece batch henüz tamamlanmamış kabul edilir.
        #
        rng_state = progress_state["rng_before_batch"]
        if rng_state is None:
            rng_state = get_rng_state()

        latest_path = os.path.join(args.checkpoint_dir, "latest.pt")

        save_checkpoint(
            latest_path, model, optimizer, lr_sched,
            current_epoch,
            current_batch,
            current_global_step,
            best_val,
            current_epoch_loss_sum,
            current_epoch_loss_count,
            args,
            rng_state=rng_state,
        )

        log(f"[+] latest.pt kaydedildi: {latest_path}")
        log(f"[+] Epoch: {current_epoch}")
        log(f"[+] Tamamlanan batch: {current_batch}")
        log(f"[+] Global step: {current_global_step}")
        log(f"[+] Bir sonraki çalıştırmada {current_batch + 1}. batch'ten devam edilecek.")
        log("[+] Eğitim güvenli şekilde durduruldu.")
        log("!" * 70)

        csv_file.flush()
        csv_file.close()
        log_file.close()
        return

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------
    csv_file.flush()
    csv_file.close()

    log("")
    log("=" * 70)
    log("EĞİTİM TAMAMLANDI!")
    log("=" * 70)
    log(f"Toplam global step: {global_step}")
    log(f"Best validation loss: {best_val:.6f}")

    log_file.close()


if __name__ == "__main__":
    main()

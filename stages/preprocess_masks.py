import os
import numpy as np
from PIL import Image
from pathlib import Path

LCIB_DATASET_DIR  = "/Volumes/KINGSTON/LCIB_DiffusionSat/lcib_dataset2"
OUTPUT_DIR        = "/Volumes/KINGSTON/LCIB_DiffusionSat/LCIB_project/binary_masks"
THRESHOLD         = 10   # 10'dan büyük piksel → 1 (bulut/gölge), küçük → 0 (temiz)

def to_binary(arr, threshold=THRESHOLD):
    """Grayscale maskeyi binary'e çevir."""
    return (arr > threshold).astype(np.uint8) * 255  # 0 veya 255


def process_dataset(lcib_dir, output_dir, threshold=THRESHOLD):
    lcib_dir   = Path(lcib_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    biome_dirs = sorted([
        d for d in lcib_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    print(f"Toplam {len(biome_dirs)} biome klasörü bulundu.")

    total_masks   = 0
    total_shadows = 0

    for biome_dir in biome_dirs:
        biome = biome_dir.name

        sample_dirs = sorted([
            d for d in biome_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        print(f"\n[{biome}] {len(sample_dirs)} klasör bulundu.")

        for sample_dir in sample_dirs:
            # GÜNCELLEME: çıktı yapısını da biome katmanıyla aynala
            # (binary_masks/<biome>/output_.../...) — dataset.py'de
            # okuma tarafı da bu yapıyla eşleşecek.
            out_sample = output_dir / biome / sample_dir.name
            out_sample.mkdir(parents=True, exist_ok=True)

            mask_files   = sorted([f for f in sample_dir.glob("*_mask.png") if not f.name.startswith("._")])
            shadow_files = sorted([f for f in sample_dir.glob("*_shadow.png") if not f.name.startswith("._")])

            for mask_path in mask_files:
                arr    = np.array(Image.open(mask_path).convert("L"))
                binary = to_binary(arr, threshold)
                out_path = out_sample / mask_path.name
                Image.fromarray(binary).save(out_path)
                total_masks += 1

            for shadow_path in shadow_files:
                arr    = np.array(Image.open(shadow_path).convert("L"))
                binary = to_binary(arr, threshold)
                out_path = out_sample / shadow_path.name
                Image.fromarray(binary).save(out_path)
                total_shadows += 1

            print(f"  {sample_dir.name}: {len(mask_files)} mask, {len(shadow_files)} shadow dönüştürüldü.")

    print(f"\nTamamlandı! Toplam {total_masks} mask, {total_shadows} shadow → {output_dir}")


def verify_binary(output_dir):
    """Dönüşümü doğrula — random bir dosyayı kontrol et."""
    output_dir = Path(output_dir)
    files = list(output_dir.rglob("*_mask.png"))
    if not files:
        print("Doğrulanacak dosya bulunamadı.")
        return

    import random
    sample = random.choice(files)
    arr = np.array(Image.open(sample))
    unique = np.unique(arr)
    print(f"\nDoğrulama — {sample.name}")
    print(f"  Unique values: {unique}")
    print(f"  Binary mi: {set(unique).issubset({0, 255})}")
    print(f"  Bulut alanı oranı: {(arr == 255).mean():.3f}")


if __name__ == "__main__":
    process_dataset(LCIB_DATASET_DIR, OUTPUT_DIR)
    verify_binary(OUTPUT_DIR)
# LCIB_DiffusionSat

Fizik-tabanlı bulut/gölge sentezi — DiffusionSat backbone'u fine-tune edilerek,
uydu görüntülerine güneş geometrisine göre fiziksel olarak tutarlı bulut ve
gölge overlay'leri üretmeyi hedefleyen proje.

## Kurulum

```bash
conda create -n diffusionsat python=3.10
conda activate diffusionsat
python -m pip install torch torchvision diffusers==0.17.0.dev0 transformers pillow numpy"<2" huggingface_hub==0.16.4
```

**Önemli:** `pip` yerine her zaman `python -m pip` kullanın (yanlış Python
kurulumuna işaret edebiliyor).

## Checkpoint ve Dataset (repo'da YOK — `.gitignore`'da)

Bu ikisi çok büyük olduğu için repo'ya dahil edilmedi, elle indirilmesi/
yerleştirilmesi gerekiyor:

- **Checkpoint:** `finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64` (Zenodo'dan)
  → `stages/stage1_encoder.py` ve `stages/train.py` içindeki `CHECKPOINT_PATH`
  değişkenini kendi yolunuza göre güncelleyin.
- **Dataset:** `lcib_dataset/` (ham veri) ve `clean_dataset/` (temiz görüntüler)
  → `stages/dataset.py` içindeki `LCIB_DATASET_DIR`, `CLEAN_DATASET_DIR`
  yollarını güncelleyin.
- **Binary maskeler:** repo'da yok, aşağıdaki komutla siz üretmelisiniz:
```bash
  python stages/preprocess_masks.py
```

## Pipeline Yapısı (`stages/`)

| Dosya | Ne yapıyor |
|---|---|
| `dataset.py` | Veri yükleme, metadata encode (9 boyutlu vektör) |
| `preprocess_masks.py` | Ham gri-tonlamalı maskeleri binary'e çevirir |
| `stage1_encoder.py` | TerrainEncoder (çok ölçekli zemin feature) + VAEEncoder |
| `stage2_metadata.py` | Metadata → embedding (`em`) |
| `stage3_displacement.py` | Displacement tahmini + latent warp (Stage 3+4) |
| `stage5_automask.py` | Auto Mask Module — güven haritası (`Minfo`) |
| `stage6_local_attention.py` | Window-based lokal terrain attention |
| `stage7_film.py` | FiLM conditioning (opacity + shadow strength) |
| `stage8_diffusion.py` | UNet'e conditioning enjeksiyonu (mid-block, ControlNet tarzı) |
| `sat_unet.py` | DiffusionSat'ın orijinal UNet sınıfı (değiştirilmedi) |
| `stage9_losses.py` | 6 loss fonksiyonu + toplam ağırlıklı loss |
| `stage10_train.py` | Tüm trainable modülleri birleştiren pipeline sınıfı |
| `train.py` | Gerçek eğitim döngüsü (checkpoint/resume destekli) |

## Eğitimi Çalıştırma

```bash
python stages/train.py --epochs 5 --batch-size 2 --log-every 5
```

Checkpoint'ler `checkpoints/` klasörüne kaydedilir (repo'da yok, otomatik
oluşturulur). Devam etmek için:

```bash
python stages/train.py --resume checkpoints/epoch_3.pt
```


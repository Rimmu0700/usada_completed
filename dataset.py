"""
dataset.py
==========
Mendefinisikan:
  1. Transform (preprocessing tensor) untuk train, val, test
  2. PlantDataset  → class Dataset kustom
  3. get_dataloaders() → fungsi pembuatan DataLoader siap pakai

REVISI MULTI-MODEL:
  - get_dataloaders() sekarang menerima parameter batch_size,
    karena ResNet50, MobileNet, dan ViT punya batch_size berbeda
    (lihat MODEL_HYPERPARAMS di config.py)
  - Test loader TETAP batch_size=1 untuk semua model (ukur ms/gambar yang adil)

ALUR GAMBAR → TENSOR:
  File JPG di disk
  → PIL Image (H × W × C, uint8, nilai 0-255)
  → Resize 224×224
  → ToTensor: ubah ke float32, nilai 0.0-1.0, shape (C, H, W)
  → Normalize: (pixel - mean) / std
  → Tensor siap masuk model (N, C, H, W) setelah dibatch DataLoader

MENGAPA NORMALIZE SAMA UNTUK KETIGA MODEL?
  ResNet50, MobileNetV3, dan ViT-Base/16 semuanya menggunakan bobot
  pretrained ImageNet1K, sehingga mean/std normalisasi yang sama berlaku
  untuk ketiganya — ini bagian dari kontrak pretraining ImageNet.
"""

import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from config import (
    AUG_TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, MEAN, STD,
    NUM_WORKERS, PIN_MEMORY,
    CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES,
    USE_WEIGHTED_SAMPLER
)


# ==============================================================
# 1. TRANSFORM PIPELINE
# ==============================================================

train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
    T.RandomErasing(p=0.1, scale=(0.02, 0.08)),
])

val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ==============================================================
# 2. DATASET CLASS
# ==============================================================

class PlantDataset(Dataset):
    """
    Dataset kustom untuk tanaman obat.

    Struktur folder:
        root_dir/
            Symphytum_officinale/
                img_001.jpg
            Euchresta_horsfieldii/
                img_001.jpg
            ...
    """

    def __init__(self, root_dir: str, transform=None, split: str = ""):
        self.root_dir  = root_dir
        self.transform = transform
        self.split     = split
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(self.root_dir, cls_name)

            if not os.path.isdir(cls_dir):
                print(f"[WARNING] Folder kelas tidak ditemukan: {cls_dir}")
                continue

            label = CLASS_TO_IDX[cls_name]

            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(cls_dir, fname)
                    self.samples.append((full_path, label))

        print(f"[DATASET] Split '{self.split}' → {len(self.samples)} gambar dari {len(CLASS_NAMES)} kelas")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_distribution(self) -> dict:
        dist = {cls: 0 for cls in CLASS_NAMES}
        for _, label in self.samples:
            dist[CLASS_NAMES[label]] += 1
        return dist

    def get_sample_weights(self) -> list:
        dist = self.get_class_distribution()
        class_weights = {cls: 1.0 / count for cls, count in dist.items() if count > 0}
        sample_weights = [class_weights[CLASS_NAMES[label]] for _, label in self.samples]
        return sample_weights


# ==============================================================
# 3. DATALOADER FACTORY
# ==============================================================

def get_dataloaders(batch_size: int = None):
    """
    Membuat DataLoader untuk train, val, dan test.

    Parameter:
        batch_size : ukuran batch untuk train & val loader.
                     Jika None, pakai default config.BATCH_SIZE.
                     Test loader SELALU batch_size=1, terlepas dari
                     parameter ini, agar inference time per-gambar
                     terukur adil & konsisten antar model.

    Return:
        (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        from config import BATCH_SIZE as _default_bs
        batch_size = _default_bs

    train_dataset = PlantDataset(root_dir=AUG_TRAIN_DIR, transform=train_transform, split="train")
    val_dataset   = PlantDataset(root_dir=VAL_DIR,       transform=val_transform,   split="val")
    test_dataset  = PlantDataset(root_dir=TEST_DIR,      transform=test_transform,  split="test")

    sampler = None
    shuffle = True
    if USE_WEIGHTED_SAMPLER:
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
        print("[DATALOADER] WeightedRandomSampler AKTIF — kelas minoritas di-oversample")

    train_loader = DataLoader(
        dataset     = train_dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        sampler     = sampler,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        drop_last   = True,
    )

    val_loader = DataLoader(
        dataset     = val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
    )

    # [PENTING] Test loader SELALU batch=1 — agar ms/gambar comparable
    # antar ResNet50, MobileNet, dan ViT tanpa bias ukuran batch.
    test_loader = DataLoader(
        dataset     = test_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
    )

    print("\n[DATALOADER] Ringkasan:")
    print(f"  Batch size    : {batch_size}")
    print(f"  Train batches : {len(train_loader)} ({len(train_dataset)} gambar)")
    print(f"  Val batches   : {len(val_loader)} ({len(val_dataset)} gambar)")
    print(f"  Test samples  : {len(test_dataset)} gambar (batch=1)")
    print(f"  Device target : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train_dist = train_dataset.get_class_distribution()
    print(f"\n[CLASS DISTRIBUTION - Train (setelah augmentasi)]")
    counts = list(train_dist.values())
    for cls, count in train_dist.items():
        bar = "█" * int(count / max(counts) * 30) if max(counts) > 0 else ""
        print(f"  {cls:28s}: {count:4d} {bar}")
    if counts and min(counts) > 0:
        print(f"  Rasio imbalance: {max(counts)/min(counts):.2f}x")

    return train_loader, val_loader, test_loader


# ==============================================================
# QUICK TEST
# ==============================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

    images, labels = next(iter(train_loader))

    print(f"\n[TENSOR CHECK]")
    print(f"  Batch tensor shape : {images.shape}")
    print(f"  Dtype              : {images.dtype}")
    print(f"  Min pixel value    : {images.min():.4f}")
    print(f"  Max pixel value    : {images.max():.4f}")
    print(f"  Label sample       : {labels[:8].tolist()}")

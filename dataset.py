import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from config import (
    AUG_TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, MEAN, STD,
    NUM_WORKERS, PIN_MEMORY,
    CLASS_NAMES, CLASS_TO_IDX,
    USE_WEIGHTED_SAMPLER,
    RANDOM_SEED,
)

# Training pipeline transformation rules implementing aggressive balance regularizations
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

# Validation pipeline processing configuration maintaining strict verification conditions
val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# Test processing rules applied sequentially across performance measurement steps
test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# Core dataset loader parsing nested category layouts from storage arrays safely
class PlantDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, split: str = ""):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.samples = []
        self._load_samples()

    # Search directory structure to collect matching images and track integer label associations
    def _load_samples(self):
        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = CLASS_TO_IDX[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(cls_dir, fname)
                    self.samples.append((full_path, label))

    # Return cumulative sample array size for length tracking hooks
    def __len__(self) -> int:
        return len(self.samples)

    # Resolve target image array, apply structural transformations, and load tensors dynamically
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    # Track complete occurrence counts mapped individually across label types
    def get_class_distribution(self) -> dict:
        dist = {cls: 0 for cls in CLASS_NAMES}
        for _, label in self.samples:
            dist[CLASS_NAMES[label]] += 1
        return dist

    # Compute custom sampling allocations to resolve class representation imbalances
    def get_sample_weights(self) -> list:
        dist = self.get_class_distribution()
        class_weights = {cls: 1.0 / count for cls, count in dist.items() if count > 0}
        sample_weights = [class_weights[CLASS_NAMES[label]] for _, label in self.samples]
        return sample_weights


# Factory controller delivering production data streaming pipelines
def get_dataloaders(batch_size: int = None):
    if batch_size is None:
        from config import BATCH_SIZE as _default_bs
        batch_size = _default_bs

    train_dataset = PlantDataset(root_dir=AUG_TRAIN_DIR, transform=train_transform, split="train")
    val_dataset = PlantDataset(root_dir=VAL_DIR, transform=val_transform, split="val")
    test_dataset = PlantDataset(root_dir=TEST_DIR, transform=test_transform, split="test")

    # --- FIX SEED: generator khusus supaya urutan shuffle/sampling deterministik ---
    # DataLoader/WeightedRandomSampler punya generator acak sendiri yang TIDAK otomatis
    # ikut ter-kunci oleh torch.manual_seed() global di train.py, jadi harus di-seed manual.
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    sampler = None
    shuffle = True
    if USE_WEIGHTED_SAMPLER:
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=g)
        shuffle = False

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        generator=g if sampler is None else None,  # generator hanya valid dipakai salah satu: shuffle ATAU sampler
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
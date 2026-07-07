import os
import shutil
import random
from pathlib import Path
from config import (
    DATA_SOURCE, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    CLASS_NAMES, RANDOM_SEED
)

# Verify that all target class folders exist within the source dataset directory
def validate_source_folders():
    print("======================================================================")
    print("DATASET FOLDER STRUCTURE VALIDATION")
    print("======================================================================")
    if not os.path.exists(DATA_SOURCE):
        raise FileNotFoundError(f"Source folder dataset_source not found at {DATA_SOURCE}")
    actual_folders = set(f for f in os.listdir(DATA_SOURCE) if os.path.isdir(os.path.join(DATA_SOURCE, f)))
    expected_folders = set(CLASS_NAMES)
    missing = expected_folders - actual_folders
    extra = actual_folders - expected_folders
    if missing:
        print(f"\n[ERROR] The following folders are defined in CLASS_NAMES but MISSING in dataset_source:")
        for f in missing:
            print(f"    - {f}")
        raise FileNotFoundError("Verify CLASS_NAMES spelling in config.py; it must be exact and case-sensitive.")
    if extra:
        print(f"\n[INFO] The following folders EXIST in dataset_source but will be IGNORED:")
        for f in sorted(extra):
            print(f"    - {f} (skipped)")
    print(f"\n[OK] All {len(CLASS_NAMES)} class directories have been successfully located.\n")

# Create the training, validation, and testing folder structure
def create_split_dirs():
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASS_NAMES:
            Path(os.path.join(split, cls)).mkdir(parents=True, exist_ok=True)
    print("[INFO] Target split directories created successfully.")

# Split and copy images into training, validation, and test subsets based on defined ratios
def split_dataset():
    random.seed(RANDOM_SEED)
    total_per_class = {}
    for cls in CLASS_NAMES:
        source_cls_dir = os.path.join(DATA_SOURCE, cls)
        if not os.path.exists(source_cls_dir):
            print(f"[WARNING] Directory not found: {source_cls_dir}")
            continue
        all_files = [f for f in os.listdir(source_cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(all_files) == 0:
            print(f"[WARNING] No image assets found in: {source_cls_dir}")
            continue
        random.shuffle(all_files)
        
        # Calculate split indices
        n_total = len(all_files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        # Copy files to designated partitions
        for fname in train_files:
            shutil.copy(os.path.join(source_cls_dir, fname), os.path.join(TRAIN_DIR, cls, fname))
        for fname in val_files:
            shutil.copy(os.path.join(source_cls_dir, fname), os.path.join(VAL_DIR, cls, fname))
        for fname in test_files:
            shutil.copy(os.path.join(source_cls_dir, fname), os.path.join(TEST_DIR, cls, fname))
            
        total_per_class[cls] = {"total": n_total, "train": n_train, "val": n_val, "test": n_test}
        print(f"[SPLIT] {cls:28s} Total: {n_total:4d} | Train: {n_train:4d} | Val: {n_val:3d} | Test: {n_test:3d}")
        
    print("\n======================================================================")
    total_all = sum(v["total"] for v in total_per_class.values())
    total_train = sum(v["train"] for v in total_per_class.values())
    total_val = sum(v["val"] for v in total_per_class.values())
    total_test = sum(v["test"] for v in total_per_class.values())
    
    # Summarize partitioning results
    print(f"[SUMMARY] Total dataset images : {total_all}")
    print(f"          Training subset      : {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"          Validation subset    : {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"          Testing subset       : {total_test} ({total_test/total_all*100:.1f}%)")
    
    min_val = min(v["val"] for v in total_per_class.values())
    min_test = min(v["test"] for v in total_per_class.values())
    if min_val < 5 or min_test < 5:
        print(f"\n[WARNING] Found target class with validation count {min_val} or testing count {min_test} under 5 samples.")
        
    print("[INFO] Dataset split operation concluded. Source files remain unchanged.")
    print("[INFO] These subsets will be shared universally across ResNet50, MobileNet, and ViT.")
    print("======================================================================")
    return total_per_class

if __name__ == "__main__":
    validate_source_folders()
    create_split_dirs()
    split_dataset()
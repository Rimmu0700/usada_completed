"""
split_dataset.py
================
Langkah 1: Membaca data asli dari dataset_source/
           lalu membagi ke dataset_split/train, val, test.

Tidak terkait pemilihan model — split dataset dilakukan SATU KALI
dan dipakai bersama oleh ResNet50, MobileNetV3-Small, dan ViT-Base/16,
agar perbandingan ketiga model adil (data train/val/test identik).

Alur:
  dataset_source/  →  dataset_split/train/  (70%)
                   →  dataset_split/val/    (15%)
                   →  dataset_split/test/   (15%)

PENTING:
  - Data ASLI tidak disentuh / dipindah, hanya di-copy.
  - Augmentasi belum dilakukan di sini.
  - Split dilakukan secara stratified (proporsional per kelas).
"""

import os
import shutil
import random
from pathlib import Path
from config import (
    DATA_SOURCE, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    CLASS_NAMES, RANDOM_SEED
)


def validate_source_folders():
    """Cek apakah semua folder di CLASS_NAMES benar-benar ada di dataset_source/."""
    print("=" * 70)
    print("VALIDASI FOLDER DATASET")
    print("=" * 70)

    if not os.path.exists(DATA_SOURCE):
        raise FileNotFoundError(f"Folder dataset_source/ tidak ditemukan: {DATA_SOURCE}")

    actual_folders = set(
        f for f in os.listdir(DATA_SOURCE)
        if os.path.isdir(os.path.join(DATA_SOURCE, f))
    )
    expected_folders = set(CLASS_NAMES)

    missing = expected_folders - actual_folders
    extra   = actual_folders - expected_folders

    if missing:
        print(f"\n[ERROR] Folder berikut ada di CLASS_NAMES tapi TIDAK ditemukan di dataset_source/:")
        for f in missing:
            print(f"    - {f}")
        raise FileNotFoundError(
            "Periksa kembali penulisan CLASS_NAMES di config.py, "
            "harus sama persis (case-sensitive) dengan nama folder."
        )

    if extra:
        print(f"\n[INFO] Folder berikut ADA di dataset_source/ tapi TIDAK dipakai saat ini:")
        for f in sorted(extra):
            print(f"    - {f}  (dilewati)")

    print(f"\n[OK] Semua {len(CLASS_NAMES)} folder kelas ditemukan dan valid.\n")


def create_split_dirs():
    """Buat struktur folder dataset_split/ jika belum ada."""
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASS_NAMES:
            Path(os.path.join(split, cls)).mkdir(parents=True, exist_ok=True)
    print("[INFO] Folder dataset_split/ berhasil dibuat.")


def split_dataset():
    """Membagi setiap kelas dari dataset_source/ ke train/val/test."""
    random.seed(RANDOM_SEED)
    total_per_class = {}

    for cls in CLASS_NAMES:
        source_cls_dir = os.path.join(DATA_SOURCE, cls)

        if not os.path.exists(source_cls_dir):
            print(f"[WARNING] Folder tidak ditemukan: {source_cls_dir}")
            continue

        all_files = [
            f for f in os.listdir(source_cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(all_files) == 0:
            print(f"[WARNING] Tidak ada gambar di {source_cls_dir}")
            continue

        random.shuffle(all_files)

        n_total = len(all_files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val   = int(n_total * VAL_RATIO)
        n_test  = n_total - n_train - n_val

        train_files = all_files[:n_train]
        val_files   = all_files[n_train:n_train + n_val]
        test_files  = all_files[n_train + n_val:]

        for fname in train_files:
            shutil.copy(os.path.join(source_cls_dir, fname), os.path.join(TRAIN_DIR, cls, fname))
        for fname in val_files:
            shutil.copy(os.path.join(source_cls_dir, fname), os.path.join(VAL_DIR, cls, fname))
        for fname in test_files:
            shutil.copy(os.path.join(source_cls_dir, fname), os.path.join(TEST_DIR, cls, fname))

        total_per_class[cls] = {"total": n_total, "train": n_train, "val": n_val, "test": n_test}

        print(f"[SPLIT] {cls:28s} → Total: {n_total:4d} | Train: {n_train:4d} | Val: {n_val:3d} | Test: {n_test:3d}")

    print("\n" + "=" * 70)
    total_all   = sum(v["total"] for v in total_per_class.values())
    total_train = sum(v["train"] for v in total_per_class.values())
    total_val   = sum(v["val"]   for v in total_per_class.values())
    total_test  = sum(v["test"]  for v in total_per_class.values())
    print(f"[SUMMARY] Total gambar : {total_all}")
    print(f"          Train        : {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"          Val          : {total_val}   ({total_val/total_all*100:.1f}%)")
    print(f"          Test         : {total_test}  ({total_test/total_all*100:.1f}%)")

    min_val  = min(v["val"]  for v in total_per_class.values())
    min_test = min(v["test"] for v in total_per_class.values())
    if min_val < 5 or min_test < 5:
        print(f"\n[WARNING] Ada kelas dengan val ({min_val}) atau test ({min_test}) < 5 gambar.")

    print("[INFO] Split selesai. Data asli tidak diubah.")
    print("[INFO] Split ini akan dipakai BERSAMA oleh ResNet50, MobileNet, dan ViT.")
    print("=" * 70)

    return total_per_class


if __name__ == "__main__":
    validate_source_folders()
    create_split_dirs()
    split_dataset()

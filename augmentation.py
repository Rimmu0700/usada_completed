import os
import math
import random
from pathlib import Path
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from config import (
    TRAIN_DIR, AUG_TRAIN_DIR,
    CLASS_NAMES, IMAGE_SIZE, RANDOM_SEED,
    TARGET_TRAIN_PER_CLASS, MAX_AUG_MULTIPLIER, MIN_AUG_MULTIPLIER
)


def create_aug_dirs():
    for cls in CLASS_NAMES:
        Path(os.path.join(AUG_TRAIN_DIR, cls)).mkdir(parents=True, exist_ok=True)
    print("[INFO] Folder dataset_augmented/train/ berhasil dibuat.")


def calculate_multiplier(n_original: int) -> int:
    """Hitung berapa kali augmentasi dibutuhkan per kelas agar mendekati target."""
    if n_original == 0:
        return 0
    raw_multiplier = TARGET_TRAIN_PER_CLASS / n_original
    multiplier = math.ceil(raw_multiplier)
    multiplier = max(MIN_AUG_MULTIPLIER, min(multiplier, MAX_AUG_MULTIPLIER))
    return multiplier


def augment_image(img: Image.Image, aug_id: int) -> Image.Image:
    """Terapkan satu set augmentasi acak ke satu gambar PIL."""
    random.seed(RANDOM_SEED + aug_id)

    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    if random.random() > 0.5:
        img = TF.hflip(img)
    if random.random() > 0.7:
        img = TF.vflip(img)

    angle = random.uniform(-30, 30)
    img = TF.rotate(img, angle)

    img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
    img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))

    if random.random() > 0.5:
        w, h   = img.size
        crop_w = int(w * random.uniform(0.80, 0.95))
        crop_h = int(h * random.uniform(0.80, 0.95))
        left   = random.randint(0, w - crop_w)
        top    = random.randint(0, h - crop_h)
        img    = img.crop((left, top, left + crop_w, top + crop_h))
        img    = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    img = TF.adjust_sharpness(img, random.uniform(0.6, 1.8))

    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.3)))

    return img


def generate_augmented_dataset():
    """Generate augmentasi proporsional per kelas, simpan ke dataset_augmented/train/."""
    total_generated = 0
    summary = []

    print("=" * 70)
    print(f"AUGMENTASI PROPORSIONAL — Target: {TARGET_TRAIN_PER_CLASS} gambar/kelas")
    print("[INFO] Hasil ini dipakai BERSAMA oleh ResNet50, MobileNet, dan ViT")
    print("=" * 70)

    for cls in CLASS_NAMES:
        cls_train_dir = os.path.join(TRAIN_DIR, cls)
        cls_aug_dir   = os.path.join(AUG_TRAIN_DIR, cls)

        if not os.path.exists(cls_train_dir):
            print(f"[WARNING] Folder tidak ditemukan: {cls_train_dir}")
            continue

        image_files = [
            f for f in os.listdir(cls_train_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(image_files) == 0:
            print(f"[WARNING] Tidak ada gambar training di {cls_train_dir}")
            continue

        multiplier = calculate_multiplier(len(image_files))
        cls_generated = 0

        for fname in image_files:
            src_path = os.path.join(cls_train_dir, fname)
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Gagal membaca {src_path}: {e}")
                continue

            base_name = os.path.splitext(fname)[0]
            orig_save = os.path.join(cls_aug_dir, f"{base_name}_orig.jpg")
            img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(orig_save, "JPEG", quality=95)
            cls_generated += 1

            for i in range(multiplier):
                aug_img  = augment_image(img.copy(), aug_id=(hash(fname) + i) % 100000)
                aug_name = f"{base_name}_aug_{i+1:03d}.jpg"
                aug_img.save(os.path.join(cls_aug_dir, aug_name), "JPEG", quality=95)
                cls_generated += 1

        total_generated += cls_generated
        summary.append({
            "class": cls, "n_original": len(image_files),
            "multiplier": multiplier, "n_generated": cls_generated,
        })

        print(f"[AUG] {cls:28s} | Asli: {len(image_files):3d} | "
              f"Multiplier: x{multiplier} | Hasil: {cls_generated:4d} gambar")

    print("\n" + "=" * 70)
    print("RINGKASAN AUGMENTASI")
    print("=" * 70)
    for s in summary:
        deviation = s["n_generated"] - TARGET_TRAIN_PER_CLASS
        flag = "✓" if abs(deviation) < 50 else "⚠"
        print(f"  {flag} {s['class']:28s}: {s['n_generated']:4d} gambar "
              f"(target: {TARGET_TRAIN_PER_CLASS}, selisih: {deviation:+d})")

    print(f"\n[SUMMARY] Total gambar augmentasi: {total_generated}")

    counts = [s["n_generated"] for s in summary]
    if counts:
        ratio = max(counts) / min(counts)
        print(f"[SUMMARY] Rasio imbalance SETELAH augmentasi: {ratio:.2f}x "
              f"({'SEIMBANG' if ratio < 1.3 else 'masih ada selisih, cek manual'})")

    print("[INFO] Augmentasi selesai. Val dan test tidak diubah.")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    create_aug_dirs()
    generate_augmented_dataset()

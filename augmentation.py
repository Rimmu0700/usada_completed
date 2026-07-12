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

# Create the directory structure for augmented training data
def create_aug_dirs():
    for cls in CLASS_NAMES:
        Path(os.path.join(AUG_TRAIN_DIR, cls)).mkdir(parents=True, exist_ok=True)
    print("[INFO] Augmented training directory created successfully.")

# Apply random sequence of geometric and photometric transformations to a PIL image
def augment_image(img: Image.Image, aug_id: int) -> Image.Image:
    random.seed(RANDOM_SEED + aug_id)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    
    # Apply random flipping augmentations
    if random.random() > 0.5:
        img = TF.hflip(img)
    if random.random() > 0.7:
        img = TF.vflip(img)
    
    # Apply random rotation and color space adjustments
    angle = random.uniform(-30, 30)
    img = TF.rotate(img, angle)
    img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
    img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
    
    # Perform random cropping and resizing to introduce variation
    if random.random() > 0.5:
        w, h = img.size
        crop_w = int(w * random.uniform(0.80, 0.95))
        crop_h = int(h * random.uniform(0.80, 0.95))
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        img = img.crop((left, top, left + crop_w, top + crop_h))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    
    # Apply sharpness and blur filtering
    img = TF.adjust_sharpness(img, random.uniform(0.6, 1.8))
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.3)))
    return img

# Execute the EXACT target dataset expansion and save output to disk
def generate_augmented_dataset():
    total_generated = 0
    summary = []
    print("======================================================================")
    print(f"EXACT AUGMENTATION TARGET: {TARGET_TRAIN_PER_CLASS} images per class")
    print("[INFO] Generated images will be shared across all architectures.")
    print("======================================================================")
    
    for cls in CLASS_NAMES:
        cls_train_dir = os.path.join(TRAIN_DIR, cls)
        cls_aug_dir = os.path.join(AUG_TRAIN_DIR, cls)
        
        if not os.path.exists(cls_train_dir):
            print(f"[WARNING] Directory not found: {cls_train_dir}")
            continue
            
        image_files = [f for f in os.listdir(cls_train_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(image_files) == 0:
            print(f"[WARNING] No training images in: {cls_train_dir}")
            continue
            
        cls_generated = 0
        
        # Phase 1: Process and save original images up to the exact target
        for fname in image_files:
            if cls_generated >= TARGET_TRAIN_PER_CLASS:
                break # Stop if we somehow already hit 400 originals
                
            src_path = os.path.join(cls_train_dir, fname)
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Failed to read {src_path}: {e}")
                continue
                
            base_name = os.path.splitext(fname)[0]
            orig_save = os.path.join(cls_aug_dir, f"{base_name}_orig.jpg")
            img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR).save(orig_save, "JPEG", quality=95)
            cls_generated += 1
            
        # Phase 2: Cycle through originals and augment EXACTLY until we hit the target limit
        aug_idx = 0
        while cls_generated < TARGET_TRAIN_PER_CLASS:
            fname = image_files[aug_idx % len(image_files)]
            src_path = os.path.join(cls_train_dir, fname)
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                aug_idx += 1
                continue
            
            base_name = os.path.splitext(fname)[0]
            # Use cls_generated in the seed to guarantee unique augmentation per loop
            aug_img = augment_image(img.copy(), aug_id=(hash(fname) + cls_generated) % 100000)
            aug_name = f"{base_name}_aug_{cls_generated:04d}.jpg"
            aug_img.save(os.path.join(cls_aug_dir, aug_name), "JPEG", quality=95)
            
            cls_generated += 1
            aug_idx += 1
                
        total_generated += cls_generated
        summary.append({
            "class": cls, "n_original": len(image_files),
            "n_generated": cls_generated,
        })
        print(f"[AUG] {cls:28s} | Original: {len(image_files):3d} | Total Generated: {cls_generated:4d}")
        
    print("\n======================================================================")
    print("AUGMENTATION SUMMARY")
    print("======================================================================")
    for s in summary:
        deviation = s["n_generated"] - TARGET_TRAIN_PER_CLASS
        flag = "PASS" if deviation == 0 else "FAIL"
        print(f"  {flag} {s['class']:28s}: {s['n_generated']:4d} images (Target: {TARGET_TRAIN_PER_CLASS}, Diff: {deviation:+d})")
    
    print(f"\n[SUMMARY] Total augmented images generated: {total_generated}")
    
    # Verify balance status after expansion
    counts = [s["n_generated"] for s in summary]
    if counts:
        ratio = max(counts) / min(counts)
        balance_status = "PERFECTLY BALANCED" if ratio == 1.0 else "UNBALANCED - Manual review required"
        print(f"[SUMMARY] Post-augmentation imbalance ratio: {ratio:.2f}x ({balance_status})")
        
    print("[INFO] Augmentation complete. Validation and test sets remain unaltered.")
    print("======================================================================")
    return summary

if __name__ == "__main__":
    create_aug_dirs()
    generate_augmented_dataset()
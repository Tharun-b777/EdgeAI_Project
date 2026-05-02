"""
augment_dataset.py
------------------
Augments images from privacy-preservation folders to create a larger
validation dataset for facial blurring pipelines.

Folder structure expected:
    base_dir/
        face/
        Laptop/
        Laptop_Face/
        phones/
        phonesPerson/

Each image gets augmented with:
    1. Original (no change)
    2. Horizontal flip
    3. Brightness increase (+50)
    4. Brightness decrease (-50)
    5. Gaussian blur (simulates motion / low quality)
    6. Rotation +15 degrees
    7. Rotation -15 degrees
    8. Grayscale
    9. Contrast enhancement (CLAHE)
   10. Salt-and-pepper noise

Output is saved to:
    base_dir/augmented/<folder_name>/<original_stem>_aug<N>.jpg
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path

# ------------------------------------------------------------------
# Augmentation functions
# ------------------------------------------------------------------

def aug_original(img):
    return img.copy()

def aug_flip(img):
    return cv2.flip(img, 1)

def aug_brightness_up(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def aug_brightness_down(img, value=50):
    return aug_brightness_up(img, value=-value)

def aug_gaussian_blur(img, ksize=15):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def aug_rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT_101)

def aug_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def aug_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def aug_salt_pepper(img, amount=0.02):
    out = img.copy()
    h, w = out.shape[:2]
    n_salt = int(amount * h * w)
    coords = [np.random.randint(0, s, n_salt) for s in (h, w)]
    out[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, s, n_salt) for s in (h, w)]
    out[coords[0], coords[1]] = 0
    return out

# Ordered list: (suffix, function)
AUGMENTATIONS = [
    ("original",        aug_original),
    ("flip",            aug_flip),
    ("bright_down",     aug_brightness_down),
    ("rotate_plus15",   lambda img: aug_rotate(img, 15)),
]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FOLDERS = ["face", "Laptop", "Laptop_Face", "phones", "phonesPerson"]

# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------

def augment_folder(folder_path: Path, out_root: Path):
    """Augment all images in one folder and save to out_root/<folder_name>/."""
    out_dir = out_root / folder_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in folder_path.iterdir()
              if p.suffix.lower() in SUPPORTED_EXTS]

    if not images:
        print(f"  [SKIP] No images found in {folder_path}")
        return 0

    total = 0
    for img_path in sorted(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read {img_path.name}, skipping.")
            continue

        for suffix, fn in AUGMENTATIONS:
            aug_img = fn(img)
            out_name = f"{img_path.stem}_{suffix}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += 1

        print(f"  [OK] {img_path.name} -> {len(AUGMENTATIONS)} augmentations")

    return total


def main(base_dir: str):
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")

    out_root = base / "augmented"
    out_root.mkdir(exist_ok=True)

    print(f"\nBase directory : {base.resolve()}")
    print(f"Output root    : {out_root.resolve()}")
    print(f"Augmentations  : {len(AUGMENTATIONS)} per image")
    print("-" * 50)

    grand_total = 0
    for folder_name in FOLDERS:
        folder_path = base / folder_name
        if not folder_path.exists():
            print(f"\n[MISSING] Folder not found: {folder_path}")
            continue

        print(f"\nProcessing: {folder_name}/")
        count = augment_folder(folder_path, out_root)
        grand_total += count
        print(f"  -> {count} augmented images saved to augmented/{folder_name}/")

    print("\n" + "=" * 50)
    print(f"Done. Total augmented images saved: {grand_total}")
    print(f"Output location: {out_root.resolve()}")
    print("=" * 50)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment images for facial blurring validation."
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Path to the root folder containing face/, Laptop/, etc."
    )
    args = parser.parse_args()
    main(args.base_dir)

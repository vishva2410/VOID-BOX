#!/usr/bin/env python3
"""
prepare_signatures.py -- Convert a classification-style signature dataset
into YOLO detection format for fine-tuning.

Because the source dataset contains isolated signature images (just the
signature on a white/transparent background), this script:
  1. Collects all PNG/JPG images from the source directory.
  2. Composites each signature onto a random solid-colour background
     (simulating a signature on a document page).
  3. Detects the actual ink region via contour analysis and writes a
     tight YOLO bounding box (class 2 = "signature").
  4. Splits 85/15 into train/val.
  5. Writes a data.yaml compatible with train.py.

Usage:
    python prepare_signatures.py
    python prepare_signatures.py --source /path/to/signatures --epochs 30
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).parent
DEST_DIR = SCRIPT_DIR / "datasets" / "signatures"
SIG_CLASS_ID = 2  # matches PII_CLASSES[2] = "signature"


def find_all_images(source: Path) -> list[Path]:
    """Recursively find all image files under source."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images = []
    for f in source.rglob("*"):
        if f.suffix.lower() in exts and not f.name.startswith("."):
            images.append(f)
    return sorted(images)


def compute_ink_bbox(img_gray: np.ndarray, threshold=200):
    """
    Find the tight bounding box around the actual ink/signature content.
    Assumes the signature is darker than the background.
    Returns (x, y, w, h) in pixel coordinates, or None if nothing found.
    """
    # Threshold: ink pixels are dark
    _, binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Merge all contours into one bounding rect
    all_pts = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_pts)
    return x, y, w, h


def composite_on_background(sig_img: np.ndarray, target_size=640):
    """
    Place the signature on a random document-like background.
    Returns (composite_rgb, bbox_xywh) in pixel coords.
    """
    h_sig, w_sig = sig_img.shape[:2]

    # Random background colours (paper-like: whites, creams, light grays)
    bg_colors = [
        (255, 255, 255), (250, 248, 240), (245, 245, 245),
        (240, 235, 225), (252, 250, 245), (230, 230, 230),
        (248, 246, 238), (255, 253, 248), (235, 232, 225),
    ]
    bg_color = random.choice(bg_colors)

    # Create background
    canvas = np.full((target_size, target_size, 3), bg_color, dtype=np.uint8)

    # Scale signature to fit (30-70% of canvas width)
    scale_factor = random.uniform(0.3, 0.7)
    new_w = int(target_size * scale_factor)
    aspect = h_sig / max(w_sig, 1)
    new_h = int(new_w * aspect)
    new_h = min(new_h, int(target_size * 0.6))  # cap height

    if new_w < 10 or new_h < 10:
        return None, None

    sig_resized = cv2.resize(sig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Random position on canvas
    max_x = target_size - new_w
    max_y = target_size - new_h
    if max_x < 0 or max_y < 0:
        return None, None

    off_x = random.randint(0, max(max_x, 0))
    off_y = random.randint(0, max(max_y, 0))

    # Detect ink region for tight bbox
    if len(sig_resized.shape) == 3:
        gray = cv2.cvtColor(sig_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = sig_resized.copy()
        sig_resized = cv2.cvtColor(sig_resized, cv2.COLOR_GRAY2BGR)

    ink = compute_ink_bbox(gray)

    # Composite
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = sig_resized

    if ink is not None:
        ix, iy, iw, ih = ink
        # Bbox in canvas coords
        bbox = (off_x + ix, off_y + iy, iw, ih)
    else:
        # Fallback: use entire placed region
        bbox = (off_x, off_y, new_w, new_h)

    return canvas, bbox


def xywh_to_yolo(bbox_xywh, img_w, img_h):
    """Convert pixel (x, y, w, h) to YOLO (cx, cy, w, h) normalized."""
    x, y, w, h = bbox_xywh
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.001, min(1.0, nw))
    nh = max(0.001, min(1.0, nh))
    return cx, cy, nw, nh


def main():
    parser = argparse.ArgumentParser(description="Prepare signature dataset for YOLO training")
    parser.add_argument("--source", type=str,
                        default=None,
                        help="Root of the downloaded signature dataset (required)")
    parser.add_argument("--size", type=int, default=640,
                        help="Output image size (default: 640)")
    parser.add_argument("--split", type=float, default=0.85,
                        help="Train fraction (default: 0.85)")
    args = parser.parse_args()

    if not args.source:
        print("ERROR: --source is required. Example:")
        print("  python prepare_signatures.py --source /path/to/signatures")
        return

    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: Source directory not found: {source}")
        return

    # Find all images
    all_images = find_all_images(source)
    print(f"Found {len(all_images)} signature images in {source}")

    if len(all_images) == 0:
        print("No images found. Check the source path.")
        return

    # Create output dirs
    for split in ["train", "val"]:
        (DEST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * args.split)
    splits = {
        "train": all_images[:split_idx],
        "val": all_images[split_idx:],
    }

    total_ok = 0
    total_skip = 0

    for split_name, images in splits.items():
        print(f"\nProcessing {split_name}: {len(images)} images...")
        for i, img_path in enumerate(images):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    total_skip += 1
                    continue

                composite, bbox = composite_on_background(img, target_size=args.size)
                if composite is None:
                    total_skip += 1
                    continue

                # Also generate a second augmented version (different placement)
                composite2, bbox2 = composite_on_background(img, target_size=args.size)

                for ver, (comp, bb) in enumerate([(composite, bbox), (composite2, bbox2)]):
                    if comp is None:
                        continue

                    fname = f"sig_{split_name}_{i:04d}_v{ver}"
                    img_out = DEST_DIR / split_name / "images" / f"{fname}.jpg"
                    lbl_out = DEST_DIR / split_name / "labels" / f"{fname}.txt"

                    cv2.imwrite(str(img_out), comp, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    cx, cy, nw, nh = xywh_to_yolo(bb, args.size, args.size)
                    with open(lbl_out, "w") as f:
                        f.write(f"{SIG_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                    total_ok += 1

            except Exception as e:
                total_skip += 1
                continue

        print(f"  {split_name} done.")

    # Write data.yaml
    data_yaml = DEST_DIR / "data.yaml"
    yaml_content = f"""# Signature Detection Dataset for VoidBox
# Auto-generated by prepare_signatures.py

path: {DEST_DIR.resolve()}
train: train/images
val: val/images

names:
  0: document
  1: face
  2: signature
  3: text_field

# Only class 2 (signature) has labels in this dataset.
# When combined with MIDV-2020 training, all 4 classes get data.
"""
    with open(data_yaml, "w") as f:
        f.write(yaml_content)

    print(f"\n{'='*50}")
    print(f"  Signature Dataset Prepared")
    print(f"{'='*50}")
    print(f"  Images created: {total_ok}")
    print(f"  Skipped:        {total_skip}")
    print(f"  Output:         {DEST_DIR}")
    print(f"  data.yaml:      {data_yaml}")
    print(f"\nTo train:")
    print(f"  python train.py --data {data_yaml} --epochs 30")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
prepare_dataset.py — Validate and prepare the MIDV-2020 dataset for YOLOv8 training.

Ensures the dataset is properly structured, remaps class IDs to our PII scheme,
and generates clean data.yaml configuration.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --verify           # Validation only
    python prepare_dataset.py --dataset-dir datasets/midv2020
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from collections import Counter


# Our target PII class mapping
# This is what VoidBox needs to detect and redact
TARGET_CLASSES = {
    0: "document",
    1: "face",
    2: "signature",
    3: "text_field",
}


def load_data_yaml(dataset_dir: str) -> dict:
    """Load the data.yaml file from the dataset."""
    yaml_path = Path(dataset_dir) / "data.yaml"
    if not yaml_path.exists():
        print(f"ERROR: data.yaml not found at {yaml_path}")
        sys.exit(1)
    
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def analyze_classes(dataset_dir: str, data_config: dict) -> dict:
    """Analyze existing classes and build a remapping dictionary."""
    existing_names = data_config.get("names", {})
    
    # If names is a list, convert to dict
    if isinstance(existing_names, list):
        existing_names = {i: name for i, name in enumerate(existing_names)}
    
    print(f"\nExisting classes in dataset:")
    for cls_id, cls_name in existing_names.items():
        print(f"  {cls_id}: {cls_name}")
    
    # Build remapping: map existing class names to our target classes
    name_to_target = {}
    
    # Flexible matching for common variations
    mappings = {
        "document": 0,  # document, doc, id_card, passport, card
        "doc": 0,
        "id_card": 0,
        "id-card": 0,
        "passport": 0,
        "card": 0,
        "mrp": 0,        # Machine Readable Passport
        "face": 1,
        "photo": 1,
        "face_photo": 1,
        "portrait": 1,
        "signature": 2,
        "sign": 2,
        "signature_caption": 2,  # MIDV-2020 specific
        "text": 3,
        "text_field": 3,
        "text-field": 3,
        "name": 3,
        "field": 3,
        "mrz": 3,         # Machine Readable Zone
    }
    
    remap = {}  # old_id -> new_id
    unmapped = []
    
    for cls_id, cls_name in existing_names.items():
        cls_lower = cls_name.lower().strip()
        if cls_lower in mappings:
            remap[int(cls_id)] = mappings[cls_lower]
        else:
            unmapped.append((cls_id, cls_name))
    
    if unmapped:
        print(f"\n⚠️  Unmapped classes (will be dropped):")
        for cls_id, cls_name in unmapped:
            print(f"  {cls_id}: {cls_name}")
    
    print(f"\nClass remapping:")
    for old_id, new_id in sorted(remap.items()):
        old_name = existing_names.get(old_id, "?")
        new_name = TARGET_CLASSES[new_id]
        print(f"  {old_id} ({old_name}) → {new_id} ({new_name})")
    
    return remap


def remap_labels(dataset_dir: str, remap: dict, dry_run: bool = False):
    """Remap class IDs in all label files."""
    dataset_path = Path(dataset_dir)
    total_remapped = 0
    total_dropped = 0
    
    for split in ["train", "valid", "test"]:
        label_dir = dataset_path / split / "labels"
        if not label_dir.exists():
            continue
        
        split_remapped = 0
        split_dropped = 0
        
        for label_file in sorted(label_dir.glob("*.txt")):
            new_lines = []
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    old_cls = int(parts[0])
                    if old_cls in remap:
                        parts[0] = str(remap[old_cls])
                        new_lines.append(" ".join(parts))
                        split_remapped += 1
                    else:
                        split_dropped += 1
            
            if not dry_run:
                with open(label_file, "w") as f:
                    f.write("\n".join(new_lines) + "\n" if new_lines else "")
        
        total_remapped += split_remapped
        total_dropped += split_dropped
        print(f"  {split}: {split_remapped} annotations remapped, {split_dropped} dropped")
    
    return total_remapped, total_dropped


def write_clean_yaml(dataset_dir: str):
    """Write a clean data.yaml with our target classes."""
    dataset_path = Path(dataset_dir)
    
    yaml_content = {
        "path": str(dataset_path.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": TARGET_CLASSES,
        "nc": len(TARGET_CLASSES),
    }
    
    # Check if test split exists
    if (dataset_path / "test" / "images").exists():
        yaml_content["test"] = "test/images"
    
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Clean data.yaml written to {yaml_path}")


def verify_dataset(dataset_dir: str):
    """Full dataset verification — check structure, labels, image-label pairing."""
    dataset_path = Path(dataset_dir)
    
    print(f"\n{'='*60}")
    print(f"  MIDV-2020 Dataset Verification Report")
    print(f"{'='*60}")
    
    issues = []
    
    # 1. Check data.yaml
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        print(f"\n📄 data.yaml:")
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        for key, val in config.items():
            if key != "names":
                print(f"    {key}: {val}")
            else:
                print(f"    names:")
                if isinstance(val, dict):
                    for k, v in val.items():
                        print(f"      {k}: {v}")
                elif isinstance(val, list):
                    for i, v in enumerate(val):
                        print(f"      {i}: {v}")
    else:
        issues.append("data.yaml not found")
    
    # 2. Check splits
    for split in ["train", "valid", "test"]:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        
        if not img_dir.exists():
            if split != "test":
                issues.append(f"{split}/images directory missing")
            continue
        
        images = set(p.stem for p in img_dir.iterdir() if p.suffix in {".jpg", ".jpeg", ".png", ".bmp"})
        labels = set(p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt") if lbl_dir.exists() else set()
        
        missing_labels = images - labels
        orphan_labels = labels - images
        
        # Count annotations per class
        class_counts = Counter()
        total_anns = 0
        for lbl_file in sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []:
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_counts[int(parts[0])] += 1
                        total_anns += 1
        
        print(f"\n📁 {split}/")
        print(f"    Images: {len(images)}")
        print(f"    Labels: {len(labels)}")
        print(f"    Total annotations: {total_anns}")
        
        if class_counts:
            print(f"    Per-class breakdown:")
            names = config.get("names", {}) if yaml_path.exists() else {}
            if isinstance(names, list):
                names = {i: n for i, n in enumerate(names)}
            for cls_id in sorted(class_counts.keys()):
                cls_name = names.get(cls_id, f"class_{cls_id}")
                print(f"      {cls_id} ({cls_name}): {class_counts[cls_id]}")
        
        if missing_labels:
            print(f"    ⚠️  {len(missing_labels)} images without labels")
            issues.append(f"{split}: {len(missing_labels)} images without labels")
        if orphan_labels:
            print(f"    ⚠️  {len(orphan_labels)} orphan label files")
            issues.append(f"{split}: {len(orphan_labels)} orphan label files")
    
    # Summary
    print(f"\n{'='*60}")
    if issues:
        print(f"⚠️  {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"    • {issue}")
    else:
        print(f"✅ Dataset verification PASSED — ready for training!")
    print(f"{'='*60}")
    
    return len(issues) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and validate MIDV-2020 dataset for VoidBox YOLOv8 training"
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        type=str,
        default="datasets/midv2020",
        help="Path to the dataset directory (default: datasets/midv2020)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify the dataset, don't modify anything"
    )
    parser.add_argument(
        "--remap",
        action="store_true",
        help="Remap class IDs to VoidBox PII scheme and regenerate data.yaml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files (use with --remap)"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.dataset_dir)
    elif args.remap:
        data_config = load_data_yaml(args.dataset_dir)
        remap = analyze_classes(args.dataset_dir, data_config)
        
        if not remap:
            print("No remapping needed — classes already match target scheme.")
        else:
            print(f"\n{'Dry run — ' if args.dry_run else ''}Remapping labels...")
            remapped, dropped = remap_labels(args.dataset_dir, remap, dry_run=args.dry_run)
            print(f"\nTotal: {remapped} remapped, {dropped} dropped")
            
            if not args.dry_run:
                write_clean_yaml(args.dataset_dir)
        
        verify_dataset(args.dataset_dir)
    else:
        # Default: just verify
        verify_dataset(args.dataset_dir)

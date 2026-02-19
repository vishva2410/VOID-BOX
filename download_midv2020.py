#!/usr/bin/env python3
"""
download_midv2020.py — Download MIDV-2020 MRP dataset from Roboflow in YOLOv8 format.

Usage:
    python download_midv2020.py --api-key YOUR_ROBOFLOW_API_KEY

Or set the ROBOFLOW_API_KEY environment variable:
    export ROBOFLOW_API_KEY=YOUR_KEY
    python download_midv2020.py
"""

import os
import sys
import argparse
import shutil
from pathlib import Path


def get_api_key(args_key: str | None) -> str:
    """Get Roboflow API key from args or environment."""
    key = args_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("ERROR: No API key provided.")
        print("  Use --api-key YOUR_KEY or set ROBOFLOW_API_KEY env var.")
        print("  Get a free key at: https://roboflow.com → Settings → API Key")
        sys.exit(1)
    return key


def download_dataset(api_key: str, output_dir: str = "datasets/midv2020"):
    """Download the MIDV-2020 MRP dataset from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    print(f"Authenticating with Roboflow...")
    rf = Roboflow(api_key=api_key)

    # MIDV-2020 MRP Object Detection Dataset from Maastricht University
    # This dataset contains real identity documents with face, signature, text annotations
    print("Downloading MIDV-2020 MRP dataset...")
    project = rf.workspace("maastricht-university-6bosg").project("midv-2020-mrp")
    
    # Get the latest version
    version = project.version(2)
    
    # Download in YOLOv8 format
    dataset = version.download("yolov8", location=output_dir)
    
    print(f"\nDataset downloaded to: {os.path.abspath(output_dir)}")
    return dataset


def validate_dataset(dataset_dir: str):
    """Validate the downloaded dataset structure."""
    dataset_path = Path(dataset_dir)
    
    required_dirs = [
        dataset_path / "train" / "images",
        dataset_path / "train" / "labels",
        dataset_path / "valid" / "images",
        dataset_path / "valid" / "labels",
    ]
    
    # Check for data.yaml
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        print(f"WARNING: data.yaml not found at {yaml_path}")
        return False
    
    print(f"\n{'='*50}")
    print(f"Dataset Validation Report")
    print(f"{'='*50}")
    
    # Read and display data.yaml
    print(f"\ndata.yaml contents:")
    with open(yaml_path) as f:
        content = f.read()
        print(content)
    
    all_valid = True
    for d in required_dirs:
        if d.exists():
            file_count = len(list(d.glob("*")))
            print(f"  ✅ {d.relative_to(dataset_path)}: {file_count} files")
        else:
            print(f"  ❌ {d.relative_to(dataset_path)}: MISSING")
            all_valid = False
    
    # Count annotations per class
    print(f"\nAnnotation Statistics:")
    for split in ["train", "valid"]:
        label_dir = dataset_path / split / "labels"
        if not label_dir.exists():
            continue
        
        class_counts = {}
        total_annotations = 0
        for label_file in label_dir.glob("*.txt"):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                        total_annotations += 1
        
        print(f"\n  {split}:")
        print(f"    Total annotations: {total_annotations}")
        for cls_id in sorted(class_counts.keys()):
            print(f"    Class {cls_id}: {class_counts[cls_id]} annotations")
    
    print(f"\n{'='*50}")
    if all_valid:
        print("✅ Dataset validation PASSED")
    else:
        print("❌ Dataset validation FAILED")
    print(f"{'='*50}")
    
    return all_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MIDV-2020 MRP dataset from Roboflow for YOLOv8 training"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="datasets/midv2020",
        help="Output directory for dataset (default: datasets/midv2020)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing dataset, don't download"
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_dataset(args.output_dir)
    else:
        api_key = get_api_key(args.api_key)
        download_dataset(api_key, args.output_dir)
        validate_dataset(args.output_dir)

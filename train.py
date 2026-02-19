#!/usr/bin/env python3
"""
train.py — Fine-tune YOLOv8 on MIDV-2020 for real PII detection.

Trains on real identity document images with classes:
  0: document  — Full ID card / passport boundary
  1: face      — Face photo on document
  2: signature — Handwritten signature
  3: text_field — Name, DOB, ID number, MRZ, etc.

Usage:
    python train.py                                       # Full training (50 epochs)
    python train.py --epochs 2 --data datasets/midv2020/data.yaml  # Quick smoke test
    python train.py --resume                              # Resume interrupted training
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for VoidBox PII detection")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="Base model to fine-tune (default: yolov8s.pt)")
    parser.add_argument("--data", type=str, default="datasets/midv2020/data.yaml",
                        help="Path to data.yaml config")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 for auto, default: -1)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (auto-detected if not set)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"{'='*50}")
    print(f"  VoidBox — YOLOv8 PII Detection Training")
    print(f"{'='*50}")
    print(f"  Model:   {args.model}")
    print(f"  Data:    {args.data}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  ImgSize: {args.imgsz}")
    print(f"  Batch:   {'auto' if args.batch == -1 else args.batch}")
    print(f"  Device:  {device}")
    print(f"{'='*50}\n")

    # Validate data.yaml exists
    if not Path(args.data).exists():
        print(f"ERROR: Data config not found at {args.data}")
        print(f"Run download_midv2020.py first to get the dataset.")
        return

    # Load model
    if args.resume:
        # Resume from last checkpoint
        last_pt = Path("runs/detect/train/weights/last.pt")
        if last_pt.exists():
            print(f"Resuming from {last_pt}...")
            model = YOLO(str(last_pt))
        else:
            print(f"No checkpoint found at {last_pt}, starting fresh with {args.model}")
            model = YOLO(args.model)
    else:
        print(f"Loading base model: {args.model}")
        model = YOLO(args.model)

    # Train with optimized hyperparameters for document detection
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            
            # Project naming
            project="runs/voidbox",
            name="pii_detector",
            exist_ok=True,
            
            # Optimizer settings
            optimizer="AdamW",
            lr0=0.001,           # Initial learning rate
            lrf=0.01,            # Final LR factor (cosine decay)
            weight_decay=0.0005,
            warmup_epochs=3,
            
            # Early stopping
            patience=10,
            
            # Data augmentation — tuned for document images
            mosaic=1.0,          # Mosaic augmentation
            mixup=0.1,           # Light mixup
            hsv_h=0.015,         # Hue variation (documents have consistent colors)
            hsv_s=0.3,           # Saturation variation
            hsv_v=0.3,           # Value/brightness variation
            degrees=15.0,        # Rotation (documents can be tilted)
            translate=0.1,       # Translation
            scale=0.5,           # Scale variation
            fliplr=0.0,          # NO horizontal flip (text becomes unreadable)
            flipud=0.0,          # NO vertical flip
            
            # Performance
            workers=4,
            cache=True,          # Cache images in RAM for faster training
            
            # Logging
            verbose=True,
        )
        
        print(f"\n{'='*50}")
        print(f"  Training Complete!")
        print(f"{'='*50}")
        
        # Copy best model to project root
        best_pt = Path("runs/voidbox/pii_detector/weights/best.pt")
        if best_pt.exists():
            dest = Path("fine_tuned.pt")
            shutil.copy2(best_pt, dest)
            print(f"\n✅ Best model saved to: {dest.resolve()}")
            print(f"   This model will be auto-loaded by app.py")
        else:
            print(f"\n⚠️  best.pt not found at {best_pt}")
        
        # Print final metrics
        last_pt = Path("runs/voidbox/pii_detector/weights/last.pt")
        if last_pt.exists():
            print(f"   Last checkpoint: {last_pt.resolve()}")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print(f"\nTroubleshooting:")
        print(f"  • Check that the dataset exists: {args.data}")
        print(f"  • Try reducing batch size: --batch 8")
        print(f"  • Try CPU training: --device cpu")
        print(f"  • For MPS issues, try: --device cpu")
        raise


if __name__ == "__main__":
    main()

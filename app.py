#!/usr/bin/env python3
"""
VoidBox — PII Detection & Redaction System

Detects personally identifiable information in images using a YOLOv8 model
fine-tuned on MIDV-2020 (real identity documents), then redacts detected
regions using LaMa inpainting for seamless removal.

Detected PII classes:
  0: document   — Full ID card / passport boundary
  1: face       — Face photo on document
  2: signature  — Handwritten signature
  3: text_field — Name, DOB, ID number, MRZ, etc.
"""

import gradio as gr
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path
from simple_lama_inpainting import SimpleLama
from PIL import Image


# ─── PII Class Configuration ──────────────────────────────────────────────────

PII_CLASSES = {
    0: {"name": "document",   "color": (255, 100, 100), "conf": 0.40, "dilation": 10},
    1: {"name": "face",       "color": (100, 255, 100), "conf": 0.50, "dilation": 15},
    2: {"name": "signature",  "color": (100, 100, 255), "conf": 0.35, "dilation": 20},
    3: {"name": "text_field", "color": (255, 255, 100), "conf": 0.30, "dilation": 25},
}


# ─── Model Initialization ─────────────────────────────────────────────────────

print("="*50)
print("  VoidBox — PII Detection & Redaction")
print("="*50)

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"  Device: {device}")

# Load YOLO model — prioritize fine-tuned PII model
script_dir = Path(__file__).parent
fine_tuned_path = script_dir / "fine_tuned.pt"

try:
    if fine_tuned_path.exists():
        print(f"  Model:  fine_tuned.pt (MIDV-2020 PII detector)")
        model = YOLO(str(fine_tuned_path))
        using_pii_model = True
    else:
        print(f"  Model:  yolov8n.pt (generic — run train.py for PII model)")
        model = YOLO("yolov8n.pt")
        using_pii_model = False

    # Load LaMa inpainting model
    simple_lama = SimpleLama()
    print(f"  LaMa:   ✅ loaded")
    print("="*50)

except Exception as e:
    print(f"  ❌ Error loading models: {e}")
    exit(1)


# ─── Core Redaction Pipeline ──────────────────────────────────────────────────

def redact_pii(input_image, redact_documents=True, redact_faces=True,
               redact_signatures=True, redact_text=True, show_detections=False):
    """
    Detect and redact PII from an image.
    
    Args:
        input_image: PIL Image from Gradio
        redact_documents: Redact full document boundaries
        redact_faces: Redact face photos
        redact_signatures: Redact signatures
        redact_text: Redact text fields
        show_detections: Return annotated image instead of redacted
    
    Returns:
        Redacted PIL Image + detection summary text
    """
    if input_image is None:
        return None, "No image provided."

    # Convert PIL → OpenCV
    image_np = np.array(input_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Build list of classes to redact
    active_classes = {}
    if redact_documents:
        active_classes[0] = PII_CLASSES[0]
    if redact_faces:
        active_classes[1] = PII_CLASSES[1]
    if redact_signatures:
        active_classes[2] = PII_CLASSES[2]
    if redact_text:
        active_classes[3] = PII_CLASSES[3]

    # Run YOLOv8 inference
    results = model(image_bgr, device=device, verbose=False)
    result = results[0]

    # Filter detections by class and confidence
    h, w, _ = image_bgr.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    detection_counts = {cls_info["name"]: 0 for cls_info in PII_CLASSES.values()}
    annotated_img = image_bgr.copy() if show_detections else None

    detections_found = False
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Check if this class is active
        if cls_id not in active_classes:
            continue

        cls_info = active_classes[cls_id]

        # Apply class-specific confidence threshold
        if conf < cls_info["conf"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections_found = True
        detection_counts[cls_info["name"]] += 1

        # Apply class-specific dilation
        dilation = cls_info["dilation"]
        kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8)

        # Create per-box mask and dilate
        box_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
        box_mask = cv2.dilate(box_mask, kernel, iterations=1)
        mask = cv2.bitwise_or(mask, box_mask)

        # Annotate for preview mode
        if show_detections:
            color = cls_info["color"]
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_info['name']} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated_img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Build summary
    total = sum(detection_counts.values())
    summary_parts = [f"**{total} PII item(s) detected**"]
    for name, count in detection_counts.items():
        if count > 0:
            summary_parts.append(f"  • {name}: {count}")
    
    if not detections_found:
        summary = "✅ No PII detected in this image."
        return input_image, summary

    summary = "\n".join(summary_parts)

    # Preview mode — return annotated image
    if show_detections:
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), summary

    # Redaction mode — inpaint masked regions
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask)

    result_image = simple_lama(image_pil, mask_pil)

    summary += "\n\n🔒 All detected PII has been redacted."
    return result_image, summary


# ─── Gradio Interface ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model_status = "🟢 Fine-tuned PII model (MIDV-2020)" if using_pii_model else "🟡 Generic model — run train.py for real PII detection"

    demo = gr.Interface(
        fn=redact_pii,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Checkbox(value=True, label="🪪 Redact Documents"),
            gr.Checkbox(value=True, label="👤 Redact Faces"),
            gr.Checkbox(value=True, label="✍️ Redact Signatures"),
            gr.Checkbox(value=True, label="📝 Redact Text Fields"),
            gr.Checkbox(value=False, label="👁️ Preview Mode (show detections)"),
        ],
        outputs=[
            gr.Image(type="pil", label="Result"),
            gr.Markdown(label="Detection Summary"),
        ],
        title="🔒 VoidBox — PII Detection & Redaction",
        description=(
            f"Upload an image containing identity documents to automatically detect "
            f"and redact personally identifiable information.\n\n"
            f"**Model:** {model_status}\n\n"
            f"**Detected PII types:** Documents, Faces, Signatures, Text Fields"
        ),
        examples=[["test.jpg", True, True, True, True, False]] if Path("test.jpg").exists() else None,
        flagging_mode="never",
    )

    print("\nLaunching VoidBox...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

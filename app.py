#!/usr/bin/env python3
"""
VoidBox — PII Detection & Redaction System (Stage 9: Robust Mask Engineering)

Multi-modal, context-aware privacy pipeline:
  1. YOLOv8 object detection  (documents, faces, signatures, text_fields)
  1.5 OpenCV Haar cascade     (license plate detection)
  2. EasyOCR text detection
  3. Regex-based PII filter    — only sensitive text is masked
  4. Smart masking:
       - Overlapping box merging
       - Proportional adaptive expansion
       - Rounded rectangle masks (no sharp corners)
       - Internal gap filling for OCR clusters
       - Per-region inpainting (each region processed separately)
       - Minimum area noise filter
  5. Hybrid inpainting per region: Telea pre-fill → LaMa × 2
"""

import re
import base64
import io
import json
from flask import Flask, request, jsonify, send_from_directory, send_file
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import easyocr
from pathlib import Path
from simple_lama_inpainting import SimpleLama
from PIL import Image


# ─── PII Class Configuration ──────────────────────────────────────────────────

PII_CLASSES = {
    0: {"name": "document",   "color": (255, 100, 100), "conf": 0.45},
    1: {"name": "face",       "color": (100, 255, 100), "conf": 0.55},
    2: {"name": "signature",  "color": (100, 100, 255), "conf": 0.40},
    3: {"name": "text_field", "color": (255, 255, 100), "conf": 0.35},
}

# Warn when masked area exceeds this fraction of total image area
LARGE_REGION_WARN_FRAC = 0.08  # 8% — expect quality drop above this

# Minimum bounding box area in pixels — boxes smaller than this are noise
MIN_BOX_AREA = 100

# License plate Haar cascade — ships with OpenCV, no extra download
_PLATE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
_plate_cascade = cv2.CascadeClassifier(_PLATE_CASCADE_PATH)
PLATE_COLOR = (0, 200, 200)  # cyan for annotation


# ─── PII Pattern Definitions ──────────────────────────────────────────────────

_PII_PATTERNS = [
    (r"\b\d{12}\b",                                          "12-digit (Aadhaar-style)"),
    (r"\b\d{16}\b",                                          "16-digit (card number)"),
    (r"\b\d{10}\b",                                          "10-digit (phone/ID)"),
    (r"\b\d{8,9}\b",                                         "8-9 digit ID"),
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",    "Email"),
    (r"\b[A-Z][0-9]{7}\b",                                   "Passport code"),
    (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",                          "PAN-style ID"),
    (r"\b[A-Z]{2}[0-9]{6,8}\b",                             "Alphanumeric ID"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                        "IPv4 address"),
    (r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",         "Date"),
    (r"[A-Z0-9<]{8,}",                                       "MRZ / machine-readable code"),
    (r"\b\d{3}-\d{2}-\d{4}\b",                               "US SSN"),
    # License plate patterns
    (r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}\b",      "License plate (IN)"),
    (r"\b[A-Z]{2}\s?\d{2}\s?[A-Z]{2}\s?\d{4}\b",           "License plate (IN alt)"),
    (r"\b[0-9]{1}[A-Z]{3}\s?[0-9]{3}\b",                   "License plate (US-style)"),
    (r"\b[A-Z]{2,3}\s?\d{3,4}\s?[A-Z]{0,3}\b",             "License plate (EU-style)"),
    (r"\b[A-Z]{2}[0-9]{2}\s?[A-Z]{3}\b",                   "License plate (UK)"),
]

_COMPILED = [(re.compile(p), label) for p, label in _PII_PATTERNS]


def classify_text(text: str) -> tuple[bool, str]:
    """Return (is_sensitive, matched_pattern_label)."""
    for pattern, label in _COMPILED:
        if pattern.search(text):
            return True, label
    return False, ""


# ─── Mask Utility Functions ───────────────────────────────────────────────────

def _merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """
    Merge overlapping bounding boxes (x1, y1, x2, y2 format).
    Iteratively unions any pair that overlaps until stable.
    """
    if not boxes:
        return []

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            ax1, ay1, ax2, ay2 = merged[i]
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                # Check overlap
                if ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1:
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used.add(j)
                    changed = True
            new_merged.append((ax1, ay1, ax2, ay2))
            used.add(i)
        merged = new_merged
    return merged


def _proportional_expand(x1: int, y1: int, x2: int, y2: int,
                         img_h: int, img_w: int,
                         base_frac: float = 0.18,
                         min_px: int = 12, max_px: int = 50
                         ) -> tuple[int, int, int, int]:
    """
    Expand a box proportionally based on its size.
    Small boxes → relatively larger padding; large boxes → smaller relative padding.
    Returns clamped (x1, y1, x2, y2).
    """
    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * base_frac)
    pad = max(min_px, min(pad, max_px))

    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(img_w - 1, x2 + pad),
        min(img_h - 1, y2 + pad),
    )


def _draw_rounded_rect(mask: np.ndarray,
                        x1: int, y1: int, x2: int, y2: int,
                        radius: int = 0) -> None:
    """
    Draw a filled rounded rectangle onto mask (in-place).
    If radius is 0 it auto-calculates from box size (~12% of short side).
    """
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return
    if radius <= 0:
        radius = max(4, int(min(bw, bh) * 0.12))
    radius = min(radius, bw // 2, bh // 2)  # clamp so circles fit

    # Inner cross (two overlapping rectangles that cover everything except corners)
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)

    # Four corner circles
    cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)


def _box_area(x1, y1, x2, y2) -> int:
    """Pixel area of a bounding box."""
    return max(0, x2 - x1) * max(0, y2 - y1)


def _boxes_overlap(a, b) -> bool:
    """Check if two (x1,y1,x2,y2) boxes overlap."""
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


# ─── Inpainting Engine ───────────────────────────────────────────────────────

def _make_binary(mask: np.ndarray) -> np.ndarray:
    """Enforce strictly binary mask — 0 or 255, nothing in between."""
    return np.where(mask > 0, np.uint8(255), np.uint8(0))


def _ensure_mask_size(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize mask to exactly (h, w) if dimensions don't match."""
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return _make_binary(mask)


def _adaptive_dilate(mask: np.ndarray, region_area: int, img_area: int) -> np.ndarray:
    """
    Dilate mask adaptively based on region size.
    Small regions → more dilation (covers edge artifacts).
    Large regions → less dilation (avoids eating too much context).
    """
    frac = region_area / max(img_area, 1)

    if frac < 0.005:
        # Very small region — dilate generously to cover fringe text
        k = 11
        iters = 2
    elif frac < 0.03:
        # Medium region — moderate dilation
        k = 9
        iters = 1
    else:
        # Large region — thin dilation only for edge smoothing
        k = 7
        iters = 1

    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iters)
    return _make_binary(dilated)


def _soft_edge_mask(mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    """
    Apply Gaussian blur to mask edges, then re-threshold.
    Larger kernel → smoother transition → fewer visible seams.
    """
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=2.0)
    return np.where(blurred > 100, np.uint8(255), np.uint8(0))


def _feather_alpha(mask: np.ndarray, ksize: int = 9) -> np.ndarray:
    """
    Create a soft alpha matte from a binary mask for seamless blending.
    Ensures mask interior stays fully opaque, with a feathered edge.
    """
    mask = _make_binary(mask)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=0)
    alpha = blurred.astype(np.float32) / 255.0
    alpha[mask == 255] = 1.0
    return np.clip(alpha, 0.0, 1.0)


def _blend_alpha(mask: np.ndarray, region_area: int, img_area: int) -> np.ndarray:
    """
    Build a size-adaptive alpha matte for compositing inpainted regions.
    Slightly dilates the original mask, then feathers the edge.
    """
    frac = region_area / max(img_area, 1)
    if frac < 0.005:
        k = 11
    elif frac < 0.03:
        k = 9
    else:
        k = 7

    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return _feather_alpha(dilated, ksize=k + 2)


def _safe_cv2_inpaint(image_bgr: np.ndarray, mask: np.ndarray,
                       radius: int = 7) -> np.ndarray:
    """
    OpenCV Telea inpaint with guaranteed size matching.
    Returns BGR result.
    """
    h, w = image_bgr.shape[:2]
    mask = _ensure_mask_size(mask, h, w)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = _make_binary(mask)
    return cv2.inpaint(image_bgr, mask, radius, cv2.INPAINT_TELEA)


def _safe_lama(image_rgb: np.ndarray, mask: np.ndarray,
               lama_model) -> np.ndarray:
    """
    Run LaMa inpainting with size-safe guards.
    Resizes output back to original dimensions if LaMa changes them.
    Falls back to OpenCV Telea if LaMa fails.
    """
    h, w = image_rgb.shape[:2]
    mask = _ensure_mask_size(mask, h, w)
    mask_pil = Image.fromarray(mask, mode="L")

    try:
        result_pil = lama_model(Image.fromarray(image_rgb), mask_pil)
        result = np.array(result_pil)

        if result.shape[0] != h or result.shape[1] != w:
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        return result

    except Exception as e:
        print(f"    ⚠  LaMa failed ({e}), falling back to OpenCV Telea")
        telea = _safe_cv2_inpaint(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), mask
        )
        return cv2.cvtColor(telea, cv2.COLOR_BGR2RGB)


def _context_border_mask(mask: np.ndarray, border_px: int = 20) -> np.ndarray:
    """
    Create a narrow band around the mask border for color sampling.
    This is the "context ring" — the surrounding pixels we match colors to.
    """
    dilated = cv2.dilate(mask, np.ones((border_px, border_px), np.uint8), iterations=1)
    border = cv2.subtract(dilated, mask)
    return _make_binary(border)


def _color_harmonize(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Match mean/std of color inside the mask to the surrounding context ring.
    This eliminates color shifts from inpainting — makes the fill blend in
    with the local neighborhood instead of the global image statistics.
    """
    h, w = image_rgb.shape[:2]
    mask = _ensure_mask_size(mask, h, w)
    border = _context_border_mask(mask, border_px=24)

    mask_bool = mask > 127
    border_bool = border > 127

    if not np.any(mask_bool) or not np.any(border_bool):
        return image_rgb

    result = image_rgb.copy().astype(np.float32)

    for c in range(3):
        channel = result[:, :, c]

        # Stats of the inpainted region
        inner_mean = channel[mask_bool].mean()
        inner_std = max(channel[mask_bool].std(), 1e-6)

        # Stats of surrounding context
        ctx_mean = channel[border_bool].mean()
        ctx_std = max(channel[border_bool].std(), 1e-6)

        # Normalize inner to match context distribution
        channel[mask_bool] = ((channel[mask_bool] - inner_mean) / inner_std) * ctx_std + ctx_mean
        result[:, :, c] = channel

    return np.clip(result, 0, 255).astype(np.uint8)


def _poisson_blend(base_rgb: np.ndarray, inpainted_rgb: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
    """
    Seamless Poisson clone of the inpainted region onto the base image.
    This eliminates hard seams at mask borders by gradient-domain blending.
    Falls back to simple overlay if cloning fails (e.g., region too close to edge).
    """
    h, w = base_rgb.shape[:2]
    mask = _ensure_mask_size(mask, h, w)

    if inpainted_rgb.shape[0] != h or inpainted_rgb.shape[1] != w:
        inpainted_rgb = cv2.resize(inpainted_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # Poisson needs the center of the mask region
    ys, xs = np.where(mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        return base_rgb

    cx = int(xs.mean())
    cy = int(ys.mean())

    # Clamp center so the paste doesn't go out of bounds
    cx = max(1, min(cx, w - 2))
    cy = max(1, min(cy, h - 2))

    try:
        blended = cv2.seamlessClone(
            cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR),
            mask,
            (cx, cy),
            cv2.NORMAL_CLONE,
        )
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    except Exception:
        # Fallback: simple mask-based composite
        mask_3ch = np.stack([mask] * 3, axis=-1) > 127
        result = base_rgb.copy()
        result[mask_3ch] = inpainted_rgb[mask_3ch]
        return result


def _inpaint_single_region(image_rgb: np.ndarray,
                            region_mask: np.ndarray,
                            lama_model,
                            region_area: int = 0) -> np.ndarray:
    """
    Production-grade inpainting for one connected region.

    Pipeline:
      1. Adaptive dilation — expand mask edges proportionally
      2. Soft-edge smoothing — remove jagged mask borders
      3. Telea pre-fill — recover rough structure and gradients
      4. LaMa × N passes — neural texture synthesis (adaptive count)
      5. Poisson seamless blend — gradient-domain border fusion
      6. Color harmonization — match local color context

    All operations are size-safe.
    """
    h, w = image_rgb.shape[:2]
    img_area = h * w
    region_area = region_area or cv2.countNonZero(region_mask)
    orig_mask = _ensure_mask_size(region_mask, h, w)
    mask_bin = orig_mask.copy()

    # Step 1: Adaptive dilation — generous for small text, light for big regions
    mask_bin = _adaptive_dilate(mask_bin, region_area, img_area)

    # Step 2: Soft-edge smoothing — wider kernel for large regions
    ksize = 9 if region_area > img_area * 0.02 else 7
    mask_bin = _soft_edge_mask(mask_bin, ksize=ksize)
    mask_bin = _make_binary(mask_bin)

    # Step 3: Context-aware Telea radius
    frac = region_area / max(img_area, 1)
    telea_radius = 5 if frac < 0.01 else (12 if frac > 0.05 else 7)

    telea_bgr = _safe_cv2_inpaint(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), mask_bin,
        radius=telea_radius,
    )
    result = cv2.cvtColor(telea_bgr, cv2.COLOR_BGR2RGB)

    # Step 4: Adaptive LaMa passes — more passes for larger/harder regions
    num_lama = 3 if frac > 0.03 else 2
    for pass_n in range(num_lama):
        result = _safe_lama(result, mask_bin, lama_model)

    # Step 5: Poisson seamless blend — removes hard seams at borders
    result = _poisson_blend(image_rgb, result, mask_bin)

    # Step 6: Color harmonization — match surrounding context
    result = _color_harmonize(result, mask_bin)

    # Step 7: Feathered composite back onto the base image to avoid over-inpainting
    alpha = _blend_alpha(orig_mask, region_area, img_area)
    alpha_3 = alpha[..., None]
    result = (result * alpha_3 + image_rgb * (1.0 - alpha_3)).astype(np.uint8)

    return result


def per_region_inpaint(image_rgb: np.ndarray,
                       mask: np.ndarray,
                       lama_model) -> Image.Image:
    """
    Per-region inpainting: split mask into connected components,
    inpaint each separately, composite results.

    Pipeline per region:
      dilate → soft-edge → Telea → LaMa × N → Poisson blend → color harmonize
    """
    h, w = image_rgb.shape[:2]

    # Pre-process the full mask
    mask = _soft_edge_mask(mask)
    mask = _make_binary(mask)

    if mask.max() == 0:
        return Image.fromarray(image_rgb)

    # Split into connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    result = image_rgb.copy()
    regions_processed = 0

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BOX_AREA:
            continue

        region_mask = np.where(labels == i, np.uint8(255), np.uint8(0))

        region_frac = area / (h * w)
        if region_frac > LARGE_REGION_WARN_FRAC:
            print(f"    ⚠  Region {i}: {region_frac:.1%} of image — quality may degrade")

        result = _inpaint_single_region(result, region_mask, lama_model,
                                         region_area=area)
        regions_processed += 1

    print(f"    Inpainted {regions_processed} region(s)")
    return Image.fromarray(result)


# ─── Model Initialization ─────────────────────────────────────────────────────

print("=" * 60)
print("  VoidBox — Stage 9: Robust Mask Engineering")
print("=" * 60)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"  Device:  {device}")

script_dir = Path(__file__).parent
models_dir = script_dir / "models"
fine_tuned = models_dir / "fine_tuned.pt"
base_model = models_dir / "yolov8n.pt"

try:
    if fine_tuned.exists():
        print("  YOLO:    fine_tuned.pt (MIDV-2020 PII detector)")
        yolo_model = YOLO(str(fine_tuned))
        using_pii_model = True
    else:
        print("  YOLO:    yolov8n.pt (generic -- run train.py for PII model)")
        yolo_model = YOLO(str(base_model)) if base_model.exists() else YOLO("yolov8n.pt")
        using_pii_model = False

    simple_lama = SimpleLama()
    print("  LaMa:    loaded")

    print("  EasyOCR: loading...")
    ocr_reader = easyocr.Reader(['en'], gpu=(device == "cuda"), verbose=False)
    print("  EasyOCR: loaded")

    if _plate_cascade.empty():
        print("  Plates:  WARNING -- Haar cascade not found, plate detection disabled")
    else:
        print("  Plates:  Haar cascade loaded")
    print("=" * 60)

except Exception as e:
    print(f"  ERROR loading models: {e}")
    exit(1)


# ─── Core Redaction Pipeline ──────────────────────────────────────────────────

def redact_pii(input_image,
               redact_documents, redact_faces,
               redact_signatures, redact_text_fields,
               redact_plates,
               ocr_mode,
               ocr_confidence,
               show_detections):
    """
    Full PII redaction pipeline with robust mask engineering.

    Stages:
      1. YOLO + Haar + OCR detection
      2. Smart masking (merge, expand, round, fill gaps, min area filter)
      3. Per-region hybrid inpainting (Telea → LaMa × 2)
    """
    if input_image is None:
        return None, None, "No image provided."

    # Work at full resolution — no downscaling
    image_np = np.array(input_image)
    if image_np.ndim == 2:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image_np.copy()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    active_classes = {}
    if redact_documents:   active_classes[0] = PII_CLASSES[0]
    if redact_faces:       active_classes[1] = PII_CLASSES[1]
    if redact_signatures:  active_classes[2] = PII_CLASSES[2]
    if redact_text_fields: active_classes[3] = PII_CLASSES[3]

    counts = {cls["name"]: 0 for cls in PII_CLASSES.values()}
    counts.update({"plate": 0, "ocr_pii": 0, "ocr_safe": 0})
    ocr_log_rows = []

    annotated = image_bgr.copy() if show_detections else None

    # Track all object regions for OCR gap-filling
    all_object_boxes = []

    # ── Phase 1: YOLO Object Detection ────────────────────────────────────
    yolo_results = yolo_model(image_bgr, device=device, verbose=False)[0]
    yolo_boxes = []

    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id not in active_classes:
            continue
        cls_info = active_classes[cls_id]
        if conf < cls_info["conf"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Min area filter — skip noise
        if _box_area(x1, y1, x2, y2) < MIN_BOX_AREA:
            continue

        yolo_boxes.append((x1, y1, x2, y2))
        counts[cls_info["name"]] += 1

        if show_detections:
            color = cls_info["color"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = f"{cls_info['name']} {conf:.2f}"
            (tw, th_t), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th_t - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, lbl, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Merge overlapping YOLO boxes → proportional expand → rounded rect mask
    merged_yolo = _merge_boxes(yolo_boxes)
    for bx1, by1, bx2, by2 in merged_yolo:
        ex1, ey1, ex2, ey2 = _proportional_expand(bx1, by1, bx2, by2, h, w)
        _draw_rounded_rect(mask, ex1, ey1, ex2, ey2)
        all_object_boxes.append((ex1, ey1, ex2, ey2))

    # ── Phase 1.5: License Plate Detection (Haar Cascade) ─────────────────
    plate_boxes = []
    if redact_plates and not _plate_cascade.empty():
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        plates = _plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20),
        )
        for (px, py, pw, ph) in plates:
            x1, y1, x2, y2 = px, py, px + pw, py + ph

            if _box_area(x1, y1, x2, y2) < MIN_BOX_AREA:
                continue

            plate_boxes.append((x1, y1, x2, y2))
            counts["plate"] += 1

            if show_detections:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), PLATE_COLOR, 2)
                lbl = "plate"
                (tw, th_t), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th_t - 8), (x1 + tw + 4, y1), PLATE_COLOR, -1)
                cv2.putText(annotated, lbl, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Merge plate boxes → expand → rounded rect mask
    merged_plates = _merge_boxes(plate_boxes)
    for bx1, by1, bx2, by2 in merged_plates:
        ex1, ey1, ex2, ey2 = _proportional_expand(bx1, by1, bx2, by2, h, w)
        _draw_rounded_rect(mask, ex1, ey1, ex2, ey2)
        all_object_boxes.append((ex1, ey1, ex2, ey2))

    # ── Phase 2: EasyOCR Text Detection ───────────────────────────────────
    if ocr_mode != "off":
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray_image)
        
        ocr_results = ocr_reader.readtext(clahe_image, mag_ratio=1.5, paragraph=False)
        ocr_pii_boxes = []

        for (bbox, text, prob) in ocr_results:
            if prob < ocr_confidence:
                continue

            sensitive, pattern_label = classify_text(text)
            should_erase = sensitive if ocr_mode == "smart" else True

            ocr_log_rows.append(
                (text, pattern_label or "--", "ERASED" if should_erase else "KEPT")
            )

            if not should_erase:
                counts["ocr_safe"] += 1
                continue

            counts["ocr_pii"] += 1
            pts = np.array(bbox, dtype=np.int32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if _box_area(x1, y1, x2, y2) < MIN_BOX_AREA:
                continue

            ocr_pii_boxes.append((x1, y1, x2, y2))

            if show_detections:
                cv2.polylines(annotated, [pts], True, (0, 140, 255), 2)
                cv2.putText(annotated, f"[PII] {text[:22]}",
                            (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1)

        # Fill internal gaps: if multiple OCR boxes fall inside a single
        # YOLO/plate region, replace individual word boxes with the full
        # object region — prevents fragmented holes on plates/IDs.
        filled_boxes = []
        used_ocr = set()
        for obj_box in all_object_boxes:
            group = []
            for idx, ocr_box in enumerate(ocr_pii_boxes):
                if idx in used_ocr:
                    continue
                if _boxes_overlap(obj_box, ocr_box):
                    group.append(idx)
            if len(group) >= 2:
                filled_boxes.append(obj_box)
                used_ocr.update(group)

        # Keep ungrouped OCR boxes
        for idx, ocr_box in enumerate(ocr_pii_boxes):
            if idx not in used_ocr:
                filled_boxes.append(ocr_box)

        # Merge → expand → rounded rect mask
        merged_ocr = _merge_boxes(filled_boxes)
        for bx1, by1, bx2, by2 in merged_ocr:
            ex1, ey1, ex2, ey2 = _proportional_expand(bx1, by1, bx2, by2, h, w)
            _draw_rounded_rect(mask, ex1, ey1, ex2, ey2)

    # ── Mask Statistics Logging ────────────────────────────────────────────
    mask_binary = _make_binary(mask)
    total_masked = cv2.countNonZero(mask_binary)
    mask_pct = total_masked / (h * w) * 100

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    num_regions = num_labels - 1

    print(f"\n  ── Mask Statistics ──")
    print(f"    Regions:     {num_regions}")
    print(f"    Masked area: {mask_pct:.2f}% ({total_masked:,} px)")
    if num_regions > 0:
        region_sizes = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        print(f"    Per-region:  min={min(region_sizes):,}  max={max(region_sizes):,}  "
              f"avg={sum(region_sizes) // num_regions:,}")

    # ── Build OCR preview table ───────────────────────────────────────────
    if ocr_log_rows:
        table_lines = ["| Detected Text | Pattern | Action |", "|---|---|---|"]
        for text, pat, action in ocr_log_rows:
            safe_text = text.replace("|", "\\|")
            table_lines.append(f"| `{safe_text}` | {pat} | {action} |")
        ocr_table_md = "\n".join(table_lines)
    else:
        ocr_table_md = "_No text detected by OCR._"

    # ── Build summary ─────────────────────────────────────────────────────
    total_erased = (sum(counts[k] for k in ["document", "face", "signature", "text_field"])
                    + counts["plate"] + counts["ocr_pii"])

    if total_erased == 0 and mask_binary.max() == 0:
        summary_parts = ["No PII detected -- image is clean."]
    else:
        summary_parts = [f"**{total_erased} PII region(s) targeted** → "
                         f"{num_regions} mask region(s), {mask_pct:.1f}% masked"]
        for name in ["document", "face", "signature", "text_field"]:
            if counts[name]:
                summary_parts.append(f"  - {name}: {counts[name]}")
        if counts["plate"]:
            summary_parts.append(f"  - license plate: {counts['plate']}")
        if counts["ocr_pii"]:
            summary_parts.append(f"  - OCR PII text: {counts['ocr_pii']}")
        if counts["ocr_safe"]:
            summary_parts.append(f"  - OCR safe text (kept): {counts['ocr_safe']}")

    summary = "\n".join(summary_parts)

    # ── Preview mode ──────────────────────────────────────────────────────
    if show_detections:
        out = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        return out, ocr_table_md, summary

    if mask_binary.max() == 0:
        return input_image, ocr_table_md, summary

    # ── Phase 3: Per-Region Hybrid Inpainting ─────────────────────────────
    result_image = per_region_inpaint(image_rgb, mask, simple_lama)
    summary += "\n\nAll flagged PII has been AI-erased (per-region hybrid inpainting)."
    return result_image, ocr_table_md, summary


# ─── Flask Application ────────────────────────────────────────────────────────

flask_app = Flask(__name__, static_folder="frontend", static_url_path="")

FRONTEND_DIR = Path(__file__).parent / "frontend"


@flask_app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@flask_app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@flask_app.route("/api/health", methods=["GET"])
def api_health():
    model_label = "Fine-tuned PII model (MIDV-2020)" if using_pii_model else "Generic YOLOv8n"
    return jsonify({
        "status": "ok",
        "device": device,
        "model": model_label,
        "fine_tuned": using_pii_model,
        "plate_cascade": not _plate_cascade.empty(),
        "stage": "Stage 9: Robust Mask Engineering",
    })


@flask_app.route("/api/redact", methods=["POST"])
def api_redact():
    """
    Accepts JSON:
      {
        "image": "<base64-encoded PNG/JPEG>",
        "redact_documents": bool,
        "redact_faces": bool,
        "redact_signatures": bool,
        "redact_text_fields": bool,
        "redact_plates": bool,
        "ocr_mode": "smart"|"all"|"off",
        "ocr_confidence": float,
        "show_detections": bool
      }
    Returns JSON:
      {
        "result_image": "<base64-encoded PNG>",
        "ocr_table": "<markdown string>",
        "summary": "<markdown string>"
      }
    """
    try:
        data = request.get_json(force=True)

        # Decode image
        img_b64 = data.get("image", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(img_b64)
        pil_input = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Run pipeline
        result_pil, ocr_table, summary = redact_pii(
            input_image       = pil_input,
            redact_documents  = bool(data.get("redact_documents", True)),
            redact_faces      = bool(data.get("redact_faces", True)),
            redact_signatures = bool(data.get("redact_signatures", True)),
            redact_text_fields= bool(data.get("redact_text_fields", True)),
            redact_plates     = bool(data.get("redact_plates", True)),
            ocr_mode          = data.get("ocr_mode", "smart"),
            ocr_confidence    = float(data.get("ocr_confidence", 0.4)),
            show_detections   = bool(data.get("show_detections", False)),
        )

        # Encode result
        buf = io.BytesIO()
        if isinstance(result_pil, Image.Image):
            result_pil.save(buf, format="PNG")
        else:
            np.array(result_pil)
            Image.fromarray(np.array(result_pil)).save(buf, format="PNG")
        result_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            "result_image": result_b64,
            "ocr_table": ocr_table,
            "summary": summary,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@flask_app.route("/api/download-model", methods=["GET"])
def api_download_model():
    """Stream the fine-tuned model file if it exists."""
    model_path = Path(__file__).parent / "models" / "fine_tuned.pt"
    if not model_path.exists():
        return jsonify({"error": "Fine-tuned model not found. Run train.py first."}), 404
    return send_file(
        str(model_path),
        as_attachment=True,
        download_name="voidbox_fine_tuned.pt",
        mimetype="application/octet-stream",
    )


if __name__ == "__main__":
    model_tag = "Fine-tuned PII model" if using_pii_model else "Generic YOLOv8n"
    print(f"\n  Model: {model_tag}")
    print(f"  Frontend: http://localhost:7860")
    print(f"  API:      http://localhost:7860/api/health\n")
    flask_app.run(host="0.0.0.0", port=7860, debug=False)

#!/usr/bin/env python3
"""
================================================================================
VoidBox — Multi-Modal PII Detection & Redaction Engine
================================================================================
A privacy-first image anonymization engine that runs 100% locally.
It integrates YOLOv8, EasyOCR, Haar Cascades, and LaMa Neural Inpainting
to detect, filter, and seamlessly redact Personally Identifiable Information (PII).

Architecture Pipeline:
  1. Image Preprocessing: Normalization and type conversion.
  2. Multi-Modal Detection:
     - YOLOv8 (fine-tuned) detects document regions, faces, signatures, and generic text fields.
     - Haar Cascades fall back to detect license plates and missed faces.
     - EasyOCR (optimized with CLAHE) extracts all raw text strings.
  3. Classification: Regex engine checks OCR text against PII patterns (Aadhaar, SSN, PAN, Card Numbers, etc.).
  4. Mask Geometry Engine: Merges overlapping boxes, expands padding proportionally, and generates rounded corner masks.
  5. Privacy & Inpainting Engine: Crops image to each disconnected mask component, applies OpenCV Telea pre-fill,
     runs LaMa Neural Inpainting, harmonizes color domains, and clones seamlessly via Poisson Blending.
================================================================================
"""

# Core Python Utilities
import re
import time
import base64
import io
import json
import uuid
import threading
from pathlib import Path

# Web Framework
from flask import Flask, request, jsonify, send_from_directory, send_file

# AI & Computer Vision Models
from ultralytics import YOLO                   # YOLOv8 object detector
import cv2                                     # OpenCV image processing
import torch                                   # Deep learning backend (CUDA/MPS/CPU support)
import numpy as np                             # Numerical array manipulations
import easyocr                                 # Optical Character Recognition (OCR) engine
from simple_lama_inpainting import SimpleLama  # Resolution-independent neural inpainting model
from PIL import Image                          # Image format wrapper for models



# ==============================================================================
# PII DETECTION CLASS CONFIGURATION
# ==============================================================================
# YOLOv8 class mappings with customized parameters for targeted detection.
# - 'conf': Minimum confidence threshold to accept the YOLO boundary prediction.
#   Signatures use lower conf (0.30) to catch faint ink. Faces use higher (0.55) to prevent false positives.
# - 'min_area': Area threshold (width * height in px). Detections below this are ignored to filter sensor noise.
PII_CLASSES = {
    0: {"name": "document",   "color": (255, 100, 100), "conf": 0.45, "min_area": 500},
    1: {"name": "face",       "color": (100, 255, 100), "conf": 0.55, "min_area": 120},
    2: {"name": "signature",  "color": (100, 100, 255), "conf": 0.30, "min_area": 60},
    3: {"name": "text_field", "color": (255, 255, 100), "conf": 0.35, "min_area": 80},
}

# Warning threshold: If any mask region exceeds 8% of the total image area,
# we log a warning since massive masks are harder for LaMa to restore naturally.
LARGE_REGION_WARN_FRAC = 0.08

# Global noise threshold for bounding boxes: Ignore boxes smaller than 100 pixels.
MIN_BOX_AREA = 100

# ==============================================================================
# IN-MEMORY RESULT CACHING SYSTEM
# ==============================================================================
# Caches processing outputs for 30 minutes. This prevents the frontend from
# having to pass huge base64 images inside HTTP session storages.
RESULT_TTL_SEC = 30 * 60  # Time to live: 30 minutes
RESULT_MAX_ITEMS = 20     # Max cached results to prevent memory leaks
_RESULT_LOCK = threading.Lock()
_RESULT_STORE: dict[str, dict] = {}


# ==============================================================================
# OPENCV HAAR CASCADE FALLBACK DETECTORS
# ==============================================================================
# Haar Cascades provide lightweight, rapid heuristics for detecting license plates
# and faces. These operate on local XML files shipped with OpenCV.
_PLATE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
_plate_cascade = cv2.CascadeClassifier(_PLATE_CASCADE_PATH)
PLATE_COLOR = (0, 200, 200)  # Cyan bounding boxes

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
FACE_COLOR = (0, 180, 255)   # Orange bounding boxes


# ==============================================================================
# REGEX PII PATTERNS & SEMANTIC CLASSIFICATION
# ==============================================================================
# A set of regular expressions for matching sensitive textual information.
# Contains patterns for Indian (Aadhaar, PAN), American (SSN), Credit Cards,
# Emails, dates, and international license plates.
_PII_PATTERNS = [
    (r"\b\d{12}\b",                                          "12-digit (Aadhaar-style)"),
    (r"\b(?:\d{4}[-\s]?){2}\d{4}\b",                         "12-digit (with separators)"),
    (r"\b\d{16}\b",                                          "16-digit (card number)"),
    (r"\b(?:\d{4}[-\s]?){3}\d{4}\b",                         "16-digit (with separators)"),
    (r"\b\d{10}\b",                                          "10-digit (phone/ID)"),
    (r"\b(?:\+?\d{1,3}[-\s]?)?(?:\d{3}[-\s]?){2}\d{4}\b",     "Phone (with separators)"),
    (r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b",
     "Phone (US format)"),
    (r"\b\d{8,9}\b",                                         "8-9 digit ID"),
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",    "Email"),
    (r"\b[A-Z][0-9]{7}\b",                                   "Passport code"),
    (r"\b[A-Z]{2}[0-9]{7}\b",                                "Passport code (2-letter)"),
    (r"\b[A-Z][0-9]{8}\b",                                   "Passport code (8-digit)"),
    (r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",                          "PAN-style ID"),
    (r"\b[A-Z]{2}[0-9]{6,8}\b",                             "Alphanumeric ID"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                        "IPv4 address"),
    (r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",         "Date"),
    (r"[A-Z0-9<]{8,}",                                       "MRZ / machine-readable code"),
    (r"\b\d{3}-\d{2}-\d{4}\b",                               "US SSN"),
    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",                     "US SSN (with separators)"),
    # License plate patterns
    (r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}\b",      "License plate (IN)"),
    (r"\b[A-Z]{2}\s?\d{2}\s?[A-Z]{2}\s?\d{4}\b",           "License plate (IN alt)"),
    (r"\b[0-9]{1}[A-Z]{3}\s?[0-9]{3}\b",                   "License plate (US-style)"),
    (r"\b[A-Z]{2,3}\s?\d{3,4}\s?[A-Z]{0,3}\b",             "License plate (EU-style)"),
    (r"\b[A-Z]{2}[0-9]{2}\s?[A-Z]{3}\b",                   "License plate (UK)"),
]

_COMPILED = [(re.compile(p), label) for p, label in _PII_PATTERNS]


def _cleanup_result_store(now_ts: float) -> None:
    """
    Evict old entries from the in-memory result cache.
    Ensures memory consumption remains bounded.
    """
    expired = []
    for key, item in _RESULT_STORE.items():
        if now_ts - item.get("ts", 0) > RESULT_TTL_SEC:
            expired.append(key)
    for key in expired:
        _RESULT_STORE.pop(key, None)

    if len(_RESULT_STORE) <= RESULT_MAX_ITEMS:
        return

    # Evict oldest first if size threshold is exceeded
    overflow = len(_RESULT_STORE) - RESULT_MAX_ITEMS
    oldest = sorted(_RESULT_STORE.items(), key=lambda kv: kv[1].get("ts", 0))[:overflow]
    for key, _ in oldest:
        _RESULT_STORE.pop(key, None)


def _store_result(result_b64: str, ocr_table: str, summary: str,
                  original_b64: str | None = None) -> str:
    """
    Store the processed base64 image and logs in the result cache.
    Returns a unique UUID string used by the frontend to fetch the cache.
    """
    now_ts = time.time()
    result_id = uuid.uuid4().hex
    with _RESULT_LOCK:
        _RESULT_STORE[result_id] = {
            "result_image": result_b64,
            "ocr_table": ocr_table,
            "summary": summary,
            "original_image": original_b64,
            "ts": now_ts,
        }
        _cleanup_result_store(now_ts)
    return result_id


def classify_text(text: str) -> tuple[bool, str]:
    """
    Classify a string detected by OCR.
    Iterates through compiled regex list. Returns (is_sensitive, matched_label).
    """
    for pattern, label in _COMPILED:
        if pattern.search(text):
            return True, label
    return False, ""


# ==============================================================================
# MASK GEOMETRY & MANIPULATION ENGINE
# ==============================================================================
# Custom geometrical algorithms designed to group, adjust, and shape
# the bounding boxes before converting them into binary mask images.

def _merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """
    Iteratively merges overlapping bounding boxes (x1, y1, x2, y2 format).
    Uses a standard intersection check to union overlapping regions.
    Merging overlapping detections prevents fragmented, close-proximity masks,
    which drastically improves the texture coherence of the neural inpainter.
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
                # Check intersection overlap
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
    Expands bounding boxes proportionally relative to their dimensions.
    - Small boxes (like small text strings) receive relatively larger padding
      to ensure the OCR boundaries are fully enclosed and not clipped.
    - Large boxes (like documents) receive smaller relative padding so we do
      not erase large sections of the clean surrounding background context.
    Clamps output boundaries to the source image dimensions.
    """
    bw = x2 - x1
    bh = y2 - y1
    img_area = img_h * img_w
    box_area = max(0, bw) * max(0, bh)
    frac = box_area / max(img_area, 1)

    # Scale down padding fraction for large regions to prevent full ID cards
    # from expanding and covering clean textures.
    if frac > 0.12:
        base_frac = min(base_frac, 0.08)
        max_px = min(max_px, 28)
    elif frac > 0.05:
        base_frac = min(base_frac, 0.12)
        max_px = min(max_px, 36)

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
    Draws a filled rounded rectangle onto the binary mask image.
    - Rounded corners (default ~12% of the short side) prevent sharp, right-angle
      inpainting seams which are highly visible and unnatural to the human eye.
    - Drawn by overlaying a horizontal rectangle, a vertical rectangle, and
      filling the four corner circles using OpenCV primitives.
    """
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return
    if radius <= 0:
        radius = max(4, int(min(bw, bh) * 0.12))
    radius = min(radius, bw // 2, bh // 2)  # Clamp to fit inside borders

    # Draw vertical and horizontal crosses to fill the center of the rectangle
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)

    # Fill in the four rounded corners
    cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)


def _box_area(x1, y1, x2, y2) -> int:
    """Calculates the pixel area of a bounding box."""
    return max(0, x2 - x1) * max(0, y2 - y1)


def _boxes_overlap(a, b) -> bool:
    """Checks if two (x1, y1, x2, y2) boxes intersect."""
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


# ==============================================================================
# AI INPAINTING & PRIVACY ENGINE
# ==============================================================================
# Highly optimized in-place reconstruction pipeline combining heuristic (Telea)
# and neural (LaMa) approaches, fortified with gradient-domain Poisson blending
# and local color harmonization.

def _make_binary(mask: np.ndarray) -> np.ndarray:
    """Enforce strictly binary mask — 0 or 255 values, nothing in between."""
    return np.where(mask > 0, np.uint8(255), np.uint8(0))


def _ensure_mask_size(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Guarantees the mask dimensions exactly match the image dimensions (h, w).
    Resizes using nearest-neighbor interpolation to prevent edge anti-aliasing.
    """
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return _make_binary(mask)


def _adaptive_dilate(mask: np.ndarray, region_area: int, img_area: int) -> np.ndarray:
    """
    Dilates the mask adaptively based on the relative size of the region.
    - Small text blocks (fraction < 0.005) are dilated heavily (k=11, iters=2)
      to cover edge-color bleeding from text outlines.
    - Medium blocks (fraction < 0.03) use moderate dilation (k=9, iters=1).
    - Large objects use thin dilation (k=7, iters=1) to prevent eating too much
      surrounding clean context needed for inpainting.
    """
    frac = region_area / max(img_area, 1)

    if frac < 0.005:
        k = 11
        iters = 2
    elif frac < 0.03:
        k = 9
        iters = 1
    else:
        k = 7
        iters = 1

    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iters)
    return _make_binary(dilated)


def _soft_edge_mask(mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    """
    Smoothes mask edges by applying a Gaussian blur, then re-thresholding.
    This eliminates jagged pixelated borders, which leads to a much smoother
    reconstruction transition at the boundary.
    """
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=2.0)
    return np.where(blurred > 100, np.uint8(255), np.uint8(0))


def _feather_alpha(mask: np.ndarray, ksize: int = 9) -> np.ndarray:
    """
    Generates a feathered alpha matte (values 0.0 to 1.0) from a binary mask.
    The interior of the mask remains 100% opaque (1.0), while the outer edges
    gradually blend down to 0.0, enabling seamless alpha-blending back onto the original.
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
    Creates a size-adaptive feathered alpha matte for final compositing.
    Dilates the mask slightly before feathering, matching region dimensions.
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
    Heuristic OpenCV Inpainting (Telea algorithm).
    Guarantees size safety and grayscale formatting of the mask.
    Used as a fast standalone eraser or as a structural pre-fill for LaMa.
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
    Runs the LaMa (Resolution-independent Neural Inpainting) model on CPU/GPU.
    - Size-safe guard: checks output dimensions and resizes back to original if needed.
    - Exception handling fallback: If LaMa runs out of memory or fails, it falls back
      to OpenCV Telea, ensuring the application never crashes.
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
    Generates a narrow pixel ring surrounding the mask boundary.
    This ring represents the local "context background" used for color sampling.
    """
    dilated = cv2.dilate(mask, np.ones((border_px, border_px), np.uint8), iterations=1)
    border = cv2.subtract(dilated, mask)
    return _make_binary(border)


def _color_harmonize(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalizes color statistics (Mean and Standard Deviation) inside the masked region
    to match the statistics of the surrounding context ring.
    This removes color shifts and lighting differences, making inpainted zones
    blend seamlessly with their local visual neighborhood.
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

        # Calculate statistics inside the mask
        inner_mean = channel[mask_bool].mean()
        inner_std = max(channel[mask_bool].std(), 1e-6)

        # Calculate statistics of the surrounding context ring
        ctx_mean = channel[border_bool].mean()
        ctx_std = max(channel[border_bool].std(), 1e-6)

        # Apply distribution shift
        channel[mask_bool] = ((channel[mask_bool] - inner_mean) / inner_std) * ctx_std + ctx_mean
        result[:, :, c] = channel

    return np.clip(result, 0, 255).astype(np.uint8)


def _poisson_blend(base_rgb: np.ndarray, inpainted_rgb: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
    """
    Performs gradient-domain Poisson Blending (OpenCV seamlessClone).
    This fuses the inpainted texture with the base image by matching local gradients,
    eliminating visible boundary seams.
    Falls back to normal alpha overlay if Poisson cloning fails.
    """
    h, w = base_rgb.shape[:2]
    mask = _ensure_mask_size(mask, h, w)

    if inpainted_rgb.shape[0] != h or inpainted_rgb.shape[1] != w:
        inpainted_rgb = cv2.resize(inpainted_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    ys, xs = np.where(mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        return base_rgb

    cx = int(xs.mean())
    cy = int(ys.mean())

    # Clamp coordinates to keep inside image boundary
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
        # Fallback overlay blend
        mask_3ch = np.stack([mask] * 3, axis=-1) > 127
        result = base_rgb.copy()
        result[mask_3ch] = inpainted_rgb[mask_3ch]
        return result


def _inpaint_single_region(image_rgb: np.ndarray,
                            region_mask: np.ndarray,
                            lama_model,
                            region_area: int = 0,
                            img_area_ref: int | None = None,
                            fast_mode: bool = False) -> np.ndarray:
    """
    Execution pipeline for inpainting a single isolated region:
      1. Adaptive Dilation: Expands mask based on region size to ensure edge containment.
      2. Edge Smoothing: Blurs mask edges to prevent hard pixel steps.
      3. Heuristic Pre-fill: Runs cv2.inpaint (Telea) to recover rough color gradients.
      4. Neural Inpainting: Runs LaMa neural model for realistic texture synthesis.
      5. Seamless Blending: Poisson clones the neural result to match image lighting gradients.
      6. Color Harmonization: Statistics shift to align average colors with local background.
      7. Feathered Composite: Blends back using feathered alpha matte to prevent double inpainting.
    """
    h, w = image_rgb.shape[:2]
    img_area = img_area_ref if img_area_ref else (h * w)
    region_area = region_area or cv2.countNonZero(region_mask)
    orig_mask = _ensure_mask_size(region_mask, h, w)
    mask_bin = orig_mask.copy()

    # Step 1: Adaptive dilation
    mask_bin = _adaptive_dilate(mask_bin, region_area, img_area)

    # Step 2: Soft-edge smoothing
    ksize = 9 if region_area > img_area * 0.02 else 7
    mask_bin = _soft_edge_mask(mask_bin, ksize=ksize)
    mask_bin = _make_binary(mask_bin)

    # Step 3: Fast cv2.inpaint pre-fill
    frac = region_area / max(img_area, 1)
    if fast_mode:
        telea_radius = 5 if frac < 0.02 else 9
    else:
        telea_radius = 5 if frac < 0.01 else (12 if frac > 0.05 else 7)

    telea_bgr = _safe_cv2_inpaint(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), mask_bin,
        radius=telea_radius,
    )
    result = cv2.cvtColor(telea_bgr, cv2.COLOR_BGR2RGB)

    # Step 4: Neural LaMa inpainting
    if fast_mode:
        if frac >= 0.02:
            result = _safe_lama(result, mask_bin, lama_model)
    else:
        num_lama = 2 if frac > 0.08 else 1
        for pass_n in range(num_lama):
            result = _safe_lama(result, mask_bin, lama_model)

    # Step 5: Poisson seamless blend (skipped for tiny text regions to optimize CPU performance)
    if not fast_mode and frac > 0.01:
        result = _poisson_blend(image_rgb, result, mask_bin)

    # Step 6: Color harmonization
    result = _color_harmonize(result, mask_bin)

    # Step 7: Final feathered composition onto original image
    alpha = _blend_alpha(orig_mask, region_area, img_area)
    alpha_3 = alpha[..., None]
    result = (result * alpha_3 + image_rgb * (1.0 - alpha_3)).astype(np.uint8)

    return result


def per_region_inpaint(image_rgb: np.ndarray,
                       mask: np.ndarray,
                       lama_model,
                       fast_mode: bool = False) -> Image.Image:
    """
    Connected Components Inpainting Router:
    - Splits the global binary mask into disconnected coordinate regions (islands).
    - Crops each region with local padding (relative to size) to limit CPU/GPU tensor overhead.
    - Sequentially executes the single region inpainting pipeline on each cropped patch.
    - Paste results back into the final image buffer.
    This connected components approach prevents huge masks from degrading global texture quality.
    """
    h, w = image_rgb.shape[:2]

    # Clean and binarize global mask
    mask = _soft_edge_mask(mask)
    mask = _make_binary(mask)

    if mask.max() == 0:
        return Image.fromarray(image_rgb)

    # Calculate labels and coordinates for connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    result = image_rgb.copy()
    regions_processed = 0
    img_area_ref = h * w

    for i in range(1, num_labels):  # Skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BOX_AREA:
            continue

        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Calculate bounding crop coordinates with localized padding
        pad = max(16, int(max(width, height) * (0.25 if fast_mode else 0.35)))
        pad = min(pad, 96 if not fast_mode else 64)
        x1 = max(0, left - pad)
        y1 = max(0, top - pad)
        x2 = min(w, left + width + pad)
        y2 = min(h, top + height + pad)

        crop_img = result[y1:y2, x1:x2]
        crop_labels = labels[y1:y2, x1:x2]
        region_mask = np.where(crop_labels == i, np.uint8(255), np.uint8(0))

        region_frac = area / img_area_ref
        if region_frac > LARGE_REGION_WARN_FRAC:
            print(f"    ⚠  Region {i}: {region_frac:.1%} of image — quality may degrade")

        # Process the single cropped region
        crop_out = _inpaint_single_region(
            crop_img,
            region_mask,
            lama_model,
            region_area=area,
            img_area_ref=img_area_ref,
            fast_mode=fast_mode,
        )
        result[y1:y2, x1:x2] = crop_out
        regions_processed += 1

    print(f"    Inpainted {regions_processed} region(s)")
    return Image.fromarray(result)



# ==============================================================================
# MODEL CONFIGURATION & INITIALIZATION
# ==============================================================================
# Detects available hardware (CUDA, Apple MPS, or CPU) and instantiates models.

print("=" * 60)
print("  VoidBox — Stage 9: Robust Mask Engineering")
print("=" * 60)

# Hardware acceleration routing
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
    # 1. Load YOLO model (prefer fine-tuned weights from MIDV-2020 training)
    if fine_tuned.exists():
        print("  YOLO:    fine_tuned.pt (MIDV-2020 PII detector)")
        yolo_model = YOLO(str(fine_tuned))
        using_pii_model = True
    else:
        print("  YOLO:    yolov8n.pt (generic -- run train.py for PII model)")
        yolo_model = YOLO(str(base_model)) if base_model.exists() else YOLO("yolov8n.pt")
        using_pii_model = False

    # 2. Load LaMa Neural Inpainter
    simple_lama = SimpleLama()
    print("  LaMa:    loaded")

    # 3. Load EasyOCR English reader (enforce CUDA GPU if available)
    print("  EasyOCR: loading...")
    ocr_reader = easyocr.Reader(['en'], gpu=(device == "cuda"), verbose=False)
    print("  EasyOCR: loaded")

    # 4. Load Haar Cascade fallbacks
    if _plate_cascade.empty():
        print("  Plates:  WARNING -- Haar cascade not found, plate detection disabled")
    else:
        print("  Plates:  Haar cascade loaded")

    if _face_cascade.empty():
        print("  Faces:   WARNING -- Haar cascade not found, face fallback disabled")
    else:
        print("  Faces:   Haar cascade loaded (fallback)")
    print("=" * 60)

except Exception as e:
    print(f"  ERROR loading models: {e}")
    exit(1)


# ==============================================================================
# CORE MULTI-MODAL REDACTION PIPELINE
# ==============================================================================

def redact_pii(input_image,
               redact_documents, redact_faces,
               redact_signatures, redact_text_fields,
               redact_plates,
               ocr_mode,
               ocr_confidence,
               show_detections,
               fast_mode: bool = False):
    """
    Main redaction entrypoint executing a three-stage pipeline:
      1. Detection: Run YOLOv8, Haar Cascades, and EasyOCR.
      2. Mask Construction: Merge boxes, pad proportionally, fill gaps, draw rounded rectangles.
      3. Inpainting: Crop each connected component and run Telea + LaMa.
    """
    if input_image is None:
        return None, None, "No image provided."

    # Standardize image array to RGB (and copy to BGR for OpenCV functions)
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

    # Track detections for OCR gap-filling
    yolo_boxes_by_label = {v["name"]: [] for v in PII_CLASSES.values()}
    gap_fill_boxes: list[tuple[int, int, int, int]] = []
    
    t_start = time.time()

    # --------------------------------------------------------------------------
    # Phase 1: YOLO Object Detection
    # --------------------------------------------------------------------------
    print("  [Perf] Starting YOLO detection...")
    t0 = time.time()
    if fast_mode:
        yolo_results = yolo_model(image_bgr, imgsz=960, device=device, verbose=False)[0]
    else:
        yolo_results = yolo_model(image_bgr, device=device, verbose=False)[0]

    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id not in active_classes:
            continue
        cls_info = active_classes[cls_id]
        if conf < cls_info["conf"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Filter noise via class-specific minimum area checks
        min_area = cls_info.get("min_area", MIN_BOX_AREA)
        if _box_area(x1, y1, x2, y2) < min_area:
            continue

        yolo_boxes_by_label[cls_info["name"]].append((x1, y1, x2, y2))
        counts[cls_info["name"]] += 1

        if show_detections:
            color = cls_info["color"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = f"{cls_info['name']} {conf:.2f}"
            (tw, th_t), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th_t - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, lbl, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # --------------------------------------------------------------------------
    # Phase 1.1: Haar Cascade Face Detection (Fallback)
    # --------------------------------------------------------------------------
    # Runs when YOLO face detections might fail on small passport photos.
    if redact_faces and not _face_cascade.empty():
        gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(
            gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24),
        )
        for (fx, fy, fw, fh) in faces:
            x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
            if _box_area(x1, y1, x2, y2) < MIN_BOX_AREA:
                continue
            # Skip if box already overlaps with a YOLO-detected face
            if any(_boxes_overlap((x1, y1, x2, y2), b) for b in yolo_boxes_by_label["face"]):
                continue
            yolo_boxes_by_label["face"].append((x1, y1, x2, y2))
            counts["face"] += 1
            if show_detections:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), FACE_COLOR, 2)
                lbl = "face (cascade)"
                (tw, th_t), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th_t - 8), (x1 + tw + 4, y1), FACE_COLOR, -1)
                cv2.putText(annotated, lbl, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Convert YOLO regions to mask layouts
    mask_labels = {"face", "signature", "text_field"}
    if redact_documents and ocr_mode == "off":
        mask_labels.add("document")

    gap_fill_labels = {"plate", "text_field"}
    gap_fill_max_frac = 0.12

    for label, boxes in yolo_boxes_by_label.items():
        if not boxes:
            continue
        merged = _merge_boxes(boxes)
        for bx1, by1, bx2, by2 in merged:
            ex1, ey1, ex2, ey2 = _proportional_expand(bx1, by1, bx2, by2, h, w)
            if label in mask_labels:
                _draw_rounded_rect(mask, ex1, ey1, ex2, ey2)
            box_area = _box_area(ex1, ey1, ex2, ey2)
            if label in gap_fill_labels and (box_area / max(h * w, 1)) <= gap_fill_max_frac:
                gap_fill_boxes.append((ex1, ey1, ex2, ey2))
        
    print(f"  [Perf] YOLO detection done in {time.time() - t0:.2f}s")

    # --------------------------------------------------------------------------
    # Phase 1.5: License Plate Detection (Haar Cascade)
    # --------------------------------------------------------------------------
    print("  [Perf] Starting Plate detection...")
    t0 = time.time()
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

    # Merge plate boxes, expand, and draw on mask
    merged_plates = _merge_boxes(plate_boxes)
    for bx1, by1, bx2, by2 in merged_plates:
        ex1, ey1, ex2, ey2 = _proportional_expand(bx1, by1, bx2, by2, h, w)
        _draw_rounded_rect(mask, ex1, ey1, ex2, ey2)
        gap_fill_boxes.append((ex1, ey1, ex2, ey2))

    print(f"  [Perf] Plate detection done in {time.time() - t0:.2f}s")

    # --------------------------------------------------------------------------
    # Phase 2: EasyOCR Text Detection (Optimized for Speed)
    # --------------------------------------------------------------------------
    # Text detection can be a latency bottleneck. We optimize it with local CLAHE contrast
    # enhancement and adaptive downscaling on massive images to speed up OCR inference.
    print("  [Perf] Starting EasyOCR detection...")
    t0 = time.time()
    if ocr_mode != "off":
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Latency optimization: Downscale very large images before passing to OCR
        resize_factor = 1.0
        max_dim = 900 if fast_mode else 1200
        if max(h, w) > max_dim:
            resize_factor = max_dim / max(h, w)
            new_w, new_h = int(w * resize_factor), int(h * resize_factor)
            ocr_input = cv2.resize(gray_image, (new_w, new_h))
        else:
            ocr_input = gray_image
            
        # Local contrast normalization (CLAHE) helps identify low-contrast or faded text
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        clahe_image = clahe.apply(ocr_input)
        
        # Extract text boxes (mag_ratio=1.0 keeps inference times reasonable)
        ocr_results = ocr_reader.readtext(clahe_image, mag_ratio=1.0, paragraph=False)
        ocr_pii_boxes = []

        for (bbox, text, prob) in ocr_results:
            if prob < ocr_confidence:
                continue
                
            # Scale coordinates back up to full resolution if image was resized
            if resize_factor != 1.0:
                bbox = [[pt[0] / resize_factor, pt[1] / resize_factor] for pt in bbox]

            # Run regex classifier to filter sensitive fields
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

        # ----------------------------------------------------------------------
        # Phase 2.5: Gap-Filling Algorithm
        # ----------------------------------------------------------------------
        # If multiple individual OCR words fall inside a single macro object region
        # (like field blocks on license plates or document IDs), we redact the
        # macro box instead of individual text holes, avoiding "swiss cheese" masks.
        filled_boxes = []
        used_ocr = set()
        for obj_box in gap_fill_boxes:
            group = []
            for idx, ocr_box in enumerate(ocr_pii_boxes):
                if idx in used_ocr:
                    continue
                if _boxes_overlap(obj_box, ocr_box):
                    group.append(idx)
            if len(group) >= 2:
                filled_boxes.append(obj_box)
                used_ocr.update(group)

        # Keep remaining standalone OCR boxes
        for idx, ocr_box in enumerate(ocr_pii_boxes):
            if idx not in used_ocr:
                filled_boxes.append(ocr_box)

        # Merge boxes, expand padding, and draw on the global mask
        merged_ocr = _merge_boxes(filled_boxes)
        for bx1, by1, bx2, by2 in merged_ocr:
            ex1, ey1, ex2, ey2 = _proportional_expand(bx1, by1, bx2, by2, h, w)
            _draw_rounded_rect(mask, ex1, ey1, ex2, ey2)
            
    print(f"  [Perf] EasyOCR detection done in {time.time() - t0:.2f}s")

    # --------------------------------------------------------------------------
    # Phase 3: Mask Statistics Logging
    # --------------------------------------------------------------------------
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

    # Build OCR table logs for UI preview
    if ocr_log_rows:
        table_lines = ["| Detected Text | Pattern | Action |", "|---|---|---|"]
        for text, pat, action in ocr_log_rows:
            safe_text = text.replace("|", "\\|")
            table_lines.append(f"| `{safe_text}` | {pat} | {action} |")
        ocr_table_md = "\n".join(table_lines)
    else:
        ocr_table_md = "_No text detected by OCR._"

    # --------------------------------------------------------------------------
    # Phase 3.5: Build Summary Report
    # --------------------------------------------------------------------------
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
    if redact_documents and ocr_mode != "off":
        summary += "\n\nDocument masking is limited when OCR is enabled to avoid erasing entire IDs."
    if redact_signatures and not using_pii_model:
        summary += "\n\nSignature detection is limited without the fine-tuned PII model."
    if fast_mode:
        summary += "\n\nFast mode enabled — lower latency, slightly reduced quality."

    # --------------------------------------------------------------------------
    # Phase 3.6: Show Detections Preview Mode
    # --------------------------------------------------------------------------
    if show_detections:
        out = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        summary += "\n\nPreview mode enabled — no redaction performed."
        return out, ocr_table_md, summary

    if mask_binary.max() == 0:
        return input_image, ocr_table_md, summary

    # --------------------------------------------------------------------------
    # Phase 4: Per-Region Hybrid Inpainting Execution
    # --------------------------------------------------------------------------
    print("  [Perf] Starting Inpainting...")
    t0 = time.time()
    result_image = per_region_inpaint(image_rgb, mask, simple_lama, fast_mode=fast_mode)
    print(f"  [Perf] Inpainting done in {time.time() - t0:.2f}s")
    
    print(f"  [Perf] TOTAL PIPELINE DONE IN {time.time() - t_start:.2f}s")
    
    summary += "\n\nAll flagged PII has been AI-erased (per-region hybrid inpainting)."
    return result_image, ocr_table_md, summary


# ==============================================================================
# FLASK WEB APP & API ROUTINGS
# ==============================================================================
# Sets up a lightweight Flask server to host local HTTP endpoints and serve
# UI files out of the 'frontend/' directory.

flask_app = Flask(__name__, static_folder="frontend", static_url_path="")
FRONTEND_DIR = Path(__file__).parent / "frontend"


@flask_app.route("/")
def serve_index():
    """Serves the primary UI dashboard file (index.html)."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@flask_app.route("/<path:filename>")
def serve_static(filename):
    """Serves static assets (JS, CSS, images) from the frontend folder."""
    return send_from_directory(FRONTEND_DIR, filename)


@flask_app.route("/api/health", methods=["GET"])
def api_health():
    """
    Health check and capability advertisement endpoint.
    Advertises loaded models, cascading status, and hardware devices (e.g. CUDA/MPS/CPU).
    """
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
    Accepts JSON containing:
      {
        "image": "<base64-encoded string>",
        "redact_documents": bool,
        "redact_faces": bool,
        "redact_signatures": bool,
        "redact_text_fields": bool,
        "redact_plates": bool,
        "ocr_mode": "smart"|"all"|"off",
        "ocr_confidence": float,
        "fast_mode": bool,
        "show_detections": bool
      }
    Returns JSON containing:
      {
        "result_image": "data:image/png;base64,...",
        "ocr_table": "<markdown table showing words classified>",
        "summary": "<redaction statistics summary markdown>",
        "result_id": "<uuid cache key>"
      }
    """
    try:
        data = request.get_json(force=True)

        # 1. Base64 Image Decoding
        orig_data_url = data.get("image", "") or ""
        img_b64 = orig_data_url
        if "base64," in img_b64:
            img_b64 = img_b64.split("base64,")[1]
            
        try:
            img_bytes = base64.b64decode(img_b64)
            pil_input = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image encoding: {str(e)}"}), 400

        # 2. Execute Redaction Pipeline
        result_pil, ocr_table, summary = redact_pii(
            input_image        = pil_input,
            redact_documents   = bool(data.get("redact_documents", True)),
            redact_faces       = bool(data.get("redact_faces", True)),
            redact_signatures  = bool(data.get("redact_signatures", True)),
            redact_text_fields = bool(data.get("redact_text_fields", True)),
            redact_plates      = bool(data.get("redact_plates", True)),
            ocr_mode           = data.get("ocr_mode", "smart"),
            ocr_confidence     = float(data.get("ocr_confidence", 0.4)),
            show_detections    = bool(data.get("show_detections", False)),
            fast_mode          = bool(data.get("fast_mode", False)),
        )

        # 3. Base64 Output Encoding
        buf = io.BytesIO()
        if isinstance(result_pil, Image.Image):
            result_pil.save(buf, format="PNG")
        else:
            Image.fromarray(np.array(result_pil)).save(buf, format="PNG")
        result_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        # 4. Cache Result and return payload
        result_id = _store_result(result_b64, ocr_table, summary, original_b64=orig_data_url)

        return jsonify({
            "result_image": result_b64,
            "ocr_table": ocr_table,
            "summary": summary,
            "result_id": result_id,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@flask_app.route("/api/result/<result_id>", methods=["GET"])
def api_result(result_id: str):
    """Fetches a previously processed redaction result from cache to avoid session storage bloating."""
    now_ts = time.time()
    with _RESULT_LOCK:
        _cleanup_result_store(now_ts)
        item = _RESULT_STORE.get(result_id)

    if not item:
        return jsonify({"error": "Result not found or expired."}), 404

    payload = {k: v for k, v in item.items() if k != "ts"}
    return jsonify(payload)


@flask_app.route("/api/download-model", methods=["GET"])
def api_download_model():
    """Streams the fine-tuned model checkpoint file to users directly from the server storage."""
    model_path = Path(__file__).parent / "models" / "fine_tuned.pt"
    if not model_path.exists():
        return jsonify({"error": "Fine-tuned model not found. Run train.py first."}), 404
    return send_file(
        str(model_path),
        as_attachment=True,
        download_name="voidbox_fine_tuned.pt",
        mimetype="application/octet-stream",
    )


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    model_tag = "Fine-tuned PII model" if using_pii_model else "Generic YOLOv8n"
    print(f"\n  Model: {model_tag}")
    print(f"  Frontend: http://localhost:7860")
    print(f"  API:      http://localhost:7860/api/health\n")
    flask_app.run(host="0.0.0.0", port=7860, debug=False)

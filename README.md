# VoidBox — Multi-Modal PII Detection and Redaction Engine

> **Evolved from [Context-Aware AI Eraser](https://github.com/vishva2410/context-aware-ai-eraser)**
> — what started as a face/plate/ID blur tool is now a full-stack AI privacy engine with object detection, OCR, semantic filtering, and neural inpainting.

Privacy-first image anonymization that combines YOLOv8 object detection, EasyOCR text recognition, regex-based PII classification, and LaMa neural inpainting into a single pipeline. No cloud. No data leaves your machine.

---

## Architecture

```
                        +--------------------+
                        |   Input Image      |
                        +--------+-----------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
+--------v---------+   +---------v--------+   +---------v--------+
|   YOLOv8 Model   |   |  Haar Cascade    |   |    EasyOCR       |
| (fine-tuned on   |   | (license plate   |   | (text detection)  |
|  MIDV-2020 docs) |   |  detection)      |   |                   |
+--------+---------+   +---------+--------+   +---------+--------+
         |                       |                       |
         |  documents, faces,    |  license plates       |  all visible text
         |  signatures,          |                       |  strings
         |  text_fields          |              +--------v---------+
         |                       |              |  PII Classifier  |
         |                       |              |  (regex engine)   |
         |                       |              +--------+---------+
         |                       |                       |
         +-----------------------+-----------------------+
                                 |
                        +--------v---------+
                        | Smart Masking    |
                        |  • Min area filt |
                        |  • Box merging   |
                        |  • Proportional  |
                        |    expansion     |
                        |  • Rounded rects |
                        |  • OCR gap fill  |
                        +--------+---------+
                                 |
                        +--------v---------+
                        |  Soft-Edge Mask  |
                        |  + Binary Enforce|
                        +--------+---------+
                                 |
                        +--------v---------+
                        | Per-Region       |
                        | Inpainting       |
                        |  Telea → LaMa ×2 |
                        +--------+---------+
                                 |
                        +--------v---------+
                        |  Redacted Image  |
                        +------------------+
```

---

## Evolution Timeline

```
Stage 1-5    Context-Aware AI Eraser
             - YOLO-based face, plate, ID detection
             - Gaussian blur / solid fill redaction
             - Public vs Private context toggle
             - Basic web UI
                  |
                  v
Stage 6      + EasyOCR Integration
             - Added text detection layer
             - Unified object + text mask
             - Replaced blur with LaMa inpainting
             - Upgraded to Gradio Blocks UI
                  |
                  v
Stage 7      + Semantic PII Filtering
             - Regex-based text classification
             - 11 PII pattern categories
             - Smart / Aggressive / Off OCR modes
             - Live text analysis table in UI
             - Production-grade binary mask pipeline
                  |
                  v
Stage 8      + Hybrid Inpainting + Plate Detection
             - Haar cascade license plate detection
             - License plate regex patterns (IN/US/EU)
             - Soft-edge mask smoothing
             - Unified inpainting (Telea + LaMa × 2)
                  |
                  v
Stage 9      + Robust Mask Engineering
             - Overlapping box merging
             - Proportional adaptive expansion
             - Rounded rectangle masks
             - OCR internal gap filling
             - Per-region inpainting
             - Minimum area noise filter
             - Confidence threshold tuning
             - Mask statistics logging
```

---

## Capabilities

### Detection Layers

| Layer | Engine | What It Finds |
|-------|--------|---------------|
| Object Detection | YOLOv8 (fine-tuned on MIDV-2020) | Documents, faces, signatures, text fields |
| Text Detection | EasyOCR | All visible text in the image |
| PII Classification | Regex Engine | Filters text into sensitive vs safe |

### PII Patterns Recognised

| Category | Pattern | Example |
|----------|---------|---------|
| Aadhaar-style | 12-digit number | `123456789012` |
| Card number | 16-digit number | `4111111111111111` |
| Phone number | 10-digit number | `9876543210` |
| Enrollment ID | 8-9 digit number | `00123456` |
| Email | standard format | `user@domain.com` |
| Passport | letter + 7 digits | `A1234567` |
| PAN-style | ABCDE1234F | `BXYPK4321M` |
| Vehicle/alpha ID | 2 letters + 6-8 digits | `MH06AB1234` |
| IP address | dotted quad | `192.168.1.1` |
| Date | DD/MM/YYYY variants | `20/02/2026` |
| MRZ | 8+ uppercase alphanum | `P<INDBOSE<<SUBHAS` |
| License plate (IN) | XX 00 XXX 0000 | `MH12ABC1234` |
| License plate (US) | 0XXX 000 | `1ABC234` |
| License plate (EU) | XX 000 XXX | `AB123CD` |

### Redaction Modes

| Mode | Behaviour |
|------|-----------|
| Smart (Stage 7) | Only erases text matching PII patterns |
| Aggressive (Stage 6) | Erases all detected text regardless |
| Off | Disables OCR entirely, YOLO-only |

---

## Project Structure

```
yolov8_project/
|-- app.py                     Main application (Gradio UI + pipeline)
|-- train.py                   YOLOv8 fine-tuning script
|-- download_midv2020.py       Dataset downloader (Roboflow API)
|-- prepare_dataset.py         Dataset validation and class remapping
|-- prepare_signatures.py      Signature dataset preparation for YOLO
|-- generate_synthetic_data.py Synthetic ID card generator (bootstrap)
|-- models/
|   |-- fine_tuned.pt          Fine-tuned PII detection model
|   |-- yolov8n.pt             Base YOLOv8 model (fallback)
|-- datasets/
|   |-- midv2020/              Real identity document dataset
|   |-- custom_id_data/        Synthetically generated training data
|   |-- signatures/            Signature detection dataset
|-- outputs/                   Test output directory
|-- venv/                      Python virtual environment
```

---

## Setup

### Prerequisites

- Python 3.10+
- macOS / Linux / Windows (MPS, CUDA, or CPU)

### Installation

```bash
cd "void box/yolov8_project"
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install ultralytics opencv-python-headless gradio
pip install simple-lama-inpainting easyocr torch numpy pillow pyyaml
```

### Run

```bash
source venv/bin/activate
python app.py
```

The Gradio interface launches at `http://localhost:7860` with a public share link.

---

## Training Your Own Model

### Option A: Download MIDV-2020 (recommended)

```bash
python download_midv2020.py --api-key YOUR_ROBOFLOW_KEY
python prepare_dataset.py --remap
python train.py --epochs 50
```

### Option B: Synthetic data (quick test)

```bash
python generate_synthetic_data.py
python train.py --data datasets/custom_id_data/data.yaml --epochs 10
```

The best model is automatically saved to `models/fine_tuned.pt` and loaded by `app.py` on next launch.

---

## Technical Details

### Mask Generation Pipeline

1. **Minimum area filter** — boxes smaller than 100px² are discarded as noise.
2. **Overlapping box merging** — `_merge_boxes()` iteratively unions overlapping detections into single regions.
3. **Proportional expansion** — `_proportional_expand()` adapts padding to box size (15% of max dimension, clamped 8–40px).
4. **Rounded rectangle masks** — `_draw_rounded_rect()` draws masks with rounded corners (~12% of short side) to reduce seam artifacts.
5. **OCR gap filling** — when multiple OCR words overlap with a single YOLO/plate region, the whole object box replaces individual word boxes.
6. **Soft-edge smoothing** — Gaussian blur on mask edges + re-threshold removes harsh borders.
7. **Binary enforcement** — `np.where(mask > 0, 255, 0)` guarantees no anti-aliased edges.
8. **Per-region inpainting** — each connected component gets its own Telea → LaMa × 2 pass, preventing large combined masks from degrading quality.
9. **Mask statistics logging** — prints region count, total masked area %, and per-region size stats.

### Device Selection

Automatically selects the best available backend:

```
CUDA (NVIDIA GPU)  >  MPS (Apple Silicon)  >  CPU
```

---

## Limitations

- OCR may miss small, low-contrast, or stylised text
- Regex does not catch names, addresses, or contextual secrets (requires NER / LLM — future work)
- LaMa inpainting quality depends on mask precision and surrounding texture availability
- Fine-tuned model accuracy depends on training data diversity

---

## Roadmap

```
Current    Stage 9: Robust Mask Engineering
                |
Planned    Stage 10: Named Entity Recognition (spaCy / transformer)
                |
           Stage 11: LLM-based contextual classification
                |
           Stage 12: Video pipeline (frame-by-frame redaction)
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**VoidBox** — from blur tool to AI privacy engine.

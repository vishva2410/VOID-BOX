# 🕳️ VoidBox
GAN  based, which is context-aware. This is a continuation of the project context-aware ai. 
> **Automated, Irreversible PII Redaction Pipeline**
>
> *Real-time sanitization of sensitive documents using Object Detection (YOLOv8) and Fourier-Convolution GANs (LaMa).*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)](https://github.com/ultralytics/ultralytics)
[![LaMa](https://img.shields.io/badge/Inpainting-LaMa-orange)](https://github.com/advimman/lama)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

**VoidBox** is a security-first computer vision tool designed to detect sensitive documents (Passports, IDs, Credit Cards) and **permanently** remove them from images.

Unlike Gaussian blurring or pixelation—which are reversible using modern De-blurring GANs—VoidBox uses **Generative Adversarial Networks (LaMa)** to hallucinate a synthetic background texture over the sensitive area. The PII is not just hidden; it is mathematically erased.

## 📸 Demo

*(Insert your GIF here showing the detection -> mask -> inpaint process)*

## 🧠 The Architecture (Cascade Pipeline)

VoidBox operates as a two-stage cascade pipeline to ensure **100% coverage** of sensitive data.

### 1. The Eye: Detection (YOLOv8)
* **Model:** YOLOv8 (Nano/Small) fine-tuned on the **MIDV-2020** dataset (Synthetic IDs).
* **Role:** Single-shot detection of `credit_card`, `passport`, `id_card`, and `signature`.
* **Performance:** ~30 FPS on CPU.

### 2. The Bridge: Dilation (The "Secret Sauce")
* **Problem:** Raw bounding boxes are often "tight," risking exposure of edge pixels (e.g., the last digit of a card number).
* **Solution:** A morphological **Dilation** operation expands the binary mask by 15-20 pixels.
* **Result:** Guarantees total encapsulation of the sensitive object before erasure.

### 3. The Eraser: Inpainting (LaMa)
* **Model:** Large Mask Inpainting (LaMa).
* **Technique:** Uses **Fast Fourier Convolutions (FFC)**. Unlike standard CNNs that struggle with large holes, LaMa analyzes the image in the frequency domain (global context) to synthesize seamless background textures (e.g., continuing a table pattern through the hole).

## 🛠️ Tech Stack

* **Core:** Python 3.9+, PyTorch
* **Computer Vision:** Ultralytics YOLOv8, OpenCV (Morphological Ops)
* **Inpainting:** SimpleLama (LaMa Wrapper)
* **Interface:** Gradio (for Human-in-the-loop validation)

## 📦 Installation & Usage

```bash
# 1. Clone the repository
git clone [https://github.com/VishvaTeja/VoidBox.git](https://github.com/VishvaTeja/VoidBox.git)
cd VoidBox

# 2. Install dependencies (CPU/CUDA agnostic)
pip install -r requirements.txt

# 3. Run the Gradio App
python app.py

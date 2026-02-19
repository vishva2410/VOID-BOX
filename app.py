import gradio as gr
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path
from simple_lama_inpainting import SimpleLama
from PIL import Image

# Initialize Models Globally
print("Initializing models...")
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

try:
    # Check for fine-tuned model first
    # Using Path relative to this script
    script_dir = Path(__file__).parent
    fine_tuned_model_path = script_dir / 'fine_tuned.pt'
    
    if fine_tuned_model_path.exists():
        print(f"Loading fine-tuned model from {fine_tuned_model_path}...")
        model = YOLO(fine_tuned_model_path)
    else:
        print(f"Fine-tuned model not found at {fine_tuned_model_path}")
        print("Loading default YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        
    simple_lama = SimpleLama()
    print(f"Models loaded successfully on {device}")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

def redact_objects(input_image):
    """
    Takes a PIL Image, detects objects using YOLOv8,
    generates a dilated mask, and removes them using LaMa inpainting.
    Returns the redacted PIL Image.
    """
    if input_image is None:
        return None

    # Convert PIL Image to OpenCV Image (RGB -> BGR)
    # Gradio passes images as numpy arrays (RGB) or PIL Images. 
    # If type='pil', input_image is a PIL Image.
    image_np = np.array(input_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 1. Run YOLOv8 Inference
    results = model(image_bgr, device=device)
    result = results[0]

    # 2. Generate Binary Mask
    h, w, _ = image_bgr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    detections_found = False
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        detections_found = True

    if not detections_found:
        print("No objects detected.")
        return input_image

    # 3. Apply Mask Dilation
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # 4. Inpainting with LaMa
    # Convert inputs to PIL format for LaMa
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(dilated_mask)

    # Run Inpainting
    result_image = simple_lama(image_pil, mask_pil)
    
    return result_image

# Build Gradio Interface
if __name__ == '__main__':
    demo = gr.Interface(
        fn=redact_objects,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=gr.Image(type="pil", label="Redacted Image"),
        title="Privacy Lens: AI Object Redaction",
        description="Upload an image to automatically detect and remove objects using YOLOv8 and LaMa inpainting.",
        examples=["test.jpg"] if Path("test.jpg").exists() else None
    )
    
    print("Launching Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


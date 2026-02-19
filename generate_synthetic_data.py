import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configuration
DATASET_DIR = "datasets/custom_id_data"
IMG_SIZE = (640, 640)
NUM_TRAIN = 50
NUM_VAL = 10
CLASSES = {0: "id_card", 1: "face", 2: "text_field"}

def create_directory_structure():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

def generate_synthetic_image(image_id, split):
    # Create a blank background (like a table)
    bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    img = Image.new("RGB", IMG_SIZE, bg_color)
    draw = ImageDraw.Draw(img)
    
    # ID Card Dimensions
    card_w = random.randint(300, 400)
    card_h = int(card_w * 0.63) # Standard ID ratio
    
    # Random Position
    x = random.randint(50, IMG_SIZE[0] - card_w - 50)
    y = random.randint(50, IMG_SIZE[1] - card_h - 50)
    
    # Draw ID Card Background
    card_color = (random.randint(230, 255), random.randint(230, 255), random.randint(240, 255))
    draw.rectangle([x, y, x + card_w, y + card_h], fill=card_color, outline=(0,0,0))
    
    labels = []
    
    # 1. ID Card Bounding Box
    # YOLO format: class x_center y_center width height (normalized)
    xc, yc = (x + card_w/2)/IMG_SIZE[0], (y + card_h/2)/IMG_SIZE[1]
    w, h = card_w/IMG_SIZE[0], card_h/IMG_SIZE[1]
    labels.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    
    # 2. Face Photo (Left side)
    photo_w = int(card_w * 0.25)
    photo_h = int(card_h * 0.5)
    px = x + int(card_w * 0.05)
    py = y + int(card_h * 0.25)
    draw.rectangle([px, py, px + photo_w, py + photo_h], fill=(100, 100, 100), outline=(0,0,0))
    
    xc, yc = (px + photo_w/2)/IMG_SIZE[0], (py + photo_h/2)/IMG_SIZE[1]
    w, h = photo_w/IMG_SIZE[0], photo_h/IMG_SIZE[1]
    labels.append(f"1 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    
    # 3. Text Fields (Name, ID, etc.)
    text_x_start = px + photo_w + 20
    for i in range(3):
        text_w = random.randint(int(card_w * 0.3), int(card_w * 0.5))
        text_h = int(card_h * 0.08)
        tx = text_x_start
        ty = y + int(card_h * (0.3 + i * 0.15))
        
        draw.rectangle([tx, ty, tx + text_w, ty + text_h], fill=(0, 0, 0))
        
        xc, yc = (tx + text_w/2)/IMG_SIZE[0], (ty + text_h/2)/IMG_SIZE[1]
        w, h = text_w/IMG_SIZE[0], text_h/IMG_SIZE[1]
        labels.append(f"2 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
    # Save Image
    img_path = os.path.join(DATASET_DIR, "images", split, f"{image_id}.jpg")
    img.save(img_path)
    
    # Save Label
    label_path = os.path.join(DATASET_DIR, "labels", split, f"{image_id}.txt")
    with open(label_path, "w") as f:
        f.write("\n".join(labels))

def create_yaml():
    yaml_content = f"""
path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val

names:
  0: id_card
  1: face
  2: text_field
"""
    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content.strip())

if __name__ == "__main__":
    create_directory_structure()
    print("Generating training data...")
    for i in range(NUM_TRAIN):
        generate_synthetic_image(f"train_{i}", "train")
        
    print("Generating validation data...")
    for i in range(NUM_VAL):
        generate_synthetic_image(f"val_{i}", "val")
        
    create_yaml()
    print(f"Dataset generated at {DATASET_DIR}")
    print("Classes: 0: id_card, 1: face, 2: text_field")

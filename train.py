from ultralytics import YOLO

def main():
    # Load the YOLOv8 Nano model
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    # Train the model
    # data='datasets/custom_id_data/data.yaml' is the config we generated
    # epochs=10 for demonstration
    # imgsz=640 standard YOLO image size
    print("Starting training...")
    try:
        model.train(
            data='datasets/custom_id_data/data.yaml',
            epochs=10,
            imgsz=640,
            device='mps'  # Use Metal Performance Shaders for Mac
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == '__main__':
    main()

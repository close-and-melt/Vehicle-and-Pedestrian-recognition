"""
Training script for YOLO vehicle and pedestrian detection model.

Uses YOLOv11n as base model and trains on filtered dataset containing
only Pedestrian and car classes.
"""

from ultralytics import YOLO


def train() -> None:
    """
    Train YOLO model for vehicle and pedestrian detection.

    Configuration:
        - Base model: YOLOv11n (nano version)
        - Dataset: Filtered dataset with Pedestrian and car classes
        - Training epochs: 50
        - Image size: 480x480
        - Batch size: 4 (reduced to prevent GPU out-of-memory)
        - Device: GPU 0 (CUDA)
        - Workers: 2 (data loading threads)
        - Cache: disk (avoids memory conflicts)
        - Early stopping patience: 20 epochs
        - Plots: enabled (generates training visualization charts)
    """
    # Load pre-trained YOLOv11n model
    model = YOLO("yolo11n.pt")

    # Start training
    results = model.train(
        data=r"E:\YOLO_training_filtered\data.yaml",  # Path to dataset configuration
        epochs=50,  # Number of training epochs
        imgsz=480,  # Input image size
        batch=4,  # Batch size (reduced to prevent OOM)
        device=0,  # GPU device ID (0 = first GPU, use 'cpu' for CPU)
        workers=2,  # Number of data loading workers
        cache="disk",  # Cache images to disk (avoid memory conflicts)
        project=r"E:\YOLO_training\runs\train",  # Root directory for saving outputs
        name="vehicle_pedestrian_v1",  # Subdirectory name for this run
        patience=20,  # Early stopping patience (stop if no improvement for 20 epochs)
        plots=True,  # Generate training plots (loss curves, PR curves, etc.)
    )


if __name__ == "__main__":
    train()
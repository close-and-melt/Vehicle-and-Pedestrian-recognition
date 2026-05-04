"""
Training script for YOLO vehicle and pedestrian detection model.

Uses YOLOv11 as base model and trains on a dataset containing
Pedestrian and car classes.
"""

import argparse
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
    parser = argparse.ArgumentParser(description="YOLO Vehicle/Pedestrian Training")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="Path to dataset config file (default: data.yaml)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Base model weights (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="Input image size (default: 480)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device ID or 'cpu' (default: 0)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Data loading workers (default: 2)")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Root directory for saving outputs (default: runs/train)")
    parser.add_argument("--name", type=str, default="vehicle_pedestrian",
                        help="Subdirectory name for this run (default: vehicle_pedestrian)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--cache", type=str, default="disk",
                        help="Cache mode: disk, ram, or None (default: disk)")
    parser.add_argument("--plots", action="store_true", default=True,
                        help="Generate training plots")
    parser.add_argument("--no-plots", action="store_false", dest="plots",
                        help="Disable training plots")
    args = parser.parse_args()

    # Load the YOLO model (pre-trained weights or a previous checkpoint)
    model = YOLO(args.model)

    # Train the model. The dataset config (data.yaml) must specify:
    #   - train/val/test image paths
    #   - number of classes (nc) and class names
    # Batch size is kept low by default to avoid GPU out-of-memory on consumer GPUs.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project=args.project,
        name=args.name,
        patience=args.patience,
        plots=args.plots,
    )


if __name__ == "__main__":
    train()

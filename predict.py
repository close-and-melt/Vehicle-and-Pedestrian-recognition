"""
Prediction script for YOLO vehicle and pedestrian detection model.

Performs object detection on images, videos, or camera streams
using a trained YOLO model.
"""

import argparse
from ultralytics import YOLO


def predict() -> None:
    """
    Run inference on images, video files, or camera streams using a trained YOLO model.

    Configuration:
        - Model: Trained checkpoint (.pt file, required)
        - Input source: Image path, video path, or camera ID (required)
        - Confidence threshold: 0.4 (minimum confidence to show detection)
        - IOU threshold: 0.45 (Non-Maximum Suppression overlap threshold)
        - Image size: 480 (must match training image size)
        - Device: GPU 0 (CUDA for faster inference)
        - Output: Display real-time window + save video file
    """
    parser = argparse.ArgumentParser(description="YOLO Vehicle/Pedestrian Prediction")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model weights (.pt file)")
    parser.add_argument("--source", type=str, required=True,
                        help="Input source: image path, video path, or camera ID (e.g., 0)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="Input image size (default: 480)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device ID or 'cpu' (default: 0)")
    parser.add_argument("--project", type=str, default="runs/predict",
                        help="Root directory for saving results (default: runs/predict)")
    parser.add_argument("--name", type=str, default="result",
                        help="Subdirectory name for this run (default: result)")
    parser.add_argument("--show", action="store_true",
                        help="Display real-time detection window")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save output video/image with detections")
    parser.add_argument("--no-save", action="store_false", dest="save",
                        help="Do not save output")
    parser.add_argument("--line-width", type=int, default=2,
                        help="Bounding box line thickness (default: 2)")
    parser.add_argument("--show-conf", action="store_true", default=True,
                        help="Display confidence score on boxes")
    parser.add_argument("--no-show-conf", action="store_false", dest="show_conf",
                        help="Hide confidence scores")
    parser.add_argument("--show-labels", action="store_true", default=True,
                        help="Display class labels on boxes")
    parser.add_argument("--no-show-labels", action="store_false", dest="show_labels",
                        help="Hide class labels")
    args = parser.parse_args()

    # Load the trained model. --model must point to a YOLO checkpoint (.pt).
    model = YOLO(args.model)

    # Run inference. --source accepts an image path, video path, or camera ID.
    # The confidence threshold (--conf) filters out weak detections;
    # the IoU threshold (--iou) controls overlap removal via NMS.
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        show=args.show,
        save=args.save,
        project=args.project,
        name=args.name,
        line_width=args.line_width,
        show_conf=args.show_conf,
        show_labels=args.show_labels,
    )

    print(f"Inference completed! Results saved to: {args.project}/{args.name}")


if __name__ == "__main__":
    predict()

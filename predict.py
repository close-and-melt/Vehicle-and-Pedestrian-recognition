"""
Prediction script for YOLO vehicle and pedestrian detection model.

Performs object detection on video files using a trained YOLO model.
Supports Pedestrian and car detection with adjustable confidence threshold.
"""

from ultralytics import YOLO


def predict_video() -> None:
    """
    Run inference on a video file using the trained YOLO model.

    Configuration:
        - Model: Best checkpoint from training (vehicle_pedestrian_v13)
        - Input source: Video file (test3.mp4)
        - Confidence threshold: 0.4 (minimum confidence to show detection)
        - IOU threshold: 0.45 (Non-Maximum Suppression overlap threshold)
        - Image size: 480 (must match training image size)
        - Device: GPU 0 (CUDA for faster inference)
        - Output: Display results in real-time window + save video file
    """
    # Load the trained YOLO model (best checkpoint from training)
    model = YOLO(r"E:\YOLO_training\runs\train\vehicle_pedestrian_v13\weights\best.pt")

    # Run prediction on the video file
    results = model.predict(
        source=r"D:\Download\test3.mp4",  # Input video path
        conf=0.4,  # Confidence threshold (boxes below this are filtered)
        iou=0.45,  # NMS IoU threshold (reduces duplicate detections)
        imgsz=480,  # Input image size (must match training size)
        device=0,  # GPU device ID (0 = first GPU, use 'cpu' for CPU)
        show=True,  # Display real-time detection window
        save=True,  # Save output video with detections
        project=r"E:\YOLO_training\runs\predict",  # Root directory for saving results
        name="test_result",  # Subdirectory name for this run
        line_width=2,  # Bounding box line thickness in pixels
        show_conf=True,  # Display confidence score on boxes
        show_labels=True,  # Display class labels (Pedestrian/car) on boxes
    )

    print("Inference completed!")
    print(r"Output video saved at: E:\YOLO_training\runs\predict\test_result")


if __name__ == "__main__":
    predict_video()
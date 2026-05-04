"""
Resume training script for YOLO vehicle and pedestrian detection model.

Loads a previously interrupted training checkpoint (last.pt) and continues
training from where it left off, automatically restoring all parameters
including epochs, image size, batch size, and optimizer state.
"""

from ultralytics import YOLO

if __name__ == '__main__':
    # Load the last checkpoint from a previous training session
    # Note: The path must point to a specific .pt file (last.pt or best.pt)
    # last.pt contains the complete training state including:
    #   - Model weights
    #   - Optimizer state
    #   - Current epoch number
    #   - Best metrics so far
    model = YOLO(
        r"E:\Administrator\three\Artificial_Intelligence\Vehicle-Pedestrian\runs\detect\improve_person_v1\yolo11s_640_res\weights\last.pt")

    # Resume training from the checkpoint
    # Setting resume=True automatically restores:
    #   - Original training configuration (imgsz, batch, epochs, etc.)
    #   - Current epoch (continues from where it stopped)
    #   - Optimizer state (momentum, learning rate schedule, etc.)
    #   - Best model tracking
    model.train(resume=True)
"""
Resume training script for YOLO vehicle and pedestrian detection model.

Loads a previously interrupted training checkpoint (last.pt) and continues
training from where it left off, automatically restoring all parameters.
"""

import argparse
from ultralytics import YOLO


def resume() -> None:
    """
    Resume training from a previously interrupted checkpoint.

    The checkpoint (last.pt) contains the complete training state including:
        - Model weights
        - Optimizer state
        - Current epoch number
        - Best metrics so far

    Calling train(resume=True) automatically restores the original training
    configuration and continues from where it stopped.
    """
    parser = argparse.ArgumentParser(description="Resume YOLO Training")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to checkpoint file (last.pt or best.pt)")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(resume=True)


if __name__ == "__main__":
    resume()

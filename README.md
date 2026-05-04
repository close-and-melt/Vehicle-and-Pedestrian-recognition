# Vehicle and Pedestrian Recognition

Vehicle and Pedestrian Recognition (Object Detection)

> 🔍 Chinese Version：[README.zh.md](./README.zh.md)

## Project Introduction

This project implements real-time and accurate object detection of vehicles and pedestrians in road scenarios based on the **YOLOv11** deep learning model. It can be applied to intelligent traffic monitoring, autonomous driving assistance systems, security patrols, and other scenarios.

## Features

- Based on the YOLOv11 architecture, achieving a good balance between speed and accuracy
- Supports detection and recognition of two target categories: **vehicles** and **pedestrians**
- Provides complete scripts for training, validation, prediction, and resuming training
- Supports images, video files, and real-time camera streams as input sources

## Environment Setup

This project uses Conda for virtual environment management. Ensure you have installed [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

1. **Create and activate a Conda environment** (Python 3.8 or higher):
   ```bash
   conda create -n yolo11 python=3.8
   conda activate yolo11
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` has not been generated yet, you can manually install the core dependencies:
   ```bash
   pip install ultralytics opencv-python torch torchvision
   ```

## Dataset

The dataset used in this project is sourced from Roboflow Universe:
- **Dataset Name**: Vehicle Pedestrian Dataset
- **Source**: [https://universe.roboflow.com/houses-bt9jv/vehicle-pedestrian/dataset/](https://universe.roboflow.com/houses-bt9jv/vehicle-pedestrian/dataset/)
- **Description**: This dataset contains annotated images of vehicles and pedestrians. Please comply with Roboflow's terms of use when using it.

> **Note**: If you cannot access the above link directly, you may need to log in to your Roboflow account or apply for dataset access permissions. Follow the prompts on the page to complete verification.

After downloading the dataset, it is recommended to organize it in your project root directory according to the following structure:
```
Vehicle-and-Pedestrian-recognition/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── train.py
├── predict.py
└── ...
```

## Quick Start

### Train the Model

Start training with default configurations:
```bash
python train.py
```

To customize training parameters (such as image size, number of training epochs, device, etc.), you can directly modify the parameters in `train.py` or pass them via the command line.

### Resume Training

If training is interrupted unexpectedly, you can use `resume_train.py` to resume the last training state:
```bash
python resume_train.py
```

### Model Prediction

Perform prediction on a single image, video file, or real-time camera stream:
```bash
# Predict on an image
python predict.py --source path/to/image.jpg

# Predict on a video
python predict.py --source path/to/video.mp4

# Real-time detection using camera (usually camera device ID is 0)
python predict.py --source 0
```

## Project File Description

| File | Description |
|------|-------------|
| `train.py` | Main script for model training |
| `predict.py` | Detection/prediction script |
| `resume_train.py` | Resume training from checkpoint |
| `filter_dataset.py` | Dataset filtering or preprocessing tool |
| `yolo11n.pt` / `yolo11s.pt` | Pre-trained weight files for YOLOv11 (need to download manually) |
| `requirements.txt` | List of Python dependencies |
| `runs/` | Training logs, weight files, and visualization results (ignored by `.gitignore`, not committed to the repository) |

## Model Weights and Dataset Notes

- **Pre-trained Weights**: YOLOv11's `yolo11n.pt` and `yolo11s.pt` can be automatically downloaded when running training or prediction for the first time, or manually obtained from the [Ultralytics official repository](https://github.com/ultralytics/ultralytics).
- **Training Outputs**: All trained weights, batch images, confusion matrices, etc., are saved in the `runs/detect/` directory. Due to their large size, these files are **not tracked by Git** (configured in `.gitignore`). If you need to share the trained model, use GitHub Releases or cloud storage services.
- **Dataset**: Please download it from Roboflow yourself and place it according to the directory structure above. The original data is not included in the Git repository.

## Frequently Asked Questions

1. **Error about email privacy when pushing to GitHub?**  
   Refer to GitHub documentation to set up a `noreply` email address, and use `git commit --amend --reset-author` to correct the commit author information.

2. **CUDA out of memory error during training/prediction?**  
   Try reducing the `batch-size` or `imgsz` parameters, or run on CPU (set `device='cpu'`).

3. **How to get more information about YOLOv11?**  
   Visit the [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The object detection framework provided by [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- The dataset platform provided by [Roboflow Universe](https://universe.roboflow.com/)
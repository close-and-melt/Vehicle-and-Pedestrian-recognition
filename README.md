# Vehicle and Pedestrian Recognition

车辆与行人识别（目标检测）| Vehicle and Pedestrian recognition (Object Detection)

## 项目简介

本项目基于 **YOLOv11** 深度学习模型，实现对道路场景中的车辆和行人进行实时、准确的目标检测。可用于智能交通监控、自动驾驶辅助系统、安防巡检等场景。

## 功能特点

- 基于 YOLOv11 架构，在速度和精度之间取得良好平衡
- 支持对**车辆**和**行人**两类目标的检测与识别
- 提供完整的训练、验证、预测和恢复训练脚本
- 支持图片、视频文件以及实时摄像头流作为输入源

## 环境配置

本项目使用 Conda 管理虚拟环境。请确保已安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/)。

1. **创建并激活 Conda 环境**（Python 3.8 或更高版本）：
   ```bash
   conda create -n yolo11 python=3.8
   conda activate yolo11
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

   如果尚未生成 `requirements.txt`，可以手动安装核心依赖：
   ```bash
   pip install ultralytics opencv-python torch torchvision
   ```

## 数据集

本项目使用的数据集来源于 Roboflow Universe：
- **数据集名称**：Vehicle Pedestrian Dataset
- **来源**：[https://universe.roboflow.com/houses-bt9jv/vehicle-pedestrian/dataset/](https://universe.roboflow.com/houses-bt9jv/vehicle-pedestrian/dataset/)
- **说明**：该数据集包含标注的车辆和行人图像。使用时请遵守 Roboflow 的使用条款。

> **注意**：如果无法直接访问上述链接，可能需要登录 Roboflow 账户或申请数据集访问权限。请按页面提示完成验证。

将数据集下载后，建议按以下结构组织在你的项目根目录下：
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

## 快速开始

### 训练模型

使用默认配置开始训练：
```bash
python train.py
```

如需自定义训练参数（如图像尺寸、训练轮数、设备等），可直接修改 `train.py` 中的参数，或通过命令行传递。

### 恢复训练

若训练意外中断，可以使用 `resume_train.py` 恢复上一次的训练状态：
```bash
python resume_train.py
```

### 模型预测

对单张图片、视频文件或摄像头实时流进行预测：
```bash
# 对图片进行预测
python predict.py --source path/to/image.jpg

# 对视频进行预测
python predict.py --source path/to/video.mp4

# 使用摄像头实时检测（通常摄像头设备号为 0）
python predict.py --source 0
```

## 项目文件说明

| 文件 | 说明 |
|------|------|
| `train.py` | 模型训练主脚本 |
| `predict.py` | 检测/预测脚本 |
| `resume_train.py` | 从检查点恢复训练 |
| `filter_dataset.py` | 数据集过滤或预处理工具 |
| `yolo11n.pt` / `yolo11s.pt` | YOLOv11 的预训练权重文件（需自行下载） |
| `requirements.txt` | Python 依赖列表 |
| `runs/` | 训练日志、权重文件和可视化结果（被 `.gitignore` 忽略，不提交到仓库） |

## 模型权重与数据集说明

- **预训练权重**：YOLOv11 的 `yolo11n.pt` 和 `yolo11s.pt` 可在首次运行训练或预测时自动下载，或从 [Ultralytics 官方仓库](https://github.com/ultralytics/ultralytics) 手动获取。
- **训练产出**：所有训练得到的权重、批次图片、混淆矩阵等均保存在 `runs/detect/` 目录下。由于文件较大，这些文件**不会被 Git 追踪**（已配置 `.gitignore`）。如需分享训练好的模型，请使用 GitHub Releases 或网盘。
- **数据集**：请自行从 Roboflow 下载，并按照上述目录结构放置。Git 仓库中不包含原始数据。

## 常见问题

1. **推送 GitHub 时提示邮箱隐私错误？**  
   请参考 GitHub 文档设置 `noreply` 邮箱地址，并使用 `git commit --amend --reset-author` 修正提交作者信息。

2. **训练/预测时提示 CUDA out of memory？**  
   可尝试减小 `batch-size` 或 `imgsz` 参数，或在 CPU 上运行（设置 `device='cpu'`）。

3. **如何获取 YOLOv11 的更多信息？**  
   访问 [Ultralytics YOLOv11 文档](https://docs.ultralytics.com/)。

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 致谢

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) 提供的目标检测框架
- [Roboflow Universe](https://universe.roboflow.com/) 提供的数据集平台

from ultralytics import YOLO

def train():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=r"E:\YOLO_training_filtered\data.yaml",
        epochs=50,
        imgsz=480,
        batch=4,          # 防止显存溢出
        device=0,
        workers=2,
        cache="disk",     # 避免内存冲突
        project=r"E:\YOLO_training\runs\train",
        name="vehicle_pedestrian_v1",
        patience=20,
        plots=True,
    )

if __name__ == "__main__":
    train()
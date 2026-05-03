from ultralytics import YOLO

def predict_video():
    model = YOLO(r"E:\YOLO_training\runs\train\vehicle_pedestrian_v13\weights\best.pt")

    results = model.predict(
        source=r"D:\Download\test3.mp4",
        conf=0.4,          # 置信度阈值，低于此值不显示
        iou=0.45,           # NMS重叠阈值
        imgsz=480,          # 与训练时保持一致
        device=0,           # GPU推理
        show=True,          # 实时显示检测窗口
        save=True,          # 保存结果视频
        project=r"E:\YOLO_training\runs\predict",  # 保存目录
        name="test_result",
        line_width=2,       # 检测框线条粗细
        show_conf=True,     # 显示置信度
        show_labels=True,   # 显示类别标签
    )

    print("推理完成！")
    print(r"结果视频保存于: E:\YOLO_training\runs\predict\test_result")

if __name__ == "__main__":
    predict_video()
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 直接加载那个“存档”文件
    # 注意：路径一定要指向具体的 last.pt
    model = YOLO(r"E:\Administrator\three\Artificial_Intelligence\PyCharmMiscProject\runs\detect\improve_person_v1\yolo11s_640_res\weights\last.pt")

    # 2. 开启续传
    # 只要设置 resume=True，YOLO会自动找回之前的 imgsz, batch 和 50轮的目标
    model.train(resume=True)
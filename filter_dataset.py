import os
import shutil
from pathlib import Path

# 原始数据集路径
SRC_ROOT = Path(r"E:\YOLO_training")

# 过滤后新数据集保存路径
DST_ROOT = Path(r"E:\YOLO_training_filtered")

# 原始类别对应索引（来自你的 data.yaml）
# 0=Pedestrian, 1=bike, 2=bus, 3=car, 4=lorry, 5=small_lorry, 6=truck, 7=van
KEEP_CLASSES = {0: 0, 3: 1}  # 原始id -> 新id（0=Pedestrian, 1=car）

SPLITS = ["train", "valid", "test"]

def filter_split(split):
    src_img_dir = SRC_ROOT / split / "images"
    src_lbl_dir = SRC_ROOT / split / "labels"
    dst_img_dir = DST_ROOT / split / "images"
    dst_lbl_dir = DST_ROOT / split / "labels"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0

    for lbl_file in src_lbl_dir.glob("*.txt"):
        new_lines = []

        with open(lbl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                if cls_id in KEEP_CLASSES:
                    # 替换为新的类别id
                    new_cls = KEEP_CLASSES[cls_id]
                    new_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        # 只保留含有目标类别的图片
        if new_lines:
            # 复制图片
            for ext in [".jpg", ".jpeg", ".png"]:
                img_file = src_img_dir / (lbl_file.stem + ext)
                if img_file.exists():
                    shutil.copy(img_file, dst_img_dir / img_file.name)
                    break

            # 写入新标签
            with open(dst_lbl_dir / lbl_file.name, "w") as f:
                f.write("\n".join(new_lines))

            kept += 1
        else:
            skipped += 1

    print(f"[{split}] 保留: {kept} 张，跳过: {skipped} 张")

if __name__ == "__main__":
    for split in SPLITS:
        filter_split(split)

    # 自动生成新的 data.yaml
    yaml_content = f"""train: {DST_ROOT / "train" / "images"}
val: {DST_ROOT / "valid" / "images"}
test: {DST_ROOT / "test" / "images"}
nc: 2
names: ['Pedestrian', 'car']
"""
    with open(DST_ROOT / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n✅ 过滤完成！新数据集保存于: {DST_ROOT}")
    print(f"新 data.yaml 已生成: {DST_ROOT / 'data.yaml'}")
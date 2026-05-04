"""
Dataset filtering script for YOLO format datasets.
Keeps only specified classes (Pedestrian and car) and remaps their IDs.

Original dataset structure:
    E:/YOLO_training/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    └── test/

Filtered dataset will be saved to:
    E:/YOLO_training_filtered/
"""

import os
import shutil
from pathlib import Path

# ==================== Configuration ====================
# Original dataset path
SRC_ROOT = Path(r"E:\YOLO_training")

# Output path for filtered dataset
DST_ROOT = Path(r"E:\YOLO_training_filtered")

# Original class mapping (from data.yaml)
# 0=Pedestrian, 1=bike, 2=bus, 3=car, 4=lorry, 5=small_lorry, 6=truck, 7=van
# Keep only Pedestrian (0) and car (3), remapping to new IDs: 0 and 1
KEEP_CLASSES = {0: 0, 3: 1}  # original_id: new_id

# Dataset splits to process
SPLITS = ["train", "valid", "test"]


# =======================================================


def filter_split(split: str) -> None:
    """
    Filter a single dataset split (train/valid/test).

    Args:
        split: Name of the split to process (e.g., 'train', 'valid', 'test')

    Returns:
        None. Creates filtered images and labels in DST_ROOT/split/
    """
    src_img_dir = SRC_ROOT / split / "images"
    src_lbl_dir = SRC_ROOT / split / "labels"
    dst_img_dir = DST_ROOT / split / "images"
    dst_lbl_dir = DST_ROOT / split / "labels"

    # Create destination directories
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    # Iterate through all label files
    for lbl_file in src_lbl_dir.glob("*.txt"):
        new_lines = []

        # Read and filter annotations
        with open(lbl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])

                # Keep only desired classes
                if cls_id in KEEP_CLASSES:
                    new_cls = KEEP_CLASSES[cls_id]  # Remap to new ID
                    new_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        # If at least one valid annotation exists, copy the corresponding image
        if new_lines:
            # Try common image extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                img_file = src_img_dir / (lbl_file.stem + ext)
                if img_file.exists():
                    shutil.copy(img_file, dst_img_dir / img_file.name)
                    break

            # Write filtered labels
            with open(dst_lbl_dir / lbl_file.name, "w") as f:
                f.write("\n".join(new_lines))

            kept += 1
        else:
            skipped += 1

    print(f"[{split}] Kept: {kept} images, Skipped: {skipped} images")


def generate_data_yaml() -> None:
    """Generate a new data.yaml file for the filtered dataset."""
    yaml_content = f"""train: {DST_ROOT / "train" / "images"}
val: {DST_ROOT / "valid" / "images"}
test: {DST_ROOT / "test" / "images"}
nc: 2
names: ['Pedestrian', 'car']
"""
    with open(DST_ROOT / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"✅ New data.yaml generated at: {DST_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    # Process each dataset split
    for split in SPLITS:
        filter_split(split)

    # Generate updated data.yaml
    generate_data_yaml()

    print(f"\n✅ Filtering completed! Filtered dataset saved to: {DST_ROOT}")
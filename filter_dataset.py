"""
Dataset filtering script for YOLO format datasets.
Keeps only specified classes (Pedestrian and car) and remaps their IDs.

Original dataset structure:
    <src_root>/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    └── test/

Filtered dataset will be saved to:
    <dst_root>/
"""

import argparse
import os
import shutil
from pathlib import Path

# Original class mapping (from the source dataset):
# 0=Pedestrian, 1=bike, 2=bus, 3=car, 4=lorry, 5=small_lorry, 6=truck, 7=van
# Keep only Pedestrian (0) and car (3), remapping to new IDs: 0 and 1
DEFAULT_KEEP = {0: 0, 3: 1}
DEFAULT_SPLITS = ["train", "valid", "test"]


def filter_split(src_root: Path, dst_root: Path, split: str,
                 keep_classes: dict[int, int]) -> None:
    """
    Filter a single dataset split (train/valid/test).

    Args:
        src_root: Root directory of the original dataset.
        dst_root: Root directory where the filtered dataset will be saved.
        split: Name of the split to process (e.g., 'train', 'valid', 'test').
        keep_classes: Dict mapping original class IDs to new class IDs.

    Returns:
        None. Creates filtered images and labels in dst_root/<split>/.
    """
    src_img_dir = src_root / split / "images"
    src_lbl_dir = src_root / split / "labels"
    dst_img_dir = dst_root / split / "images"
    dst_lbl_dir = dst_root / split / "labels"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    # Iterate through all label files and remap class IDs
    for lbl_file in src_lbl_dir.glob("*.txt"):
        new_lines = []
        with open(lbl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                # Keep only classes in the mapping, remapping to new IDs
                if cls_id in keep_classes:
                    new_cls = keep_classes[cls_id]
                    new_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        # Only copy the image if at least one target annotation was kept
        if new_lines:
            for ext in [".jpg", ".jpeg", ".png"]:
                img_file = src_img_dir / (lbl_file.stem + ext)
                if img_file.exists():
                    shutil.copy(img_file, dst_img_dir / img_file.name)
                    break
            with open(dst_lbl_dir / lbl_file.name, "w") as f:
                f.write("\n".join(new_lines))
            kept += 1
        else:
            skipped += 1

    print(f"[{split}] Kept: {kept} images, Skipped: {skipped} images")


def generate_data_yaml(dst_root: Path, keep_classes: dict[int, int]) -> None:
    """Generate data.yaml for the filtered dataset."""
    class_names = ["Pedestrian", "car"]  # ordered by new class IDs
    yaml_content = (
        f"train: {dst_root / 'train' / 'images'}\n"
        f"val: {dst_root / 'valid' / 'images'}\n"
        f"test: {dst_root / 'test' / 'images'}\n"
        f"nc: {len(keep_classes)}\n"
        f"names: {class_names}\n"
    )
    yaml_path = dst_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"data.yaml generated at: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter YOLO Dataset")
    parser.add_argument("--src", type=str, required=True,
                        help="Source dataset root directory")
    parser.add_argument("--dst", type=str, required=True,
                        help="Destination directory for filtered dataset")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=DEFAULT_SPLITS,
                        help=f"Dataset splits to process (default: {DEFAULT_SPLITS})")
    parser.add_argument("--keep", type=str, nargs="+",
                        help="Classes to keep: 'original_id:new_id' pairs "
                             "(default: '0:0 3:1')")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    keep_classes = DEFAULT_KEEP
    if args.keep:
        keep_classes = {}
        for item in args.keep:
            orig, new = item.split(":")
            keep_classes[int(orig)] = int(new)

    for split in args.splits:
        filter_split(src_root, dst_root, split, keep_classes)

    generate_data_yaml(dst_root, keep_classes)
    print(f"Filtering completed! Filtered dataset saved to: {dst_root}")


if __name__ == "__main__":
    main()

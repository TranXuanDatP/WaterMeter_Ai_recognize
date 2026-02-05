"""
M1 Utility Functions

Helper functions for data augmentation, preprocessing, and dataset management.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional
import shutil


logger = logging.getLogger(__name__)


def create_dataset_yaml(
    dataset_path: str,
    train_images: str,
    val_images: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Create YOLO dataset.yaml configuration file.

    Args:
        dataset_path: Root path to dataset directory
        train_images: Path to training images (relative to dataset_path)
        val_images: Path to validation images (relative to dataset_path)
        output_path: Output path for yaml (defaults to dataset_path/dataset.yaml)

    Returns:
        Path to created yaml file
    """
    if output_path is None:
        output_path = os.path.join(dataset_path, "dataset.yaml")

    yaml_content = f"""# M1 Watermeter Detection Dataset Configuration
path: {os.path.abspath(dataset_path)}
train: {train_images}
val: {val_images}

nc: 1  # number of classes
names: ['watermeter']  # class names
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content)

    logger.info(f"Created dataset.yaml at: {output_path}")
    return output_path


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Split dataset into train and validation sets.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files (.txt)
        output_dir: Output directory for split dataset
        train_ratio: Fraction of data for training (default: 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dir, val_dir) paths
    """
    import random
    from pathlib import Path

    random.seed(seed)

    # Get all image files
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    image_files = list(images_path.glob("*.jpg")) + \
                  list(images_path.glob("*.png"))

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    logger.info(f"Found {len(image_files)} images")

    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)

    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create output directories
    train_img_dir = Path(output_dir) / "images" / "train"
    val_img_dir = Path(output_dir) / "images" / "val"
    train_lbl_dir = Path(output_dir) / "labels" / "train"
    val_lbl_dir = Path(output_dir) / "labels" / "val"

    for dir_path in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_files(files, img_dir, lbl_dir):
        for img_path in files:
            # Copy image
            shutil.copy(img_path, img_dir / img_path.name)

            # Copy label if exists
            label_path = labels_path / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, lbl_dir / label_path.name)
            else:
                logger.warning(f"Label not found: {label_path}")

    logger.info("Copying training files...")
    copy_files(train_files, train_img_dir, train_lbl_dir)

    logger.info("Copying validation files...")
    copy_files(val_files, val_img_dir, val_lbl_dir)

    logger.info(f"Dataset split complete: {output_dir}")
    return str(train_img_dir), str(val_img_dir)


def verify_yolo_annotations(
    labels_dir: str,
    num_classes: int = 1,
    check_values: bool = True,
) -> dict:
    """
    Verify YOLO format annotations.

    Args:
        labels_dir: Directory containing .txt label files
        num_classes: Expected number of classes
        check_values: Check if values are in valid range [0, 1]

    Returns:
        Dictionary with verification statistics
    """
    from pathlib import Path

    labels_path = Path(labels_dir)
    label_files = list(labels_path.glob("*.txt"))

    if not label_files:
        raise ValueError(f"No label files found in {labels_dir}")

    stats = {
        "total_files": len(label_files),
        "valid_files": 0,
        "invalid_files": 0,
        "empty_files": 0,
        "total_boxes": 0,
        "errors": [],
    }

    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                stats["empty_files"] += 1
                continue

            valid = True
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    stats["errors"].append(
                        f"{label_file.name}: Invalid format (expected 5 values)"
                    )
                    valid = False
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Check class ID
                if class_id < 0 or class_id >= num_classes:
                    stats["errors"].append(
                        f"{label_file.name}: Invalid class ID {class_id}"
                    )
                    valid = False

                # Check value ranges
                if check_values:
                    for name, val in [
                        ("x_center", x_center),
                        ("y_center", y_center),
                        ("width", width),
                        ("height", height),
                    ]:
                        if val < 0 or val > 1:
                            stats["errors"].append(
                                f"{label_file.name}: {name}={val} not in [0, 1]"
                            )
                            valid = False

                stats["total_boxes"] += 1

            if valid:
                stats["valid_files"] += 1
            else:
                stats["invalid_files"] += 1

        except Exception as e:
            stats["errors"].append(f"{label_file.name}: {str(e)}")
            stats["invalid_files"] += 1

    # Log results
    logger.info(f"Verification complete:")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Valid: {stats['valid_files']}")
    logger.info(f"  Invalid: {stats['invalid_files']}")
    logger.info(f"  Empty: {stats['empty_files']}")
    logger.info(f"  Total boxes: {stats['total_boxes']}")

    if stats["errors"]:
        logger.warning(f"Found {len(stats['errors'])} errors:")
        for error in stats["errors"][:10]:  # Show first 10
            logger.warning(f"  - {error}")
        if len(stats["errors"]) > 10:
            logger.warning(f"  ... and {len(stats['errors']) - 10} more")

    return stats


def convert_bbox_to_yolo(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from absolute to YOLO format.

    Args:
        x_min, y_min, x_max, y_max: Bounding box coordinates (pixels)
        img_width, img_height: Image dimensions

    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return x_center, y_center, width, height


def convert_yolo_to_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from YOLO to absolute format.

    Args:
        x_center, y_center, width, height: Normalized YOLO coordinates [0, 1]
        img_width, img_height: Image dimensions

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in pixels
    """
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    x_min = int(x_center_abs - width_abs / 2)
    y_min = int(y_center_abs - height_abs / 2)
    x_max = int(x_center_abs + width_abs / 2)
    y_max = int(y_center_abs + height_abs / 2)

    return x_min, y_min, x_max, y_max

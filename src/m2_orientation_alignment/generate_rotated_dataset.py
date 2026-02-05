"""
Generate Rotated Training Dataset from Upright Ground Truth

Takes auto-aligned upright images (0°) and applies random rotations
to create labeled training data for M2 orientation model.
"""

import os
import csv
import random
import logging
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_rotated_dataset(
    upright_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    image_size: int = 640,
):
    """
    Generate rotated training dataset from upright ground truth.

    Args:
        upright_dir: Directory with auto-aligned upright images
        output_dir: Output directory for rotated dataset
        train_split: Fraction for training (rest for validation)
        image_size: Image size
    """
    upright_path = Path(upright_dir)
    output_path = Path(output_dir)

    # Create output directories
    train_img_dir = output_path / "train" / "images"
    val_img_dir = output_path / "val" / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    # Get all upright images (both train and val)
    upright_images = []

    train_labels_file = upright_path / "train" / "labels.csv"
    val_labels_file = upright_path / "val" / "labels.csv"

    # Load train images
    if train_labels_file.exists():
        with open(train_labels_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    img_name = parts[0]
                    upright_images.append((upright_path / "train" / "images" / img_name, True))

    # Load val images
    if val_labels_file.exists():
        with open(val_labels_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    img_name = parts[0]
                    upright_images.append((upright_path / "val" / "images" / img_name, False))

    logger.info(f"Found {len(upright_images)} upright images")

    if not upright_images:
        logger.error(f"No upright images found in {upright_dir}")
        return

    # Generate rotated versions
    train_samples = []
    val_samples = []
    sample_count = 0

    for img_path, _ in tqdm(upright_images, desc="Processing images"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Generate random rotation angle
        angle = random.uniform(0, 360)

        # Rotate image
        center = (image_size // 2, image_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image_size, image_size),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)

        # Determine split (train/val)
        is_train = random.random() < train_split

        # Generate filename
        output_name = f"rot_{sample_count:06d}.jpg"
        sample_count += 1

        # Save image
        if is_train:
            output_path_full = train_img_dir / output_name
            train_samples.append((output_name, angle))
        else:
            output_path_full = val_img_dir / output_name
            val_samples.append((output_name, angle))

        cv2.imwrite(str(output_path_full), rotated)

    # Save label files
    train_labels = output_path / "train" / "labels.csv"
    val_labels = output_path / "val" / "labels.csv"

    with open(train_labels, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'angle'])
        writer.writerows(train_samples)

    with open(val_labels, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'angle'])
        writer.writerows(val_samples)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Rotated dataset generated successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"Upright images processed: {len(upright_images)}")
    logger.info(f"Total rotated samples: {sample_count}")
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    logger.info(f"\nOutput: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate rotated training dataset from upright images")
    parser.add_argument("--upright", type=str, required=True,
                       help="Directory with auto-aligned upright images")
    parser.add_argument("--output", type=str, default="data/m2_rotated",
                       help="Output directory for rotated dataset")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training split fraction (default: 0.8)")
    parser.add_argument("--size", type=int, default=640,
                       help="Image size (default: 640)")

    args = parser.parse_args()

    generate_rotated_dataset(
        upright_dir=args.upright,
        output_dir=args.output,
        train_split=args.train_split,
        image_size=args.size,
    )

"""
Generate M2 Orientation Training Dataset

Create rotation-labeled dataset from existing watermeter images.
"""

import os
import csv
import random
import logging
from pathlib import Path
from tqdm import tqdm

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    raise ImportError(f"Required dependencies: {e}\nInstall: pip install opencv-python pillow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_orientation_dataset(
    source_images_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    num_rotations_per_image: int = 5,
    image_size: int = 640,
):
    """
    Generate orientation training dataset.

    Args:
        source_images_dir: Directory with source watermeter images
        output_dir: Output directory for generated dataset
        train_split: Fraction for training (rest for validation)
        num_rotations_per_image: Number of rotated versions per source image
        image_size: Output image size (square)
    """
    source_dir = Path(source_images_dir)
    output_path = Path(output_dir)

    # Create output directories
    train_img_dir = output_path / "train" / "images"
    val_img_dir = output_path / "val" / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    # Get source images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(source_dir.glob(ext))
        image_files.extend(source_dir.glob(f'**/{ext}'))

    logger.info(f"Found {len(image_files)} source images")

    if not image_files:
        logger.error(f"No images found in {source_images_dir}")
        return

    # Generate dataset
    train_samples = []
    val_samples = []
    sample_count = 0

    for img_path in tqdm(image_files, desc="Processing images"):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to load: {img_path}")
            continue

        # Resize to target size
        h, w = img.shape[:2]
        scale = image_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Center crop to square
        h, w = img.shape[:2]
        if h > image_size:
            y = (h - image_size) // 2
            img = img[y:y+image_size, :]
        if w > image_size:
            x = (w - image_size) // 2
            img = img[:, x:x+image_size]

        # Ensure exact size
        img = cv2.resize(img, (image_size, image_size))

        # Generate rotated versions
        for _ in range(num_rotations_per_image):
            # Random rotation angle [0, 360)
            angle = random.uniform(0, 360)

            # Rotate image
            center = (image_size // 2, image_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (image_size, image_size),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)

            # Determine split (train/val)
            is_train = random.random() < train_split

            # Generate filename
            output_name = f"orient_{sample_count:06d}.jpg"
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

    logger.info(f"\n✅ Dataset generated successfully!")
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"\nLabel files:")
    logger.info(f"  Train: {train_labels}")
    logger.info(f"  Val: {val_labels}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate M2 orientation dataset")
    parser.add_argument("--source", type=str, required=True,
                       help="Source images directory")
    parser.add_argument("--output", type=str, default="data/m2_orientation",
                       help="Output directory")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training split fraction (default: 0.8)")
    parser.add_argument("--rotations", type=int, default=5,
                       help="Rotations per image (default: 5)")
    parser.add_argument("--size", type=int, default=640,
                       help="Image size (default: 640)")

    args = parser.parse_args()

    generate_orientation_dataset(
        source_images_dir=args.source,
        output_dir=args.output,
        train_split=args.train_split,
        num_rotations_per_image=args.rotations,
        image_size=args.size,
    )

"""
Auto-align Watermeter Crops to Upright

Part 2 of the M2 dataset pipeline:
Takes M1-cropped watermeters and auto-aligns them to upright orientation using CV heuristics.
These aligned crops become ground truth (0° labels) for M2 training.
"""

import os
import csv
import logging
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_dominant_orientation(image: np.ndarray) -> float:
    """
    Detect dominant orientation using edge/line detection heuristics.

    For watermeters: upright = text is horizontal, numbers read left-to-right
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray_blurred, 50, 150, apertureSize=3)

    # Use probabilistic Hough transform to detect line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=60,
        maxLineGap=10
    )

    if lines is not None and len(lines) > 0:
        # Analyze line orientations
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1

            # Calculate angle in degrees
            angle = np.degrees(np.arctan2(dy, dx))

            # Normalize to [0, 180)
            if angle < 0:
                angle += 180

            # Filter for near-horizontal and near-vertical lines
            if abs(angle) < 45 or abs(angle - 90) < 45 or abs(angle - 180) < 45:
                angles.append(angle)

        if angles:
            # Find most common orientation using histogram
            hist, bins = np.histogram(angles, bins=36, range=(0, 180))
            dominant_angle = bins[np.argmax(hist)]

            # Normalize angle: if > 90, assume it's vertical and adjust
            if dominant_angle > 90:
                dominant_angle -= 90

            # If angle > 45, normalize to [-45, 45]
            if dominant_angle > 45:
                dominant_angle -= 90

            return dominant_angle

    # Default: assume 0° (already upright)
    return 0.0


def auto_align_crops(
    crops_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    image_size: int = 640,
):
    """
    Auto-align cropped watermeters to upright orientation.

    Args:
        crops_dir: Directory with M1-cropped watermeters
        output_dir: Output directory for aligned images (train/val split)
        train_split: Fraction for training (rest for validation)
        image_size: Image size
    """
    crops_path = Path(crops_dir)
    output_path = Path(output_dir)

    # Create output directories
    train_img_dir = output_path / "train" / "images"
    val_img_dir = output_path / "val" / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    # Get all crop files
    crop_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        crop_files.extend(crops_path.glob(ext))

    logger.info(f"Found {len(crop_files)} crop files")

    if not crop_files:
        logger.error(f"No crop files found in {crops_dir}")
        return

    # Process crops
    train_samples = []
    val_samples = []
    aligned_count = 0

    for crop_path in tqdm(crop_files, desc="Aligning crops"):
        # Read crop
        image = cv2.imread(str(crop_path))
        if image is None:
            continue

        # Detect orientation
        angle = detect_dominant_orientation(image)

        # Rotate if needed
        if abs(angle) > 1:
            center = (image_size // 2, image_size // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            image_aligned = cv2.warpAffine(image, M, (image_size, image_size),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT)
            aligned_count += 1
        else:
            image_aligned = image

        # Determine split (using hash for reproducibility)
        import hashlib
        is_train = int(hashlib.md5(str(crop_path).encode()).hexdigest(), 16) % 100 < (train_split * 100)

        # Generate filename
        output_name = f"aligned_{crop_path.name}"
        if is_train:
            output_path_full = train_img_dir / output_name
            train_samples.append((output_name, 0.0))
        else:
            output_path_full = val_img_dir / output_name
            val_samples.append((output_name, 0.0))

        cv2.imwrite(str(output_path_full), image_aligned)

    # Save label files (all 0° since all are aligned)
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
    logger.info(f"✅ Auto-alignment complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Crops processed: {len(crop_files)}")
    logger.info(f"Crops aligned: {aligned_count}")
    logger.info(f"Crops already upright: {len(crop_files) - aligned_count}")
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    logger.info(f"\nOutput: {output_path}")
    logger.info(f"\nAll samples are upright (0°) - ground truth for M2 training")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-align cropped watermeters to upright")
    parser.add_argument("--crops", type=str, required=True,
                       help="Directory with M1-cropped watermeters")
    parser.add_argument("--output", type=str, default="data/m2_upright_gt",
                       help="Output directory for aligned images")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training split fraction (default: 0.8)")
    parser.add_argument("--size", type=int, default=640,
                       help="Image size (default: 640)")

    args = parser.parse_args()

    auto_align_crops(
        crops_dir=args.crops,
        output_dir=args.output,
        train_split=args.train_split,
        image_size=args.size,
    )

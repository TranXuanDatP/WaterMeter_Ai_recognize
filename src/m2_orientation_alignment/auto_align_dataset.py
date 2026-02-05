"""
Auto-align images to upright orientation using CV heuristics

Detects dominant orientation of watermeter display and rotates to upright.
"""

import os
import csv
import logging
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

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
            # These correspond to text lines and digit boundaries
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


def auto_align_images(
    source_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    max_images: int = None,
):
    """
    Auto-align images to upright orientation.

    Args:
        source_dir: Source images (mixed orientations)
        output_dir: Output directory for aligned images
        train_split: Train/val split
        max_images: Maximum number of images to process (None = all)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create output directories
    train_dir = output_path / "train" / "images"
    val_dir = output_path / "val" / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(source_path.glob(ext))
        image_files.extend(source_path.glob(f'**/{ext}'))

    # Limit images if specified
    if max_images:
        image_files = image_files[:max_images]

    logger.info(f"Found {len(image_files)} images")

    if not image_files:
        logger.error(f"No images found in {source_dir}")
        return

    train_samples = []
    val_samples = []

    # Process images
    for img_path in tqdm(image_files, desc="Aligning images"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Resize to 640x640
        h, w = image.shape[:2]
        scale = 640 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

        # Center crop to 640x640
        h, w = image.shape[:2]
        if h > 640:
            y = (h - 640) // 2
            image = image[y:y+640, :, :]
        if w > 640:
            x = (w - 640) // 2
            image = image[:, x:x+640, :]

        # Ensure exact size
        image = cv2.resize(image, (640, 640))

        # Detect orientation
        angle = detect_dominant_orientation(image)

        # Rotate to upright
        if abs(angle) > 1:  # Only rotate if angle > 1 degree
            center = (320, 320)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            image_aligned = cv2.warpAffine(image, M, (640, 640), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            image_aligned = image

        # Determine split
        is_train = hash(str(img_path)) % 100 < (train_split * 100)

        # Save image
        output_name = f"aligned_{img_path.stem}.jpg"
        if is_train:
            output_path_full = train_dir / output_name
            train_samples.append((output_name, 0.0))  # Aligned = 0°
        else:
            output_path_full = val_dir / output_name
            val_samples.append((output_name, 0.0))

        cv2.imwrite(str(output_path_full), image_aligned)

    # Save labels (all 0° since aligned)
    train_labels = output_path / "train" / "labels.csv"
    val_labels = output_path / "val" / "labels.csv"

    with open(train_labels, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'angle'])
        for name, angle in train_samples:
            writer.writerow([name, angle])

    with open(val_labels, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'angle'])
        for name, angle in val_samples:
            writer.writerow([name, angle])

    logger.info(f"\n✅ Auto-aligned dataset created!")
    logger.info(f"Source images: {len(image_files)}")
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    logger.info(f"\nOutput: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-align watermeter images")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/m2_aligned")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to process (for testing)")

    args = parser.parse_args()

    auto_align_images(
        source_dir=args.source,
        output_dir=args.output,
        train_split=args.train_split,
        max_images=args.max_images,
    )

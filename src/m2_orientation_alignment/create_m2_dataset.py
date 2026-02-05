"""
Create M2 Training Dataset from M1 Detections

Pipeline:
1. Run M1 detection on source images
2. Crop detected watermeters
3. Treat crops as "upright ground truth"
4. Apply random rotations to create training samples
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


def create_m2_dataset_from_m1(
    source_images_dir: str,
    m1_model_path: str,
    output_dir: str,
    train_split: float = 0.8,
    num_rotations_per_crop: int = 10,
    image_size: int = 640,
    confidence_threshold: float = 0.50,
):
    """
    Create M2 orientation dataset using M1 detections.

    Pipeline:
    1. Detect watermeters with M1
    2. Crop detections (treat as "upright" ground truth)
    3. Apply random rotations to create labeled training data

    Args:
        source_images_dir: Directory with source full images
        m1_model_path: Path to trained M1 model
        output_dir: Output directory for M2 dataset
        train_split: Fraction for training (rest for validation)
        num_rotations_per_crop: Number of rotated versions per crop
        image_size: Output image size (square)
        confidence_threshold: M1 detection confidence threshold
    """
    from src.m1_watermeter_detection import M1Inference

    source_dir = Path(source_images_dir)
    output_path = Path(output_dir)

    # Create output directories
    train_img_dir = output_path / "train" / "images"
    val_img_dir = output_path / "val" / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    # Load M1 model
    logger.info(f"Loading M1 model from: {m1_model_path}")

    # Auto-detect device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
    except ImportError:
        device = "cpu"

    m1_inference = M1Inference(m1_model_path, confidence_threshold=confidence_threshold, device=device)

    # Get source images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(source_dir.glob(ext))
        image_files.extend(source_dir.glob(f'**/{ext}'))

    logger.info(f"Found {len(image_files)} source images")

    if not image_files:
        logger.error(f"No images found in {source_images_dir}")
        return

    # Process images and generate dataset
    train_samples = []
    val_samples = []
    sample_count = 0
    detection_count = 0
    no_detection_count = 0

    for img_path in tqdm(image_files, desc="Processing images"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Convert BGR to RGB for M1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # Run M1 detection
            result = m1_inference.predict(image_rgb)

            if result['success']:
                detection_count += 1

                # Get cropped meter (M1 output)
                cropped = result['cropped_region']  # Already 640x640

                # Ensure it's in BGR for OpenCV saving
                if cropped.shape[2] == 3:  # RGB
                    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                else:
                    cropped_bgr = cropped

                # Generate rotated versions
                for _ in range(num_rotations_per_crop):
                    # Random rotation angle [0, 360)
                    angle = random.uniform(0, 360)

                    # Rotate image
                    center = (image_size // 2, image_size // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(cropped_bgr, M, (image_size, image_size),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT)

                    # Determine split (train/val)
                    is_train = random.random() < train_split

                    # Generate filename
                    output_name = f"m2_{sample_count:06d}.jpg"
                    sample_count += 1

                    # Save image
                    if is_train:
                        output_path_full = train_img_dir / output_name
                        train_samples.append((output_name, angle))
                    else:
                        output_path_full = val_img_dir / output_name
                        val_samples.append((output_name, angle))

                    cv2.imwrite(str(output_path_full), rotated)

            else:
                no_detection_count += 1

        except Exception as e:
            logger.warning(f"Error processing {img_path.name}: {e}")
            continue

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
    logger.info(f"✅ M2 Dataset generated successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"Source images processed: {len(image_files)}")
    logger.info(f"Detections: {detection_count}")
    logger.info(f"No detections: {no_detection_count}")
    logger.info(f"Total samples generated: {sample_count}")
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    logger.info(f"\nOutput directory: {output_path}")
    logger.info(f"\nLabel files:")
    logger.info(f"  Train: {train_labels}")
    logger.info(f"  Val: {val_labels}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create M2 orientation dataset from M1 detections")
    parser.add_argument("--source", type=str, required=True,
                       help="Source images directory")
    parser.add_argument("--m1-model", type=str, required=True,
                       help="Path to M1 model (detect_watermeter.pt)")
    parser.add_argument("--output", type=str, default="data/m2_orientation",
                       help="Output directory for M2 dataset")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training split fraction (default: 0.8)")
    parser.add_argument("--rotations", type=int, default=10,
                       help="Rotations per detection (default: 10)")
    parser.add_argument("--size", type=int, default=640,
                       help="Image size (default: 640)")
    parser.add_argument("--conf-thresh", type=float, default=0.50,
                       help="M1 confidence threshold (default: 0.50)")

    args = parser.parse_args()

    create_m2_dataset_from_m1(
        source_images_dir=args.source,
        m1_model_path=args.m1_model,
        output_dir=args.output,
        train_split=args.train_split,
        num_rotations_per_crop=args.rotations,
        image_size=args.size,
        confidence_threshold=args.conf_thresh,
    )

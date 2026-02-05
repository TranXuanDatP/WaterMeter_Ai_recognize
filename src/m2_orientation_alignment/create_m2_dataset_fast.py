"""
Fast M2 Dataset Creation from M1 Detections

Optimized version with batch processing and parallel execution.
"""

import os
import csv
import random
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_image(args):
    """
    Process a single image - designed for multiprocessing.

    Args:
        args: (image_path, m1_model_path, confidence_threshold, num_rotations, image_size)

    Returns:
        (train_samples, val_samples) or None if failed
    """
    img_path, m1_model_path, confidence_threshold, num_rotations, image_size, train_split = args

    from src.m1_watermeter_detection import M1Inference

    # Auto-detect device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    try:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            return None

        # Convert BGR to RGB for M1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load M1 model (each process loads its own model to avoid GIL issues)
        m1_inference = M1Inference(m1_model_path, confidence_threshold=confidence_threshold, device=device)

        # Run M1 detection
        result = m1_inference.predict(image_rgb)

        if result['success']:
            # Get cropped meter
            cropped = result['cropped_region']

            # Ensure BGR format
            if cropped.shape[2] == 3:  # RGB
                cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            else:
                cropped_bgr = cropped

            # Resize if needed
            if cropped_bgr.shape[0] != image_size or cropped_bgr.shape[1] != image_size:
                cropped_bgr = cv2.resize(cropped_bgr, (image_size, image_size))

            # Generate rotated versions
            train_samples = []
            val_samples = []

            for i in range(num_rotations):
                angle = random.uniform(0, 360)

                # Rotate
                center = (image_size // 2, image_size // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(cropped_bgr, M, (image_size, image_size),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)

                # Determine split
                is_train = random.random() < train_split

                # Encode to JPEG in-memory (faster than writing individual files)
                _, buffer = cv2.imencode('.jpg', rotated)
                jpg_bytes = buffer.tobytes()

                sample = (img_path.stem, i, angle, jpg_bytes, is_train)

                if is_train:
                    train_samples.append(sample)
                else:
                    val_samples.append(sample)

            return (train_samples, val_samples)

        return None

    except Exception as e:
        logger.debug(f"Error processing {img_path.name}: {e}")
        return None


def create_m2_dataset_fast(
    source_images_dir: str,
    m1_model_path: str,
    output_dir: str,
    train_split: float = 0.8,
    num_rotations_per_crop: int = 10,
    image_size: int = 640,
    confidence_threshold: float = 0.50,
    num_workers: int = None,
    max_images: int = None,
):
    """
    Create M2 dataset with multiprocessing for speed.

    Args:
        source_images_dir: Directory with source images
        m1_model_path: Path to M1 model
        output_dir: Output directory
        train_split: Train/val split
        num_rotations_per_crop: Rotations per crop
        image_size: Image size
        confidence_threshold: M1 confidence threshold
        num_workers: Number of parallel workers (default: CPU count)
        max_images: Max images to process (for testing)
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

    if max_images:
        image_files = image_files[:max_images]

    logger.info(f"Found {len(image_files)} source images")

    if not image_files:
        logger.error(f"No images found in {source_images_dir}")
        return

    # Set number of workers
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 4)  # Max 4 to avoid memory issues

    logger.info(f"Using {num_workers} parallel workers")

    # Prepare arguments for multiprocessing
    args_list = [
        (img_path, m1_model_path, confidence_threshold, num_rotations_per_crop, image_size, train_split)
        for img_path in image_files
    ]

    # Process images in parallel
    all_train_samples = []
    all_val_samples = []
    detection_count = 0
    sample_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_image, args) for args in args_list]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result:
                train_samples, val_samples = result
                if train_samples or val_samples:
                    detection_count += 1
                    all_train_samples.extend(train_samples)
                    all_val_samples.extend(val_samples)
                    sample_count += len(train_samples) + len(val_samples)

    # Save images and labels
    logger.info(f"\nSaving {sample_count} images...")

    # Save train images
    for img_stem, idx, angle, jpg_bytes, is_train in tqdm(all_train_samples, desc="Saving train"):
        output_name = f"m2_{img_stem}_{idx}.jpg"
        output_path_full = train_img_dir / output_name

        # Write bytes directly
        with open(output_path_full, 'wb') as f:
            f.write(jpg_bytes)

    # Save val images
    for img_stem, idx, angle, jpg_bytes, is_train in tqdm(all_val_samples, desc="Saving val"):
        output_name = f"m2_{img_stem}_{idx}.jpg"
        output_path_full = val_img_dir / output_name

        with open(output_path_full, 'wb') as f:
            f.write(jpg_bytes)

    # Save label files
    train_labels = output_path / "train" / "labels.csv"
    val_labels = output_path / "val" / "labels.csv"

    with open(train_labels, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'angle'])
        for img_stem, idx, angle, _, _ in all_train_samples:
            writer.writerow([f"m2_{img_stem}_{idx}.jpg", angle])

    with open(val_labels, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'angle'])
        for img_stem, idx, angle, _, _ in all_val_samples:
            writer.writerow([f"m2_{img_stem}_{idx}.jpg", angle])

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ M2 Dataset generated successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"Source images processed: {len(image_files)}")
    logger.info(f"Detections: {detection_count}")
    logger.info(f"Total samples: {sample_count}")
    logger.info(f"Train samples: {len(all_train_samples)}")
    logger.info(f"Val samples: {len(all_val_samples)}")
    logger.info(f"\nOutput: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast M2 dataset creation")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--m1-model", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/m2_orientation")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--rotations", type=int, default=5)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--conf-thresh", type=float, default=0.50)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None,
                       help="Limit images for testing")

    args = parser.parse_args()

    create_m2_dataset_fast(
        source_images_dir=args.source,
        m1_model_path=args.m1_model,
        output_dir=args.output,
        train_split=args.train_split,
        num_rotations_per_crop=args.rotations,
        image_size=args.size,
        confidence_threshold=args.conf_thresh,
        num_workers=args.workers,
        max_images=args.max_images,
    )

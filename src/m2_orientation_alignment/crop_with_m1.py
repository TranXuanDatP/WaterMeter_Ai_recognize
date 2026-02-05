"""
Crop Watermeters Using M1 Detection

Simple script: Use M1 to detect and crop watermeters from source images.
"""

import os
import logging
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def crop_watermeters_with_m1(
    source_images_dir: str,
    m1_model_path: str,
    output_dir: str,
    confidence_threshold: float = 0.50,
    image_size: int = 640,
    max_images: int = None,
):
    """
    Crop watermeters using M1 detection.

    Args:
        source_images_dir: Directory with source full images
        m1_model_path: Path to trained M1 model
        output_dir: Output directory for cropped watermeters
        confidence_threshold: M1 detection confidence threshold
        image_size: Output image size (square)
        max_images: Max images to process (for testing)
    """
    from src.m1_watermeter_detection import M1Inference

    source_dir = Path(source_images_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

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

    # Get source images (recursive search for all subdirectories)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(source_dir.glob(f'**/{ext}'))

    # Remove duplicates (in case of symlinks or other issues)
    image_files = list(set(image_files))

    if max_images:
        image_files = image_files[:max_images]

    logger.info(f"Found {len(image_files)} source images")

    if not image_files:
        logger.error(f"No images found in {source_images_dir}")
        return

    # Process images and save crops
    detection_count = 0
    no_detection_count = 0
    error_count = 0

    for img_path in tqdm(image_files, desc="Cropping watermeters"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            error_count += 1
            continue

        # Convert BGR to RGB for M1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # Run M1 detection
            result = m1_inference.predict(image_rgb)

            if result['success']:
                detection_count += 1

                # Get cropped meter (M1 output - already 640x640)
                cropped = result['cropped_region']

                # Ensure it's in BGR for OpenCV saving
                if cropped.shape[2] == 3:  # RGB
                    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                else:
                    cropped_bgr = cropped

                # Generate filename
                output_name = f"crop_{img_path.stem}.jpg"
                output_path_full = output_path / output_name

                # Save crop
                cv2.imwrite(str(output_path_full), cropped_bgr)

            else:
                no_detection_count += 1

        except Exception as e:
            logger.warning(f"Error processing {img_path.name}: {e}")
            error_count += 1
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Cropping complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Source images processed: {len(image_files)}")
    logger.info(f"Detections: {detection_count}")
    logger.info(f"No detections: {no_detection_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"\nCrops saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop watermeters using M1 detection")
    parser.add_argument("--source", type=str, required=True,
                       help="Source images directory")
    parser.add_argument("--m1-model", type=str, required=True,
                       help="Path to M1 model (detect_watermeter.pt)")
    parser.add_argument("--output", type=str, default="data/m2_crops",
                       help="Output directory for cropped watermeters")
    parser.add_argument("--conf-thresh", type=float, default=0.50,
                       help="M1 confidence threshold (default: 0.50)")
    parser.add_argument("--size", type=int, default=640,
                       help="Image size (default: 640)")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Max images to process (for testing)")

    args = parser.parse_args()

    crop_watermeters_with_m1(
        source_images_dir=args.source,
        m1_model_path=args.m1_model,
        output_dir=args.output,
        confidence_threshold=args.conf_thresh,
        image_size=args.size,
        max_images=args.max_images,
    )

"""
M1 Evaluation Script

Evaluate trained YOLOv8 watermeter detection model.
"""

import argparse
import logging
import os
import sys
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate M1 Watermeter Detection Model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pt file)",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset.yaml configuration",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu, default: cuda)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for metrics",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark",
    )

    parser.add_argument(
        "--benchmark-images",
        type=int,
        default=100,
        help="Number of images for benchmark (default: 100)",
    )

    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.data):
        logger.error(f"Dataset config not found: {args.data}")
        sys.exit(1)

    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Import here
    try:
        from .model import M1Model
        from .inference import M1Inference
        from .config import get_config
    except ImportError as e:
        logger.error(f"Import error: {e}")
        sys.exit(1)

    # Load model
    logger.info(f"Loading model from: {args.model}")
    model = M1Model(model_path=args.model)

    # Validate
    logger.info("Running validation...")
    metrics = model.validate(
        data_yaml=args.data,
        batch=args.batch,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("Validation Results")
    logger.info("=" * 60)
    logger.info(f"mAP@0.5:  {metrics.get('maps50', 'N/A'):.4f}" if 'maps50' in metrics else "mAP@0.5:  N/A")
    logger.info(f"mAP@0.95: {metrics.get('maps95', 'N/A'):.4f}" if 'maps95' in metrics else "mAP@0.95: N/A")
    logger.info(f"Precision: {metrics.get('precision', 'N/A'):.4f}" if 'precision' in metrics else "Precision: N/A")
    logger.info(f"Recall:    {metrics.get('recall', 'N/A'):.4f}" if 'recall' in metrics else "Recall: N/A")

    # Check if target met
    target_map = 0.98
    current_map = metrics.get('maps50', 0)
    if current_map >= target_map:
        logger.info(f"✅ Target met: mAP@0.5 {current_map:.4f} >= {target_map}")
    else:
        logger.warning(f"⚠️  Target not met: mAP@0.5 {current_map:.4f} < {target_map}")

    # Benchmark if requested
    if args.benchmark:
        logger.info("\nRunning inference benchmark...")

        inference = M1Inference(args.model, device=args.device)

        # Load sample images for benchmark
        # Note: In real usage, you'd load actual test images
        logger.info(f"Benchmarking on {args.benchmark_images} random images...")
        logger.warning("⚠️  Benchmark requires actual test images - skipping for now")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

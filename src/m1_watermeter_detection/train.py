"""
M1 Training Script

Command-line interface for training YOLOv8 watermeter detection model.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train M1 Watermeter Detection Model"
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset.yaml configuration",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model architecture (default: yolov8s.pt)",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model for fine-tuning",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer type (default: AdamW)",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu, default: cuda)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)",
    )

    # Output
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="MLflow project name (default: runs/detect)",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="m1_train",
        help="Run name (default: m1_train)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for final model",
    )

    # Features
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.data):
        logger.error(f"Dataset config not found: {args.data}")
        sys.exit(1)

    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Import here to avoid import errors if dependencies not installed
    try:
        from .model import M1Model
        from .config import get_config
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install required packages: pip install torch ultralytics mlflow")
        sys.exit(1)

    # Create config
    config = get_config(
        model_name=args.model,
        batch_size=args.batch,
        epochs=args.epochs,
        input_size=args.imgsz,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        device=args.device,
        num_workers=args.workers,
        mlflow_enabled=not args.no_mlflow,
    )

    # Initialize model
    logger.info("Initializing M1 model...")
    model = M1Model(config=config, model_path=args.pretrained or args.resume)

    # Train
    logger.info("Starting training...")
    logger.info(f"Dataset: {args.data}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch}")
    logger.info(f"Device: {args.device}")

    try:
        metrics = model.train(
            data_yaml=args.data,
            project=args.project,
            name=args.name,
        )

        # Print results
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        logger.info(f"mAP@0.5: {metrics.get('maps50', 'N/A')}")
        logger.info(f"mAP@0.95: {metrics.get('maps95', 'N/A')}")
        logger.info(f"Precision: {metrics.get('precision', 'N/A')}")
        logger.info(f"Recall: {metrics.get('recall', 'N/A')}")

        # Save final model
        if args.output:
            output_path = os.path.join(args.output, "m1_watermeter_detection.pt")
            model.save(output_path)
            logger.info(f"Model saved to: {output_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

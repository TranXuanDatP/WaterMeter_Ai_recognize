"""
M2 Training Script

Train sin/cos angle regression model for watermeter orientation alignment.
"""

import os
import logging
from pathlib import Path
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import numpy as np
    from PIL import Image
    import mlflow
    import mlflow.pytorch
except ImportError as e:
    raise ImportError(
        f"Required dependencies not installed: {e}\n"
        "Please install: pip install torch torchvision pillow mlflow"
    )

from .model import (
    M2AngleRegressor,
    SinCosLoss,
    angle_to_sin_cos,
    sin_cos_to_angle,
    compute_circular_mae,
)

# Legacy alias for backward compatibility
compute_mae = compute_circular_mae

from .config import M2_CONFIG, get_config


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WatermeterAngleDataset(Dataset):
    """Dataset for watermeter angle regression."""

    def __init__(
        self,
        images_dir: str,
        labels_file: str,
        transform=None,
        input_size=640,
    ):
        """
        Initialize dataset.

        Args:
            images_dir: Directory containing images
            labels_file: File containing angle labels (CSV format: image_name,angle)
            transform: Optional transforms
            input_size: Input image size
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.input_size = input_size

        # Load labels
        self.samples = []
        with open(labels_file, 'r') as f:
            next(f)  # Skip header row
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    image_name = parts[0]
                    try:
                        angle = float(parts[1])
                        self.samples.append((image_name, angle))
                    except ValueError:
                        continue  # Skip invalid rows

        logger.info(f"Loaded {len(self.samples)} samples from {labels_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, angle = self.samples[idx]

        # Load image
        image_path = self.images_dir / image_name
        image = Image.open(image_path).convert('RGB')

        # Resize
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert angle to sin/cos
        # angle is scalar, convert to tensor then to sin/cos [1, 2]
        angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32))
        sin_theta = torch.sin(angle_rad)
        cos_theta = torch.cos(angle_rad)
        sin_cos = torch.stack([sin_theta, cos_theta], dim=0)  # Shape: [2]

        return image, sin_cos


def create_train_transform(input_size=640):
    """Create training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),  # ±30° rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def create_val_transform(input_size=640):
    """Create validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Store predictions for MAE calculation
        pred_angles = sin_cos_to_angle(outputs)
        target_angles = sin_cos_to_angle(targets)
        all_preds.append(pred_angles.cpu())
        all_targets.append(target_angles.cpu())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = compute_mae(all_preds, all_targets)

    return avg_loss, mae


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)

            pred_angles = sin_cos_to_angle(outputs)
            target_angles = sin_cos_to_angle(targets)
            all_preds.append(pred_angles.cpu())
            all_targets.append(target_angles.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = compute_mae(all_preds, all_targets)

    return avg_loss, mae


def train_m2_model(
    train_images_dir: str,
    train_labels_file: str,
    val_images_dir: str,
    val_labels_file: str,
    output_dir: str = "checkpoints/orientation/",
    config=None,
):
    """
    Train M2 angle regression model.

    Args:
        train_images_dir: Training images directory
        train_labels_file: Training labels file (CSV)
        val_images_dir: Validation images directory
        val_labels_file: Validation labels file (CSV)
        output_dir: Directory to save checkpoints
        config: Model configuration (uses default if None)
    """
    config = config or get_config()

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # MLflow tracking
    if config.mlflow_enabled:
        mlflow.set_experiment(config.mlflow_experiment_name)
        mlflow.start_run()
        mlflow.log_params({
            "backbone": config.backbone,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "dropout": config.dropout,
        })

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = WatermeterAngleDataset(
        train_images_dir,
        train_labels_file,
        transform=create_train_transform(config.input_size),
        input_size=config.input_size,
    )
    val_dataset = WatermeterAngleDataset(
        val_images_dir,
        val_labels_file,
        transform=create_val_transform(config.input_size),
        input_size=config.input_size,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create model
    logger.info("Creating model...")
    model = M2AngleRegressor(
        backbone=config.backbone,
        pretrained=True,
        dropout=config.dropout,
    ).to(device)

    # Loss and optimizer
    criterion = SinCosLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    if config.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )

    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Train
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        logger.info(f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f}°")

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}°")

        # MLflow logging
        if config.mlflow_enabled:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "lr": optimizer.param_groups[0]['lr'],
            }, step=epoch)

        # Learning rate scheduling
        scheduler.step(val_mae)

        # Save checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'backbone': config.backbone,
                'dropout': config.dropout,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_mae < best_val_mae - config.min_delta:
            best_val_mae = val_mae
            patience_counter = 0

            best_model_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'backbone': config.backbone,
                'dropout': config.dropout,
            }, best_model_path)
            logger.info(f"✅ Saved best model (MAE: {val_mae:.2f}°)")

            # Check if target reached
            if val_mae <= config.target_mae:
                logger.info(f"🎉 Target MAE ({config.target_mae}°) reached!")
                break
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered (patience={config.patience})")
            break

    # Final evaluation
    logger.info("\n" + "="*50)
    logger.info(f"Training complete! Best Val MAE: {best_val_mae:.2f}°")
    logger.info(f"Target MAE: {config.target_mae}°")

    if config.mlflow_enabled:
        mlflow.end_run()

    return model, best_val_mae


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train M2 orientation alignment model")
    parser.add_argument("--train-images", type=str, required=True, help="Training images directory")
    parser.add_argument("--train-labels", type=str, required=True, help="Training labels CSV file")
    parser.add_argument("--val-images", type=str, required=True, help="Validation images directory")
    parser.add_argument("--val-labels", type=str, required=True, help="Validation labels CSV file")
    parser.add_argument("--output", type=str, default="checkpoints/orientation/", help="Output directory")

    args = parser.parse_args()

    train_m2_model(
        train_images_dir=args.train_images,
        train_labels_file=args.train_labels,
        val_images_dir=args.val_images,
        val_labels_file=args.val_labels,
        output_dir=args.output,
    )

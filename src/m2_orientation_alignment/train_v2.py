"""
Enhanced M2 Training Script

Improvements:
1. Advanced data augmentation
2. Learning rate scheduling
3. Early stopping with patience
4. Mixed precision training
5. Better logging
6. Validation during training
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.m2_orientation_alignment.model import (
    angle_to_sin_cos,
    sin_cos_to_angle,
    compute_circular_mae
)

# Import improved model
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from improve_m2_model import (
    M2AngleRegressorV2,
    CombinedLoss,
    AngularLoss,
    SinCosLoss,
    M2_TRAINING_CONFIGS
)

logger = logging.getLogger(__name__)


# ==========================================
# 1. DATASET WITH AUGMENTATION
# ==========================================

class MeterRotationDataset(Dataset):
    """
    Dataset for meter orientation with advanced augmentation.

    Augmentations:
    - Random rotation (for robustness)
    - Color jitter (lighting variation)
    - Gaussian blur (focus variation)
    - Random perspective (viewpoint variation)
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        split: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.split = split

        # Get all image files
        self.images = list(self.data_dir.glob("*.jpg")) + \
                     list(self.data_dir.glob("*.png"))

        if len(self.images) == 0:
            raise ValueError(f"No images found in {data_dir}")

        # Default transforms
        if transform is None:
            if split == "train":
                # Aggressive augmentation for training
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomRotation(30),  # Random rotation ±30°
                    transforms.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.2,
                        hue=0.1
                    ),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                    ], p=0.3),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                # No augmentation for validation
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        else:
            self.transform = transform

        logger.info(f"Dataset [{split}]: {len(self.images)} images from {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        img_path = self.images[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Extract angle from filename or use default (0°)
        # Format: image_045.jpg -> angle=45°
        try:
            angle_str = img_path.stem.split('_')[-1]
            angle = float(angle_str)
        except:
            # If no angle in filename, assume 0° (upright)
            angle = 0.0

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Convert angle to sin/cos
        angle_tensor = torch.tensor([angle], dtype=torch.float32)
        sin_cos_label = angle_to_sin_cos(angle_tensor)

        return img, sin_cos_label.squeeze(0), img_path.name


# ==========================================
# 2. TRAINER CLASS
# ==========================================

class M2Trainer:
    """
    Enhanced M2 Trainer with:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Comprehensive logging
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: dict,
        output_dir: str = "outputs/m2_training"
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=1e-4
        )

        # Scheduler: ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )

        # Mixed precision
        self.scaler = GradScaler() if torch.cuda.is_available() else None

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 15

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'lr': []
        }

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Output dir: {self.output_dir}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                # Backward with scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_pred_angles = []
        all_target_angles = []

        for images, labels, _ in tqdm(val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            total_loss += loss.item()

            # Decode angles for MAE calculation
            pred_angles = sin_cos_to_angle(outputs)
            target_angles = sin_cos_to_angle(labels)

            all_pred_angles.append(pred_angles.cpu())
            all_target_angles.append(target_angles.cpu())

        avg_loss = total_loss / len(val_loader)

        # Compute MAE
        all_pred_angles = torch.cat(all_pred_angles)
        all_target_angles = torch.cat(all_target_angles)
        mae = compute_circular_mae(all_pred_angles, all_target_angles)

        return avg_loss, mae

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ):
        """Full training loop"""
        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.epoch = epoch + 1

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_mae = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_mae={val_mae:.2f}°, "
                f"lr={current_lr:.2e}"
            )

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['lr'].append(current_lr)

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_mae)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_best_model(val_mae)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.max_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save history
        self.save_history()
        logger.info("Training complete!")

    def save_checkpoint(self, epoch: int, val_loss: float, val_mae: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae,
            'config': self.config,
            'history': self.history
        }

        path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def save_best_model(self, val_mae: float):
        """Save best model"""
        torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
        logger.info(f"✅ Saved best model (MAE: {val_mae:.2f}°)")

    def save_history(self):
        """Save training history"""
        with open(self.output_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)


# ==========================================
# 3. MAIN TRAINING SCRIPT
# ==========================================

def main():
    """
    Main training script.

    Usage:
        python train_v2.py --config improved_v1 --data_dir data/m2_dataset
    """
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced M2 Training")
    parser.add_argument("--config", type=str, default="improved_v1",
                       choices=list(M2_TRAINING_CONFIGS.keys()),
                       help="Training configuration")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="outputs/m2_training",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get config
    config = M2_TRAINING_CONFIGS[args.config]
    logger.info(f"Using config: {args.config}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Create model and loss
    model, loss_fn = get_model(args.config)

    # Create trainer
    trainer = M2Trainer(model, loss_fn, config, args.output_dir)

    # Create datasets
    # Assuming data is split into train/val subdirectories
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    if not train_dir.exists() or not val_dir.exists():
        # If no split, use all data for training
        logger.warning("No train/val split found, using all data for training")
        train_dataset = MeterRotationDataset(args.data_dir, split="train")
        val_dataset = train_dataset  # Use same for validation
    else:
        train_dataset = MeterRotationDataset(train_dir, split="train")
        val_dataset = MeterRotationDataset(val_dir, split="val")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Train
    trainer.fit(
        train_loader,
        val_loader,
        epochs=config["epochs"]
    )


if __name__ == "__main__":
    main()

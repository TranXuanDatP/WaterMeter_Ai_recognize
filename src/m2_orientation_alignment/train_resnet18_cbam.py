"""
Train M2 Model: ResNet18 + CBAM + Deep Head

Configuration:
- Backbone: ResNet18 (11M params)
- Attention: CBAM (Channel + Spatial)
- Head: Deep with LayerNorm (512→256→128→2)
- Loss: Combined (MSE + Angular)
- Expected MAE: ~1.1-1.2° (25-30% improvement from baseline)

Usage:
    python train_resnet18_cbam.py --data_dir data/m2_dataset --output_dir outputs/m2_resnet18_cbam
"""

import os
import sys
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
from PIL import Image

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from src.m2_orientation_alignment.model import (
    angle_to_sin_cos,
    sin_cos_to_angle,
    compute_circular_mae
)
from improve_m2_model import (
    M2Backbone,
    CBAM,
    CombinedLoss
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# MODEL: ResNet18 + CBAM + Deep Head
# ==========================================

class M2ResNet18CBAM(nn.Module):
    """
    ResNet18 + CBAM + Deep Regression Head

    Architecture:
        Input (3, 224, 224)
          ↓
        ResNet18 Backbone (features: 512)
          ↓
        CBAM Attention (channel + spatial)
          ↓
        Deep Head with LayerNorm:
          FC(512 → 256) → LayerNorm → ReLU → Dropout(0.3)
          FC(256 → 128) → LayerNorm → ReLU → Dropout(0.2)
          FC(128 → 2)
          ↓
        Output: [sin(θ), cos(θ)] normalized
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        # Backbone: ResNet18
        self.backbone = M2Backbone("resnet18", pretrained)

        # Attention: CBAM
        self.attention = CBAM(512, reduction=16)

        # Deep Regression Head with LayerNorm
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),

            nn.Linear(128, 2)  # [sin, cos]
        )

        logger.info("✅ M2ResNet18CBAM initialized")
        logger.info(f"   - Backbone: ResNet18 ({'pretrained' if pretrained else 'scratch'})")
        logger.info(f"   - Attention: CBAM (512 channels)")
        logger.info(f"   - Head: Deep (512→256→128→2)")
        logger.info(f"   - Dropout: {dropout}")

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Apply attention
        features = self.attention(features)

        # Predict sin/cos
        sin_cos = self.regressor(features)

        # Normalize to unit circle
        sin_cos = nn.functional.normalize(sin_cos, p=2, dim=1)

        return sin_cos


# ==========================================
# DATASET
# ==========================================

class MeterRotationDataset(Dataset):
    """Dataset for meter orientation training"""

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split

        # Get images
        self.images = list(self.data_dir.glob("*.jpg")) + \
                     list(self.data_dir.glob("*.png"))

        if len(self.images) == 0:
            raise ValueError(f"No images found in {data_dir}")

        # Transforms
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
            ])

        logger.info(f"Dataset [{split}]: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Extract angle from filename (e.g., image_045.jpg -> 45°)
        try:
            angle_str = img_path.stem.split('_')[-1]
            angle = float(angle_str)
        except:
            angle = 0.0

        # Apply transform
        img = self.transform(img)

        # Convert angle to sin/cos
        angle_tensor = torch.tensor([angle], dtype=torch.float32)
        sin_cos_label = angle_to_sin_cos(angle_tensor)

        return img, sin_cos_label.squeeze(0)


# ==========================================
# TRAINER
# ==========================================

class Trainer:
    """Simplified trainer for ResNet18+CBAM"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str = "outputs/m2_resnet18_cbam"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        logger.info(f"Training on: {self.device}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=5e-5,
            weight_decay=1e-4
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )

        # Loss
        self.loss_fn = CombinedLoss(mse_weight=0.5, angular_weight=0.5)

        # Mixed precision
        self.scaler = GradScaler() if torch.cuda.is_available() else None

        # State
        self.best_val_mae = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_pred = []
        all_target = []

        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            total_loss += loss.item()

            pred_angles = sin_cos_to_angle(outputs)
            target_angles = sin_cos_to_angle(labels)

            all_pred.append(pred_angles.cpu())
            all_target.append(target_angles.cpu())

        avg_loss = total_loss / len(self.val_loader)

        # Compute MAE
        all_pred = torch.cat(all_pred)
        all_target = torch.cat(all_target)
        mae = compute_circular_mae(all_pred, all_target)

        return avg_loss, mae

    def fit(self, epochs: int = 75):
        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_mae = self.validate()

            # Update scheduler
            self.scheduler.step(val_mae)

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_mae={val_mae:.2f}°, "
                f"lr={lr:.2e}"
            )

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)

            # Save best model
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(),
                          self.output_dir / "best_model.pth")
                logger.info(f"✅ Saved best model (MAE: {val_mae:.2f}°)")

            # Save checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_mae': val_mae,
                }, self.output_dir / f"checkpoint_epoch_{epoch}.pth")

            # Early stopping
            if epoch - self.history['val_mae'].index(min(self.history['val_mae'])) > 20:
                logger.info("Early stopping!")
                break

        # Save history
        with open(self.output_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Training complete! Best MAE: {self.best_val_mae:.2f}°")


# ==========================================
# MAIN
# ==========================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train M2: ResNet18 + CBAM + Deep Head"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/m2_resnet18_cbam",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    # Create model
    logger.info("=" * 60)
    logger.info("M2 Training: ResNet18 + CBAM + Deep Head")
    logger.info("=" * 60)

    model = M2ResNet18CBAM(pretrained=True, dropout=0.3)

    # Create datasets
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    if train_dir.exists():
        train_dataset = MeterRotationDataset(train_dir, "train")
    else:
        train_dataset = MeterRotationDataset(args.data_dir, "train")

    if val_dir.exists():
        val_dataset = MeterRotationDataset(val_dir, "val")
    else:
        # Use 20% of training data for validation
        val_size = len(train_dataset) // 5
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, args.output_dir)

    # Train
    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()

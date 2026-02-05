"""
M2 Model Implementation - Sin/Cos Angle Regression CNN

Predicts watermeter orientation angle using sin/cos representation.
Handles periodicity correctly: 350° ≡ -10°
"""

import math
import logging
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
except ImportError as e:
    raise ImportError(
        f"Required dependencies not installed: {e}\n"
        "Please install: pip install torch torchvision"
    )

try:
    from .config import M2ModelConfig, get_config
except ImportError:
    # Allow running this file standalone for testing
    M2ModelConfig = None
    get_config = None

logger = logging.getLogger(__name__)


# ==========================================
# 1. MODEL ARCHITECTURE (M2)
# ==========================================
class M2AngleRegressor(nn.Module):
    """
    M2 Angle Regression CNN

    Predicts watermeter rotation angle using sin/cos representation.
    Input: RGB image (Batch, 3, H, W)
    Output: Vector [sin(theta), cos(theta)]
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        """
        Initialize M2 angle regressor.

        Args:
            backbone: Backbone architecture ('resnet18' or 'mobilenet_v2')
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate for regression head
        """
        super(M2AngleRegressor, self).__init__()

        self.backbone_name = backbone

        # 1. Backbone (Feature Extractor)
        if backbone == "resnet18":
            weights = 'DEFAULT' if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove old classification head

        elif backbone == "mobilenet_v2":
            weights = 'DEFAULT' if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # Remove old classification head

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18' or 'mobilenet_v2'")

        # 2. Regression Head (2 outputs: sin, cos)
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # Output: [sin, cos]
        )

        logger.info(f"M2 Model initialized: backbone={backbone}, pretrained={pretrained}, features={num_features}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (batch, 3, H, W)

        Returns:
            Sin/cos predictions (batch, 2) normalized to unit circle
        """
        # Extract features
        features = self.backbone(x)

        # Predict raw sin/cos values
        sin_cos = self.regressor(features)

        # IMPORTANT: Normalize to unit circle (sin² + cos² = 1)
        # This ensures predictions stay on valid unit circle
        sin_cos = F.normalize(sin_cos, p=2, dim=1)

        return sin_cos


# ==========================================
# 2. LOSS FUNCTION
# ==========================================
class SinCosLoss(nn.Module):
    """
    Sin/Cos Regression Loss Function

    L = MSE(pred_sin, target_sin) + MSE(pred_cos, target_cos)

    This handles periodicity correctly: 350° ≡ -10°
    """

    def __init__(self):
        super(SinCosLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate sin/cos loss.

        Args:
            pred: (batch, 2) - [sin, cos] predictions
            target: (batch, 2) - [sin, cos] targets

        Returns:
            Scalar loss value
        """
        # Direct MSE on both sin and cos components
        return self.mse(pred, target)


# ==========================================
# 3. UTILITY FUNCTIONS (Convert & Metric)
# ==========================================

def angle_to_sin_cos(angle_degrees: torch.Tensor) -> torch.Tensor:
    """
    Convert angle in degrees to sin/cos values.

    Args:
        angle_degrees: Angle in degrees, shape (batch,) or scalar
                      Range: [0, 360)

    Returns:
        Sin/cos tensor (batch, 2)
        Example: angle=90° → [1.0, 0.0] (sin=1, cos=0)
    """
    angles_rad = torch.deg2rad(angle_degrees)
    sin_val = torch.sin(angles_rad)
    cos_val = torch.cos(angles_rad)
    return torch.stack([sin_val, cos_val], dim=1)


def sin_cos_to_angle(sin_cos: torch.Tensor) -> torch.Tensor:
    """
    Convert sin/cos to angle in degrees using atan2.

    θ = atan2(sin, cos) mod 360

    Args:
        sin_cos: Sin/cos values (batch, 2)

    Returns:
        Angle in degrees [0, 360)

    Examples:
        [1.0, 0.0] → 90°
        [0.0, 1.0] → 0°
        [-1.0, 0.0] → 270°
    """
    sin_val = sin_cos[:, 0]
    cos_val = sin_cos[:, 1]

    # atan2 returns radians in range (-pi, pi)
    angles_rad = torch.atan2(sin_val, cos_val)

    # Convert to degrees
    angles_deg = torch.rad2deg(angles_rad)

    # Normalize to [0, 360)
    # Example: -10° → 350°
    angles_deg = (angles_deg + 360) % 360

    return angles_deg


def compute_circular_mae(pred_angles: torch.Tensor, target_angles: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error (MAE) on circular domain.

    Handles periodicity correctly:
    - Error between 359° and 1° is 2°, NOT 358°
    - Error between 90° and 270° is 180°

    Args:
        pred_angles: Predicted angles in degrees [0, 360)
        target_angles: Target angles in degrees [0, 360)

    Returns:
        MAE in degrees (scalar)

    Example:
        pred=[359°], target=[1°] → MAE=2° (not 358°)
    """
    diff = torch.abs(pred_angles - target_angles)
    # Take shortest distance on circle
    diff = torch.minimum(diff, 360 - diff)
    return diff.mean().item()


# ==========================================
# 4. LEGACY ALIASES (for backward compatibility)
# ==========================================
# Old function names → mapped to new names
M2DetectionError = RuntimeError  # Legacy compatibility


# ==========================================
# TEST / DEMO
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("M2 Model Test - Sin/Cos Angle Regression")
    print("="*60)

    # 1. Create model
    model = M2AngleRegressor(backbone='resnet18', pretrained=False)
    loss_fn = SinCosLoss()

    # 2. Create dummy data (batch_size=4)
    # Target angles: 0°, 90°, 180°, 270° (4 quadrants)
    target_angles = torch.tensor([0.0, 90.0, 180.0, 270.0])

    # Convert to sin/cos labels
    target_sincos = angle_to_sin_cos(target_angles)

    print(f"\nTarget Angles: {target_angles.tolist()}")
    print(f"Target Sin/Cos:\n{target_sincos.numpy()}")

    # 3. Simulate input images (4 images, 3 channels, 224x224)
    dummy_input = torch.randn(4, 3, 224, 224)

    # 4. Forward pass
    print("\nRunning forward pass...")
    pred_sincos = model(dummy_input)

    print(f"Predicted Sin/Cos:\n{pred_sincos.detach().numpy()}")

    # 5. Compute loss
    loss = loss_fn(pred_sincos, target_sincos)
    print(f"\nLoss: {loss.item():.4f}")

    # 6. Decode predictions to angles
    pred_angles = sin_cos_to_angle(pred_sincos)
    print(f"\nPredicted Angles: {pred_angles.detach().numpy()}")

    # 7. Compute circular MAE
    mae = compute_circular_mae(pred_angles, target_angles)
    print(f"\nCircular MAE: {mae:.2f}°")

    # 8. Test periodicity handling
    print("\n" + "="*60)
    print("Testing Periodicity Handling")
    print("="*60)

    # Test case: 359° vs 1° → should be 2° error (not 358°)
    pred_test = torch.tensor([359.0])
    target_test = torch.tensor([1.0])
    mae_test = compute_circular_mae(pred_test, target_test)
    print(f"\nPred: 359°, Target: 1°")
    print(f"Error: {mae_test:.0f}° ✓ (should be 2°, not 358°)")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

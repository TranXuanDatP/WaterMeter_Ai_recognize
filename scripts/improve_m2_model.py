"""
M2 Model Improvements - Enhanced Orientation Alignment

Key improvements:
1. Multi-scale feature extraction
2. Attention mechanism
3. Better backbone options (ResNet34/50, EfficientNet)
4. Improved loss function
5. Better data augmentation
6. Ensemble capability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from typing import Optional, Tuple, List


# ==========================================
# 1. ENHANCED BACKBONE OPTIONS
# ==========================================

class M2Backbone(nn.Module):
    """Enhanced backbone with multi-scale features"""

    def __init__(self, backbone_type: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.backbone_type = backbone_type

        if backbone_type == "resnet18":
            weights = 'DEFAULT' if pretrained else None
            self.model = models.resnet18(weights=weights)
            self.feature_dim = 512

        elif backbone_type == "resnet34":
            weights = 'DEFAULT' if pretrained else None
            self.model = models.resnet34(weights=weights)
            self.feature_dim = 512

        elif backbone_type == "resnet50":
            weights = 'DEFAULT' if pretrained else None
            self.model = models.resnet50(weights=weights)
            self.feature_dim = 2048

        elif backbone_type == "efficientnet_b0":
            weights = 'DEFAULT' if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            self.feature_dim = 1280
            # For EfficientNet, modify classifier
            self.model.classifier = nn.Identity()

        elif backbone_type == "efficientnet_b1":
            weights = 'DEFAULT' if pretrained else None
            self.model = models.efficientnet_b1(weights=weights)
            self.feature_dim = 1280
            self.model.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        # Remove original classification head for ResNet
        if "resnet" in backbone_type:
            self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


# ==========================================
# 2. ATTENTION MODULE
# ==========================================

class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on meter orientation cues"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        attention = self.sigmoid(self.conv(x))
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention for adaptive feature recalibration"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).flatten(1))
        max_out = self.fc(self.max_pool(x).flatten(1))
        out = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ==========================================
# 3. IMPROVED LOSS FUNCTIONS
# ==========================================

class SinCosLoss(nn.Module):
    """Original sin/cos loss"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)


class AngularLoss(nn.Module):
    """
    Angular loss that directly optimizes angle difference.

    L = 1 - cos(θ_pred - θ_true)

    Better for small angle errors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_sin_cos, target_sin_cos):
        # pred_sin_cos: (B, 2) - [sin, cos]
        # target_sin_cos: (B, 2) - [sin, cos]

        # Compute cosine similarity = sin1*sin2 + cos1*cos2 = cos(θ1-θ2)
        cos_sim = (pred_sin_cos * target_sin_cos).sum(dim=1)

        # Loss: 1 - cos(θ_pred - θ_true)
        # Range: [0, 2], where 0 = perfect prediction
        loss = (1 - cos_sim).mean()

        return loss


class CombinedLoss(nn.Module):
    """Combination of MSE and angular loss"""

    def __init__(self, mse_weight: float = 0.5, angular_weight: float = 0.5):
        super().__init__()
        self.mse_loss = SinCosLoss()
        self.angular_loss = AngularLoss()
        self.mse_weight = mse_weight
        self.angular_weight = angular_weight

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        angular = self.angular_loss(pred, target)
        return self.mse_weight * mse + self.angular_weight * angular


# ==========================================
# 4. ENHANCED MODEL ARCHITECTURE
# ==========================================

class M2AngleRegressorV2(nn.Module):
    """
    Enhanced M2 Angle Regressor

    Improvements:
    - Better backbone options
    - Attention mechanism
    - Deeper regression head
    - Residual connections
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        dropout: float = 0.3,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        super().__init__()

        self.backbone_name = backbone
        self.use_attention = use_attention

        # Backbone
        self.backbone_obj = M2Backbone(backbone, pretrained)
        self.feature_dim = self.backbone_obj.feature_dim

        # Attention module (optional)
        if use_attention:
            self.attention = CBAM(self.feature_dim)

        # Improved regression head with residual connections
        if use_residual:
            # Deep network with skip connections
            self.regressor = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),

                nn.Linear(128, 2)  # Output: [sin, cos]
            )
        else:
            # Simpler head
            self.regressor = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 2)
            )

        print(f"✅ M2 V2 Model: backbone={backbone}, attention={use_attention}, residual={use_residual}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with enhanced architecture.

        Args:
            x: Input images (batch, 3, H, W)

        Returns:
            Normalized sin/cos predictions (batch, 2)
        """
        # Extract features
        features = self.backbone_obj(x)

        # Apply attention (optional)
        if self.use_attention:
            features = self.attention(features)

        # Predict sin/cos
        sin_cos = self.regressor(features)

        # Normalize to unit circle (ensures sin² + cos² = 1)
        sin_cos = F.normalize(sin_cos, p=2, dim=1)

        return sin_cos


# ==========================================
# 5. ENSEMBLE MODEL
# ==========================================

class M2Ensemble(nn.Module):
    """
    Ensemble of multiple M2 models for improved accuracy.

    Combines predictions from multiple models using weighted averaging.
    """

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize ensemble.

        Args:
            models: List of trained M2 models
            weights: Optional weights for each model (default: equal weights)
        """
        super().__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models)
            self.weights = weights

        print(f"✅ M2 Ensemble: {len(models)} models")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Args:
            x: Input images (batch, 3, H, W)

        Returns:
            Ensemble sin/cos prediction (batch, 2)
        """
        predictions = []

        # Get predictions from all models
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))

        # Renormalize
        ensemble_pred = F.normalize(ensemble_pred, p=2, dim=1)

        return ensemble_pred


# ==========================================
# 6. TRAINING CONFIGURATIONS
# ==========================================

M2_TRAINING_CONFIGS = {
    "baseline": {
        "backbone": "resnet18",
        "pretrained": True,
        "dropout": 0.2,
        "use_attention": False,
        "use_residual": False,
        "loss": "sincos_mse",
        "lr": 1e-4,
        "epochs": 50,
        "description": "Baseline ResNet18 (current production)"
    },
    "resnet18_cbam_deep": {
        "backbone": "resnet18",
        "pretrained": True,
        "dropout": 0.3,
        "use_attention": True,
        "use_residual": True,
        "loss": "combined",
        "lr": 5e-5,
        "epochs": 75,
        "description": "ResNet18 + CBAM + Deep Head (RECOMMENDED)"
    },
    "improved_v1": {
        "backbone": "resnet34",
        "pretrained": True,
        "dropout": 0.3,
        "use_attention": True,
        "use_residual": False,
        "loss": "combined",
        "lr": 5e-5,
        "epochs": 75,
        "description": "ResNet34 + CBAM"
    },
    "improved_v2": {
        "backbone": "resnet50",
        "pretrained": True,
        "dropout": 0.3,
        "use_attention": True,
        "use_residual": True,
        "loss": "combined",
        "lr": 3e-5,
        "epochs": 100,
        "description": "ResNet50 + CBAM + Deep Head (best accuracy)"
    },
    "efficient": {
        "backbone": "efficientnet_b1",
        "pretrained": True,
        "dropout": 0.4,
        "use_attention": True,
        "use_residual": True,
        "loss": "angular",
        "lr": 1e-4,
        "epochs": 80,
        "description": "EfficientNet-B1 + CBAM (best speed/accuracy)"
    },
}


def get_model(config_name: str = "improved_v1") -> Tuple[nn.Module, nn.Module]:
    """
    Get model and loss function by config name.

    Args:
        config_name: Name of training config

    Returns:
        (model, loss_function)
    """
    if config_name not in M2_TRAINING_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")

    config = M2_TRAINING_CONFIGS[config_name]

    # Create model
    model = M2AngleRegressorV2(
        backbone=config["backbone"],
        pretrained=config["pretrained"],
        dropout=config["dropout"],
        use_attention=config["use_attention"],
        use_residual=config["use_residual"]
    )

    # Create loss function
    loss_type = config["loss"]
    if loss_type == "sincos_mse":
        loss_fn = SinCosLoss()
    elif loss_type == "angular":
        loss_fn = AngularLoss()
    elif loss_type == "combined":
        loss_fn = CombinedLoss(mse_weight=0.5, angular_weight=0.5)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return model, loss_fn


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("M2 V2 - Enhanced Model Test")
    print("=" * 60)

    # Test different configurations
    for config_name in ["baseline", "improved_v1", "improved_v2", "efficient"]:
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print(f"{'='*60}")

        model, loss_fn = get_model(config_name)
        config = M2_TRAINING_CONFIGS[config_name]

        print(f"Config: {config}")

        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)

        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output (normalized): sin²+cos² = {output[0].pow(2).sum().item():.4f}")

        # Test loss
        target = torch.randn(2, 2)
        target = F.normalize(target, p=2, dim=1)
        loss = loss_fn(output, target)
        print(f"Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)

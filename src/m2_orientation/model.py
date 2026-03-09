"""
M2: Orientation Model for Meter Reading

Correct architecture matching M2_Orientation.pth checkpoint.
ResNet18 + Custom MLP head for angle regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict


class M2_OrientationModel(nn.Module):
    """
    M2: Angle regression model (ResNet18 + Custom Head)

    Architecture (matching checkpoint):
        Input: (B, 3, 224, 224) - RGB image
          ↓
        ResNet18 backbone (without FC and avgpool)
          → Output: (B, 512, 7, 7)
          ↓
        Flatten
          → Output: (B, 25088)
          ↓
        angle_head.0: [Placeholder - matches checkpoint]
        angle_head.1: Linear(25088, 1024)
        angle_head.2: LayerNorm1d(1024)
        angle_head.3: [ReLU]
        angle_head.4: [Dropout]
        angle_head.5: Linear(1024, 512)
        angle_head.6: LayerNorm1d(512)
        angle_head.7: [ReLU]
        angle_head.8: [Dropout]
        angle_head.9: Linear(512, 2)
          ↓
        Output: (B, 2) - [sin(angle), cos(angle)]
    """

    def __init__(self, dropout=0.4):
        super(M2_OrientationModel, self).__init__()

        # Load ResNet18 backbone (without FC and avgpool)
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (B, 512, H/32, W/32)

        # Angle regression head using OrderedDict to match checkpoint keys exactly
        # Indices 0, 3, 4, 7, 8 are non-parameter layers (ReLU, Dropout)
        angle_head_layers = OrderedDict([
            ('0', nn.Identity()),              # Placeholder
            ('1', nn.Linear(25088, 1024)),     # Linear(25088, 1024)
            ('2', nn.LayerNorm(1024)),         # LayerNorm(1024)
            ('3', nn.ReLU()),                  # ReLU (not saved in checkpoint)
            ('4', nn.Dropout(dropout)),        # Dropout (not saved in checkpoint)
            ('5', nn.Linear(1024, 512)),       # Linear(1024, 512)
            ('6', nn.LayerNorm(512)),          # LayerNorm(512)
            ('7', nn.ReLU()),                  # ReLU (not saved in checkpoint)
            ('8', nn.Dropout(dropout)),        # Dropout (not saved in checkpoint)
            ('9', nn.Linear(512, 2)),          # Linear(512, 2)
        ])

        self.angle_head = nn.Sequential(angle_head_layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (B, 3, H, W) - RGB image

        Returns:
            sin_cos: Normalized 2D vector (sin(angle), cos(angle))
                     Shape: (B, 2)
        """
        # Extract features using ResNet18 backbone
        features = self.backbone(x)  # (B, 512, H/32, W/32)

        # Flatten
        features = features.flatten(start_dim=1)  # (B, 25088)

        # Pass through angle_head
        sin_cos = self.angle_head(features)

        # Normalize to unit vector
        sin_cos = F.normalize(sin_cos, p=2, dim=1)

        return sin_cos


# Test loading
if __name__ == "__main__":
    import sys

    # Fix encoding for Windows
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

    print("M2 Orientation Model Test")
    print("=" * 60)

    MODEL_PATH = r"F:\Workspace\Project\model\M2_Orientation.pth"

    # Create model
    model = M2_OrientationModel(dropout=0.4)

    # Load checkpoint
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        # Use strict=False because ReLU and Dropout layers don't have parameters in checkpoint
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"[OK] Model loaded successfully")
        print(f"      Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"      Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm:  {torch.norm(output, p=2, dim=1).item():.4f}")

    # Test angle conversion
    sin_val = output[0, 0].item()
    cos_val = output[0, 1].item()
    import numpy as np
    angle_rad = np.arctan2(sin_val, cos_val)
    angle_deg = np.degrees(angle_rad)
    angle_deg = (angle_deg + 360) % 360

    print(f"\n  Sin: {sin_val:.4f}, Cos: {cos_val:.4f}")
    print(f"  Angle: {angle_deg:.2f} degrees")

    print("\n" + "=" * 60)
    print("[OK] M2 Orientation Model ready!")
    print("=" * 60)

    print("\nUsage:")
    print("  from src.m2_orientation.model import M2_OrientationModel")
    print("  model = M2_OrientationModel()")
    print("  checkpoint = torch.load('model/M2_Orientation.pth')")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")
    print("=" * 60)

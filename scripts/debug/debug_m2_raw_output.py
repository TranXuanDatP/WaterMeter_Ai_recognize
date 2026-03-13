#!/usr/bin/env python3
"""
Debug M2 raw sin/cos output to understand what model predicts
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
INPUT_DIR = r"F:\Workspace\Project\data\m2_crops"

print("="*70)
print("M2 RAW SIN/COS OUTPUT DEBUG")
print("="*70)

# ====================== M2 MODEL ======================
class M2_OrientationModel(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 512, 1),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.6),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        ca = torch.sigmoid(self.channel_att(x))
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        x = self.gap(x).flatten(1)
        x = self.regressor(x)
        return F.normalize(x, p=2, dim=1)


def sin_cos_to_angle(sin_cos):
    """Convert sin/cos to angle in degrees"""
    sin_val = sin_cos[:, 0]
    cos_val = sin_cos[:, 1]
    angles_rad = torch.atan2(sin_val, cos_val)
    angles_deg = torch.rad2deg(angles_rad)
    angles_deg = (angles_deg + 360) % 360
    return angles_deg


# Load model
print("\nLoading M2 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = M2_OrientationModel().to(device)
checkpoint = torch.load(M2_MODEL, map_location=device)
model.load_state_dict(checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model_state_dict', checkpoint), strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"Model loaded on {device}")

# Test on first 3 images
image_files = list(Path(INPUT_DIR).glob('*.jpg'))[:3]

print(f"\nTesting {len(image_files)} images...\n")

for img_path in image_files:
    filename = os.path.basename(img_path)
    print(f"{'='*70}")
    print(f"File: {filename}")
    print(f"{'='*70}")

    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    # Get raw output
    with torch.no_grad():
        sin_cos = model(tensor)

    # Extract values
    sin_val = sin_cos[0, 0].cpu().item()
    cos_val = sin_cos[0, 1].cpu().item()

    # Current angle calculation
    angle_current = sin_cos_to_angle(sin_cos)[0].cpu().item()

    # Alternative calculations
    angle_swap = torch.rad2deg(torch.atan2(torch.tensor(cos_val), torch.tensor(sin_val))).item() % 360
    angle_inverted = torch.rad2deg(torch.atan2(torch.tensor(-sin_val), torch.tensor(-cos_val))).item() % 360
    angle_sin_cos = torch.rad2deg(torch.atan2(torch.tensor(sin_val), torch.tensor(cos_val))).item() % 360
    angle_cos_sin = torch.rad2deg(torch.atan2(torch.tensor(cos_val), torch.tensor(sin_val))).item() % 360

    print(f"\nRaw sin/cos output:")
    print(f"  sin[0] = {sin_val:+.6f}")
    print(f"  cos[1] = {cos_val:+.6f}")
    print(f"  sqrt(sin^2 + cos^2) = {math.sqrt(sin_val**2 + cos_val**2):.6f} (should be ~1.0)")

    print(f"\nAngle calculations:")
    print(f"  Current (atan2(sin, cos)):     {angle_current:.1f}°")
    print(f"  Swapped (atan2(cos, sin)):     {angle_swap:.1f}°")
    print(f"  Inverted (-sin, -cos):         {angle_inverted:.1f}°")
    print(f"  Using atan2(sin, cos) raw:     {angle_sin_cos:.1f}°")
    print(f"  Using atan2(cos, sin) raw:     {angle_cos_sin:.1f}°")

    # What angle SHOULD produce upright (0°)?
    # If image is upright (0°), model should output angle that when negated gives 0
    print(f"\nIf image is upright (0°):")
    print(f"  Model should predict: ~0° (so rotate by -0° = 0°)")
    print(f"  Model actually predicts: {angle_current:.1f}°")
    print(f"  Error: {abs(angle_current):.1f}°")

    # Check if swapping sin/cos helps
    print(f"\nIf we SWAP sin/cos:")
    print(f"  Predicted angle: {angle_swap:.1f}°")
    print(f"  Error from 0°: {abs(angle_swap):.1f}°")

    # Expected angles for common rotations
    print(f"\nReference angles:")
    print(f"  0° = 0.0 rad")
    print(f"  90° = π/2 = 1.571 rad")
    print(f"  180° = π = 3.142 rad")
    print(f"  270° = 3π/2 = 4.712 rad")

    print()

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print("For UPRIGHT images (0°), the model should predict angles close to 0°.")
print("If the model consistently predicts ~250-270° for upright images,")
print("there may be a training label issue or model output interpretation issue.")
print("="*70)

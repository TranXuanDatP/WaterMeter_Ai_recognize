#!/usr/bin/env python3
"""
Fixed M2 Rotation Test - Correct rotation direction
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
INPUT_DIR = r"F:\Workspace\Project\data\m2_crops"
OUTPUT_DIR = r"F:\Workspace\Project\results\m2_fixed_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("M2 ROTATION FIX TEST")
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


def rotate_image(image, angle):
    """Rotate image by given angle (degrees)"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))

    return rotated


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

print(f"\nTesting {len(image_files)} images...")

for img_path in image_files:
    filename = os.path.basename(img_path)
    print(f"\n{'='*70}")
    print(f"Testing: {filename}")
    print(f"{'='*70}")

    # Load image
    img = cv2.imread(str(img_path))

    # Predict angle
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sin_cos = model(tensor)

    angle = sin_cos_to_angle(sin_cos)[0].cpu().item()

    print(f"Predicted angle: {angle:.1f}")

    # Test THREE rotation strategies:
    # 1. Rotate by -angle (current/WRONG)
    # 2. Rotate by +angle (maybe correct?)
    # 3. Rotate by (360-angle) if angle > 180 (alternative fix)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original (0°)')
    axes[0, 0].axis('off')

    # Strategy 1: Rotate by -angle (current method)
    rot1 = rotate_image(img, -angle)
    axes[0, 1].imshow(cv2.cvtColor(rot1, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Method 1: Rotate by -{angle:.1f}° (CURRENT)')
    axes[0, 1].axis('off')

    # Strategy 2: Rotate by +angle
    rot2 = rotate_image(img, angle)
    axes[1, 0].imshow(cv2.cvtColor(rot2, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Method 2: Rotate by +{angle:.1f}° (TEST)')
    axes[1, 0].axis('off')

    # Strategy 3: Rotate by (360-angle) if angle > 180 else -angle
    if angle > 180:
        rot_angle = -(360 - angle)
        method_name = f"Rotate by ({360:.0f}-{angle:.1f}) = {rot_angle:.1f}°"
    else:
        rot_angle = -angle
        method_name = f"Rotate by -{angle:.1f}°"
    rot3 = rotate_image(img, rot_angle)
    axes[1, 1].imshow(cv2.cvtColor(rot3, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Method 3: {method_name}')
    axes[1, 1].axis('off')

    plt.suptitle(f'Rotation Strategy Comparison - {filename}\nPredicted Angle: {angle:.1f}°', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"fixed_{filename}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")
    print("\n  Which method looks upright?")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print(f"\nCheck outputs at: {OUTPUT_DIR}")
print("\nCompare the 4 methods:")
print("  - Original: Should be upright (images from M1)")
print("  - Method 1 (-angle): Currently makes images tilted [WRONG]")
print("  - Method 2 (+angle): Test if positive rotation fixes it")
print("  - Method 3 (adjusted): Alternative correction method")

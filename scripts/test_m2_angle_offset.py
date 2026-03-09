#!/usr/bin/env python3
"""
Test M2 with different angle offsets to find correct formula
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

sys.path.insert(0, str(Path(__file__).parent.parent))

M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
INPUT_DIR = r"F:\Workspace\Project\data\m2_crops"
OUTPUT_DIR = r"F:\Workspace\Project\results\m2_angle_offset_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("M2 ANGLE OFFSET TEST")
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

# Test on first image
image_files = list(Path(INPUT_DIR).glob('*.jpg'))[:1]

for img_path in image_files:
    filename = os.path.basename(img_path)
    print(f"\nTesting: {filename}")

    img = cv2.imread(str(img_path))

    # Predict angle
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sin_cos = model(tensor)

    raw_angle = sin_cos_to_angle(sin_cos)[0].cpu().item()

    print(f"Raw predicted angle: {raw_angle:.1f}°")

    # Test different angle interpretations
    # Expected: upright images should have corrected angle ≈ 0°

    angle_formulas = {
        'Original (no rotation)': 0,
        f'Raw prediction: {raw_angle:.1f}°': raw_angle,
        f'Negative: -{raw_angle:.1f}°': -raw_angle,
        f'Add 180: {raw_angle:.1f} + 180 = {(raw_angle + 180) % 360:.1f}°': (raw_angle + 180) % 360,
        f'Subtract 180: {raw_angle:.1f} - 180 = {(raw_angle - 180) % 360:.1f}°': (raw_angle - 180) % 360,
        f'360 - angle: {360 - raw_angle:.1f}°': 360 - raw_angle,
        f'Rotate BY angle (not TO): -{(raw_angle + 180) % 360:.1f}°': -((raw_angle + 180) % 360),
        f'Invert sin/cos: 180 - {raw_angle:.1f} = {180 - raw_angle:.1f}°': 180 - raw_angle,
    }

    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for idx, (label, angle) in enumerate(angle_formulas.items()):
        if idx >= len(axes):
            break

        rotated = rotate_image(img, angle)
        axes[idx].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(label, fontsize=10)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(angle_formulas), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Testing Angle Correction Formulas\nRaw Prediction: {raw_angle:.1f}°\nGoal: Find formula that keeps image upright',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"offset_test_{filename}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")

    # Print analysis
    print("\n" + "="*70)
    print("ANGLE FORMULA ANALYSIS")
    print("="*70)
    print("Since the original image is already upright (0°),")
    print("the CORRECT formula should produce an upright image.")
    print("\nLook at the visualization and find which formula gives upright result.")
    print("="*70)

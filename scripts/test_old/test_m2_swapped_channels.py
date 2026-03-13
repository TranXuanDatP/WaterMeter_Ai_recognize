#!/usr/bin/env python3
"""
Test if M2 model's sin/cos channels are swapped
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
OUTPUT_DIR = r"F:\Workspace\Project\results\m2_channel_swap_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("M2 CHANNEL SWAP TEST")
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

    # Extract raw values
    ch0 = sin_cos[0, 0].cpu().item()  # Channel 0
    ch1 = sin_cos[0, 1].cpu().item()  # Channel 1

    print(f"\nRaw model output:")
    print(f"  channel[0] = {ch0:+.6f}  (assuming sin)")
    print(f"  channel[1] = {ch1:+.6f}  (assuming cos)")

    # Test different angle interpretations
    angles = {}

    # Method 1: channel[0]=sin, channel[1]=cos (as labeled)
    ang1 = (torch.rad2deg(torch.atan2(torch.tensor(ch0), torch.tensor(ch1))) % 360).item()
    angles['Method 1: atan2(ch0, ch1)\n[ch0=sin, ch1=cos]'] = (ang1, -ang1)

    # Method 2: channel[0]=cos, channel[1]=sin (SWAPPED)
    ang2 = (torch.rad2deg(torch.atan2(torch.tensor(ch1), torch.tensor(ch0))) % 360).item()
    angles['Method 2: atan2(ch1, ch0)\n[ch0=cos, ch1=sin] SWAPPED'] = (ang2, -ang2)

    # Method 3: Use -ch0, -ch1
    ang3 = (torch.rad2deg(torch.atan2(torch.tensor(-ch0), torch.tensor(-ch1))) % 360).item()
    angles['Method 3: atan2(-ch0, -ch1)\n[Negated]'] = (ang3, -ang3)

    # Method 4: Swapped AND negated
    ang4 = (torch.rad2deg(torch.atan2(torch.tensor(-ch1), torch.tensor(-ch0))) % 360).item()
    angles['Method 4: atan2(-ch1, -ch0)\n[Swapped + Negated]'] = (ang4, -ang4)

    # Method 5: Add 180 to method 1
    ang5 = (ang1 + 180) % 360
    angles['Method 5: Method1 + 180\n[Offset]'] = (ang5, -ang5)

    # Method 6: Subtract 180 from method 1
    ang6 = (ang1 - 180) % 360
    angles['Method 6: Method1 - 180\n[Offset]'] = (ang6, -ang6)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Original (reference)
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original\n(Should stay upright)', fontweight='bold')
    axes[0].axis('off')

    for idx, (label, (angle, rot_angle)) in enumerate(angles.items()):
        if idx >= len(axes) - 1:
            break
        ax = axes[idx + 1]

        rotated = rotate_image(img, rot_angle)
        ax.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{label}\nAngle: {angle:.1f}, Rotate: {rot_angle:.1f}',
                    fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Channel Swap Test - {filename}\nGoal: Find method that keeps image upright',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"swap_test_{filename}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {save_path}")
    print("\nAngle summary:")
    for label, (angle, rot_angle) in angles.items():
        print(f"  {label}")
        print(f"    Predicted: {angle:.1f}, Rotate: {rot_angle:.1f}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("The CORRECT method should:")
    print("  1. Predict angle close to 0 for upright images")
    print("  2. Keep the image upright after rotation")
    print("\nLook at the visualization to identify which method works!")
    print("="*70)

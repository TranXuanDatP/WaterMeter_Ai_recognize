#!/usr/bin/env python3
"""
Definitive M2 test with proper seeding for consistency
"""

import os
import sys
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(Path(__file__).parent.parent))

M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
INPUT_DIR = r"F:\Workspace\Project\data\m2_crops"
OUTPUT_DIR = r"F:\Workspace\Project\results\m2_definitive_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("DEFINITIVE M2 TEST (with seed=42 for reproducibility)")
print("="*70)


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
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(rotation_matrix[0, 0]), np.abs(rotation_matrix[0, 1])
    new_w, new_h = int((height * sin) + (width * cos)), int((height * cos) + (width * sin))
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    return rotated


def sin_cos_to_angle(sin_cos):
    """Convert sin/cos to angle in degrees (ORIGINAL method)"""
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
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict, strict=False)
model.eval()  # Important: disables dropout

print(f"Model loaded on {device}")
print(f"Model is in eval mode: {not model.training}")
print(f"Dropout disabled in eval mode")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test on 3 images
image_files = sorted(list(Path(INPUT_DIR).glob('*.jpg')))[:3]

print(f"\nTesting {len(image_files)} images...")

results = []

for idx, img_path in enumerate(image_files):
    filename = os.path.basename(img_path)
    print(f"\n{'='*70}")
    print(f"[{idx+1}/{len(image_files)}] {filename}")
    print(f"{'='*70}")

    # Load image
    img = cv2.imread(str(img_path))

    # Predict angle
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sin_cos = model(tensor)

    # Get angle using ORIGINAL method
    angle = sin_cos_to_angle(sin_cos)[0].cpu().item()

    # Rotate to correct
    corrected = rotate_image(img, -angle)

    print(f"Predicted angle: {angle:.1f}")
    print(f"Rotation applied: {-angle:.1f}")

    # Save corrected image
    out_path = os.path.join(OUTPUT_DIR, f"corrected_{filename}")
    cv2.imwrite(out_path, corrected)

    results.append({
        'filename': filename,
        'angle': angle,
        'corrected_path': out_path
    })

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Corrected\nAngle: {angle:.1f}, Rotate: {-angle:.1f}', fontweight='bold')
    axes[1].axis('off')

    plt.suptitle(f'{filename}', fontsize=14)
    plt.tight_layout()

    viz_path = os.path.join(OUTPUT_DIR, f"viz_{filename}")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {out_path}")
    print(f"Visualization: {viz_path}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Filename':<50} {'Angle':>10} {'Status':>15}")
print("-"*70)

for r in results:
    angle = r['angle']
    if abs(angle) < 15 or abs(angle - 360) < 15:
        status = "GOOD (upright)"
    elif abs(angle) < 45 or abs(angle - 360) < 45:
        status = "OK (small rotation)"
    else:
        status = "CHECK (large rotation)"

    print(f"{r['filename']:<50} {angle:>9.1f}° {status:>15}")

print("="*70)
print("\nKEY FINDING:")
print("  - Model predictions should be close to 0° for upright images")
print("  - If predictions are > 45°, images might not be upright")
print("  - Or model might need angle adjustment")
print("\nCheck visualizations to verify correctness!")
print("="*70)
print(f"\nOutputs saved to: {OUTPUT_DIR}")

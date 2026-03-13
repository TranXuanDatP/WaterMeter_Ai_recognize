#!/usr/bin/env python3
"""
Verify M2 fix with visualization
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

print("="*70)
print("VERIFY M2 FIX")
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


# Load model
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

# Test on one image
img_path = list(Path(INPUT_DIR).glob('*.jpg'))[0]
filename = os.path.basename(img_path)
img = cv2.imread(str(img_path))

pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
tensor = transform(pil_img).unsqueeze(0).to(device)

with torch.no_grad():
    sin_cos = model(tensor)

ch0 = sin_cos[0, 0].cpu().item()
ch1 = sin_cos[0, 1].cpu().item()

print(f"\nRaw model output:")
print(f"  ch0 = {ch0:+.6f}")
print(f"  ch1 = {ch1:+.6f}")

# Test calculations
print(f"\nAngle calculations:")

# Method that should work (from test)
ang_test = (torch.rad2deg(torch.atan2(torch.tensor(-ch1), torch.tensor(-ch0))) % 360).item()
print(f"  atan2(-ch1, -ch0) = {ang_test:.1f}")

# Try tensor-based calculation
sin_val = -sin_cos[:, 1]
cos_val = -sin_cos[:, 0]
ang_tensor = sin_cos_to_angle = torch.rad2deg(torch.atan2(sin_val, cos_val)) % 360
print(f"  Using tensor negation: {ang_tensor[0].item():.1f}")

# Try element-wise
sin_elem = -sin_cos[0, 1]
cos_elem = -sin_cos[0, 0]
ang_elem = (torch.rad2deg(torch.atan2(sin_elem, cos_elem)) % 360).item()
print(f"  Element-wise: {ang_elem:.1f}")

# Rotate and visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original\n(Upright)', fontweight='bold')
axes[0].axis('off')

# Use the angle from test (8.1°)
rot_test = rotate_image(img, -ang_test)
axes[1].imshow(cv2.cvtColor(rot_test, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Using test angle\n{ang_test:.1f}, Rotate: -{ang_test:.1f}', fontweight='bold', color='green')
axes[1].axis('off')

# Use tensor-based
rot_tensor = rotate_image(img, -ang_tensor[0].item())
axes[2].imshow(cv2.cvtColor(rot_tensor, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'Using tensor-based\n{ang_tensor[0].item():.1f}, Rotate: -{ang_tensor[0].item():.1f}')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('F:/Workspace/Project/results/m2_verify_fix.png', dpi=150)
plt.close()

print(f"\nVisualization saved.")
print("\nWhich rotation keeps the image upright?")

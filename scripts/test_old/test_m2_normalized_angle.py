"""
Test M2 with (-180, 180] angle range instead of (0, 360]
"""
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from pathlib import Path
import pandas as pd

MODEL_PATH = r"F:\Workspace\Project\model\m2_angle_model_best (2).pth"
M1_CROPS_DIR = r"F:\Workspace\Project\results\test_pipeline\m1_crops"
METADATA_PATH = r"F:\Workspace\Project\results\test_pipeline\m2_aligned\metadata.csv"

class M2_OrientationModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.angle_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 1024),
            nn.GroupNorm(32, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
            nn.Tanh()
        )

    def forward(self, x):
        feats = self.backbone(x)
        vec = self.angle_head(feats)
        return vec

print("="*70)
print("M2 TEST WITH (-180, 180] ANGLE RANGE")
print("="*70)

# Load model
device = torch.device('cpu')
model = M2_OrientationModel(pretrained=False).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load metadata
metadata_df = pd.read_csv(METADATA_PATH)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get first 10 images
m1_dir = Path(M1_CROPS_DIR)
image_files = sorted(list(m1_dir.glob("*.jpg")) + list(m1_dir.glob("*.png")))[:10]

print(f"\nProcessing {len(image_files)} images...\n")

results = []
for i, img_path in enumerate(image_files, 1):
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # Predict with (-180, 180] range
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        vec = model(tensor)

    cos_val = vec[0, 0].item()
    sin_val = vec[0, 1].item()
    angle_rad = np.arctan2(sin_val, cos_val)  # Already in (-pi, pi]
    angle_deg = np.degrees(angle_rad)  # (-180, 180] - NO % 360!

    # Get metadata
    fname = img_path.name
    meta_row = metadata_df[metadata_df['filename'] == fname]
    if len(meta_row) > 0:
        meta_angle = meta_row.iloc[0]['angle']  # (0, 360]

        # Convert metadata to (-180, 180]
        if meta_angle > 180:
            meta_angle_180 = meta_angle - 360
        else:
            meta_angle_180 = meta_angle

        # Calculate difference
        diff = abs(angle_deg - meta_angle_180)

        print(f"{i:2d}. {fname[:50]}...")
        print(f"    Predicted: {angle_deg:7.2f}°, Metadata: {meta_angle:7.2f}° ({meta_angle_180:7.2f}°), Diff: {diff:5.2f}°")

        results.append({
            'filename': fname,
            'predicted_180': angle_deg,
            'predicted_360': (angle_deg + 360) % 360,
            'metadata_360': meta_angle,
            'metadata_180': meta_angle_180,
            'diff': diff
        })

# Statistics
pred_angles_180 = [r['predicted_180'] for r in results]
pred_angles_360 = [r['predicted_360'] for r in results]

print(f"\n{'='*70}")
print(f"STATISTICS")
print(f"{'='*70}")
print(f"\nWith (-180, 180] range:")
print(f"  Mean:     {np.mean(pred_angles_180):.2f}°")
print(f"  Std:      {np.std(pred_angles_180):.2f}°")
print(f"  Min:      {np.min(pred_angles_180):.2f}°")
print(f"  Max:      {np.max(pred_angles_180):.2f}°")

print(f"\nWith (0, 360] range:")
print(f"  Mean:     {np.mean(pred_angles_360):.2f}°")
print(f"  Std:      {np.std(pred_angles_360):.2f}°")
print(f"  Min:      {np.min(pred_angles_360):.2f}°")
print(f"  Max:      {np.max(pred_angles_360):.2f}°")

avg_error = np.mean([r['diff'] for r in results])
print(f"\nAverage error vs metadata: {avg_error:.2f}°")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("Using (-180, 180] range gives cleaner correction angles!")
print("Example: 356.74° -> -3.26° -> correction +3.26° (shortest path)")

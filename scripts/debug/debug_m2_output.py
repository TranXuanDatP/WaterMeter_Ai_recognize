"""
Debug M2 model raw output
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

MODEL_PATH = r"F:\Workspace\Project\model\m2_angle_model_best (2).pth"
TEST_IMAGE = r"F:\Workspace\Project\results\test_pipeline\m1_crops\meter4_00000_validate_00000_00385501ab4d419fa7b0bdf0d9f8451f.jpg"

class M2_OrientationModel(nn.Module):
    """M2 with Tanh"""
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
            nn.Tanh()  # ← IMPORTANT!
        )

    def forward(self, x):
        feats = self.backbone(x)
        vec = self.angle_head(feats)
        return vec

print("="*70)
print("DEBUG M2 MODEL OUTPUT")
print("="*70)

# Load model
device = torch.device('cpu')
model = M2_OrientationModel(pretrained=False).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and transform image
img = cv2.imread(TEST_IMAGE)
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tensor = transform(pil_img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    vec = model(tensor)

print(f"\nRaw model output:")
print(f"  vec[0, 0]: {vec[0, 0].item():.6f}")
print(f"  vec[0, 1]: {vec[0, 1].item():.6f}")
print(f"  Magnitude: {torch.norm(vec[0]).item():.6f}")

# Method 1: Direct arctan2
cos_val = vec[0, 0].item()
sin_val = vec[0, 1].item()
angle_rad = np.arctan2(sin_val, cos_val)
angle_deg = np.degrees(angle_rad) % 360

print(f"\nMethod 1: Direct arctan2 (with Tanh output)")
print(f"  cos: {cos_val:.6f}, sin: {sin_val:.6f}")
print(f"  Angle: {angle_deg:.2f}°")

# Method 2: Normalize first (if needed)
vec_normalized = vec / torch.norm(vec, dim=1, keepdim=True)
cos_norm = vec_normalized[0, 0].item()
sin_norm = vec_normalized[0, 1].item()
angle_rad_norm = np.arctan2(sin_norm, cos_norm)
angle_deg_norm = np.degrees(angle_rad_norm) % 360

print(f"\nMethod 2: Normalize then arctan2")
print(f"  cos: {cos_norm:.6f}, sin: {sin_norm:.6f}")
print(f"  Angle: {angle_deg_norm:.2f}°")

print(f"\nDifference: {abs(angle_deg - angle_deg_norm):.2f}°")

print(f"\nMetadata expected: 23.12°")
print(f"  Method 1 error: {abs(angle_deg - 23.12):.2f}°")
print(f"  Method 2 error: {abs(angle_deg_norm - 23.12):.2f}°")

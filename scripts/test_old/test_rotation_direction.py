"""
Test rotation direction for M2 alignment

This script tests different rotation strategies to find the correct one.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights

# ============================================
# MODEL (Same as test_m2_model.py)
# ============================================

class AngleRegressionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(AngleRegressionModel, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
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
        features = self.backbone(x)
        angle = self.angle_head(features)
        return angle

def sin_cos_to_angle(sin_val, cos_val):
    angle_rad = np.arctan2(sin_val, cos_val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg % 360

def smart_rotate(image, angle):
    """Rotate image with smart cropping"""
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle)
    new_w = int(h * abs(np.sin(angle_rad)) + w * abs(np.cos(angle_rad)))
    new_h = int(h * abs(np.cos(angle_rad)) + w * abs(np.sin(angle_rad)))
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ============================================
# TEST
# ============================================

INPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m1_crops")
MODEL_PATH = Path(r"F:\Workspace\Project\model\M2_Orientation.pth")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\rotation_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = AngleRegressionModel(pretrained=False).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get first image
image_files = sorted(list(INPUT_DIR.glob('*.jpg')))[:3]

print("=" * 80)
print("ROTATION DIRECTION TEST")
print("=" * 80)
print("Testing 3 different rotation strategies:")
print("1. correction = -angle (rotate opposite to detected angle)")
print("2. correction = angle (rotate same as detected angle)")
print("3. correction = -(angle - 180) for angle > 180, else -angle")
print("=" * 80)

for img_path in image_files:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std
    img_tensor = torch.from_numpy(img_normalized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)

    cos_val = output[0, 0].cpu().item()
    sin_val = output[0, 1].cpu().item()
    angle = sin_cos_to_angle(sin_val, cos_val)

    print(f"\nImage: {img_path.name}")
    print(f"Detected angle: {angle:.1f}° (cos: {cos_val:.3f}, sin: {sin_val:.3f})")

    # Strategy 1: correction = -angle
    if angle <= 180:
        corr1 = -angle
    else:
        corr1 = -(angle - 360)
    result1 = smart_rotate(img, corr1)

    # Strategy 2: correction = angle (same direction)
    result2 = smart_rotate(img, angle)

    # Strategy 3: Alternative interpretation
    if angle > 180:
        corr3 = -(angle - 180)
    else:
        corr3 = angle
    result3 = smart_rotate(img, corr3)

    # Create comparison
    h = 400
    img_v = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    r1_v = cv2.resize(result1, (int(result1.shape[1] * h / result1.shape[0]), h))
    r2_v = cv2.resize(result2, (int(result2.shape[1] * h / result2.shape[0]), h))
    r3_v = cv2.resize(result3, (int(result3.shape[1] * h / result3.shape[0]), h))

    comp = np.hstack([img_v, r1_v, r2_v, r3_v])

    # Add labels
    cv2.putText(comp, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comp, f"Angle: {angle:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    w1 = img_v.shape[1]
    cv2.putText(comp, "STRATEGY 1", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(comp, f"Corr: {corr1:.1f}", (w1 + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    w2 = w1 + r1_v.shape[1]
    cv2.putText(comp, "STRATEGY 2", (w2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(comp, f"Corr: {angle:.1f}", (w2 + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    w3 = w2 + r2_v.shape[1]
    cv2.putText(comp, "STRATEGY 3", (w3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(comp, f"Corr: {corr3:.1f}", (w3 + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    base_name = img_path.stem
    cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_rotation_test.jpg"), comp)
    print(f"  Saved: {base_name}_rotation_test.jpg")

print("\n" + "=" * 80)
print("Check the comparison images to see which strategy aligns correctly!")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

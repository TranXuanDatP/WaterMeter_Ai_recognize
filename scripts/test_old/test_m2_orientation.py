"""
M2 Orientation Test - Test trained model with M1 crops
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# Config
INPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m1_crops")
MODEL_PATH = Path(r"F:\Workspace\Project\model\M2_Orientation.pth")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m2_test_results")
NUM_SAMPLES = 10
IMG_SIZE = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("M2 ORIENTATION TEST")
print("=" * 80)
print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print(f"Samples: {NUM_SAMPLES}")
print("=" * 80)

# ============================================
# MODEL DEFINITION
# ============================================

class M2OrientationModel(nn.Module):
    def __init__(self):
        super(M2OrientationModel, self).__init__()
        
        # Backbone: ResNet18 (partial, up to layer4)
        # Using torchvision.models.resnet18 structure
        from torchvision.models import resnet18, ResNet18_Weights
        
        # Load pretrained ResNet18 (partial)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Extract layers we need (conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        self.avgpool = resnet.avgpool
        
        # Angle head (matching checkpoint structure)
        self.angle_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # sin, cos output
        )
    
    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        # Angle prediction
        angle = self.angle_head(x)
        return angle

# ============================================
# UTILS
# ============================================

def sin_cos_to_angle(sin_val, cos_val):
    angle_rad = np.arctan2(sin_val, cos_val)
    return np.degrees(angle_rad) % 360

def smart_rotate(image, angle):
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle)
    
    new_w = int(h * abs(np.sin(angle_rad)) + w * abs(np.cos(angle_rad)))
    new_h = int(h * abs(np.cos(angle_rad)) + w * abs(np.sin(angle_rad)))
    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# ============================================
# LOAD MODEL
# ============================================

print(f"\n[1/4] Loading model...")
model = M2OrientationModel().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"      ✓ Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})")

# ============================================
# LOAD IMAGES
# ============================================

print(f"\n[2/4] Loading images...")
image_files = sorted(list(INPUT_DIR.glob('*.jpg')))[:NUM_SAMPLES]

if len(image_files) == 0:
    print(f"      ERROR: No images found in {INPUT_DIR}")
    sys.exit(1)

print(f"      Found {len(image_files)} images")

# ============================================
# PROCESS
# ============================================

print(f"\n[3/4] Running inference...")
results = []

for img_path in tqdm(image_files, desc="Testing", leave=False):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    h, w = img.shape[:2]
    
    # Preprocess for ResNet18
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std
    
    # Convert to tensor: (H,W,C) -> (C,H,W)
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)  # (1, 2)
    
    sin_val = output[0, 0].cpu().item()
    cos_val = output[0, 1].cpu().item()
    angle = sin_cos_to_angle(sin_val, cos_val)
    correction = -angle
    
    # Rotate
    aligned = smart_rotate(img, correction)
    
    # Save results
    base_name = img_path.stem
    cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_1_original.jpg"), img)
    cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_2_aligned.jpg"), aligned)
    
    # Comparison
    h1, w1 = img.shape[:2]
    h2, w2 = aligned.shape[:2]
    max_h = max(h1, h2)
    
    img_v = cv2.resize(img, (int(w1 * max_h / h1), max_h))
    alg_v = cv2.resize(aligned, (int(w2 * max_h / h2), max_h))
    
    comp = np.hstack([img_v, alg_v])
    cv2.putText(comp, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comp, f"{angle:.1f} deg", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(comp, "ALIGNED", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comp, f"corrected: {correction:.1f}", (w1 + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_3_comparison.jpg"), comp)
    
    results.append({
        'filename': img_path.name,
        'angle': angle,
        'correction': correction,
        'original': f'{w}x{h}',
        'aligned': f'{aligned.shape[1]}x{aligned.shape[0]}'
    })

# ============================================
# RESULTS
# ============================================

print(f"\n[4/4] Results:")
print("-" * 80)

for i, r in enumerate(results, 1):
    print(f"  {i:2d}. {r['filename'][:40]:40s}")
    print(f"      {r['original']} → {r['aligned']}")
    print(f"      Angle: {r['angle']:6.1f}° → Correction: {r['correction']:6.1f}°")

angles = [r['angle'] for r in results]
corrections = [abs(r['correction']) for r in results]

print("\nStatistics:")
print(f"  Angle range:    {min(angles):.1f}° to {max(angles):.1f}°")
print(f"  Correction avg: {np.mean(corrections):.1f}°")

print(f"\n✅ Results saved to: {OUTPUT_DIR}")
print("=" * 80)

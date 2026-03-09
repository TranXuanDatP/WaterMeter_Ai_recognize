"""
Simple M2 Orientation Test - Load checkpoint and test
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================

INPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m1_crops")
MODEL_PATH = Path(r"F:\Workspace\Project\model\M2_Orientation.pth")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m2_test_results")
NUM_SAMPLES = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224

print("=" * 80)
print("M2: ORIENTATION ALIGNMENT TEST")
print("=" * 80)
print(f"Input:  {INPUT_DIR}")
print(f"Model:  {MODEL_PATH}")
print(f"Samples: {NUM_SAMPLES}")
print("=" * 80)

# ============================================
# DEFINE MODEL (inline)
# ============================================

class M2AngleRegressor(nn.Module):
    def __init__(self, num_classes=2):
        super(M2AngleRegressor, self).__init__()
        
        # ResNet18 backbone
        resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        
        # Remove FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adaptive pool
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        features = self.cnn(x)
        pooled = self.adaptive_pool(features)
        flattened = torch.flatten(pooled, 1)
        output = self.fc(flattened)
        return output

def sin_cos_to_angle(sin_val, cos_val):
    """Convert sin/cos to angle in degrees"""
    angle_rad = np.arctan2(sin_val, cos_val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg % 360

# ============================================
# LOAD MODEL
# ============================================

print(f"\n[LOAD] Loading M2 model...")

try:
    model = M2AngleRegressor(num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"[LOAD] ✓ Model loaded")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# ============================================
# SMART ROTATION
# ============================================

def smart_rotate(image, angle):
    """Rotate with smart cropping"""
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle)
    
    new_w = int(h * abs(np.sin(angle_rad)) + w * abs(np.cos(angle_rad)))
    new_h = int(h * abs(np.cos(angle_rad)) + w * abs(np.sin(angle_rad)))
    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated

# ============================================
# PROCESS
# ============================================

image_files = sorted(list(INPUT_DIR.glob('*.jpg')))[:NUM_SAMPLES]

print(f"\n[SCAN] Found {len(image_files)} images")

if len(image_files) == 0:
    print(f"[ERROR] No images in {INPUT_DIR}")
    sys.exit(1)

results = []

print(f"\n[PROCESS] Testing...")
for img_path in tqdm(image_files, desc="M2 Test"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    h, w = img.shape[:2]
    
    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
    
    sin_val = output[0, 0].cpu().item()
    cos_val = output[0, 1].cpu().item()
    angle = sin_cos_to_angle(sin_val, cos_val)
    correction = -angle
    
    # Rotate
    aligned = smart_rotate(img, correction)
    
    # Save
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
    cv2.putText(comp, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comp, f"{angle:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comp, "ALIGNED", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comp, f"{correction:.1f}", (w1 + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_3_compare.jpg"), comp)
    
    results.append({
        'filename': img_path.name,
        'angle': angle,
        'correction': correction
    })

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

for i, r in enumerate(results, 1):
    print(f"  {i:2d}. {r['filename'][:45]:45s}")
    print(f"      Angle: {r['angle']:6.1f}° → Corrected: {r['correction']:6.1f}°")

print(f"\n✅ Saved to: {OUTPUT_DIR}")
print("=" * 80)

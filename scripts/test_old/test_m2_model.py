"""
M2 Orientation Model Test - Using trained model from Colab

This script uses the actual trained M2_Orientation.pth model
to predict rotation angles and apply smart rotation to M1 crops.

Usage:
    python test_m2_model.py

Input:
    F:\\Workspace\\Project\\results\\test_pipeline\\m1_crops

Output:
    F:\\Workspace\\Project\\results\\test_pipeline\\m2_aligned
    F:\\Workspace\\Project\\results\\test_pipeline\\m2_test_results
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

# Import smart_rotate from utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.image_rotation import smart_rotate

# Fix encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ============================================
# CONFIGURATION
# ============================================

INPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m1_crops")
MODEL_PATH = Path(r"F:\Workspace\Project\model\M2_Orientation.pth")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m2_aligned")
TEST_RESULTS_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m2_test_results")
NUM_SAMPLES = None  # Set to None to process all images
IMG_SIZE = 224

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("M2 ORIENTATION MODEL TEST")
print("=" * 80)
print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print(f"Samples: {NUM_SAMPLES if NUM_SAMPLES else 'ALL'}")
print("=" * 80)

# ============================================
# MODEL DEFINITION (Matching Colab Architecture)
# ============================================

class AngleRegressionModel(nn.Module):
    """
    Exact architecture from M2_Angle_Training_AutoDrive.ipynb

    Backbone: ResNet18 (partial)
    Output: (cos, sin) normalized vector
    Loss: Cosine Similarity Loss
    """
    def __init__(self, pretrained=True):
        super(AngleRegressionModel, self).__init__()

        # Load ResNet18
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Extract backbone (remove FC layer and avgpool)
        # backbone: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Angle head (matching Colab structure exactly)
        # Input: 512*7*7 = 25088 features after backbone
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
            nn.Linear(512, 2),  # Output: (cos, sin)
            nn.Tanh()  # Constrain to [-1, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, 224, 224)

        Returns:
            angle: (B, 2) tensor with (cos, sin) values
        """
        features = self.backbone(x)  # (B, 512, 7, 7)
        angle = self.angle_head(features)  # (B, 2)
        return angle

# ============================================
# UTILITIES
# ============================================

def sin_cos_to_angle(sin_val, cos_val):
    """
    Convert sin/cos representation to angle in degrees

    Args:
        sin_val: Sine value
        cos_val: Cosine value

    Returns:
        Angle in degrees [0, 360)
    """
    angle_rad = np.arctan2(sin_val, cos_val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg % 360

# ============================================
# LOAD MODEL
# ============================================

print(f"\n[1/4] Loading model...")

if not MODEL_PATH.exists():
    print(f"      ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

model = AngleRegressionModel(pretrained=False).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"      ✓ Model loaded")
print(f"      Epoch: {checkpoint['epoch']}")
print(f"      Val Loss: {checkpoint['val_loss']:.6f}")
print(f"      Train Loss: {checkpoint.get('train_loss', 'N/A')}")

# ============================================
# LOAD IMAGES
# ============================================

print(f"\n[2/4] Loading images...")

image_files = sorted(list(INPUT_DIR.glob('*.jpg')) + list(INPUT_DIR.glob('*.png')))

if len(image_files) == 0:
    print(f"      ERROR: No images found in {INPUT_DIR}")
    sys.exit(1)

if NUM_SAMPLES:
    image_files = image_files[:NUM_SAMPLES]

print(f"      Found {len(image_files)} images to process")

# ============================================
# PROCESS IMAGES
# ============================================

print(f"\n[3/4] Running inference...")

results = []
success_count = 0
error_count = 0

for img_path in tqdm(image_files, desc="Testing"):
    try:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            error_count += 1
            continue

        h, w = img.shape[:2]
        original_size = f"{w}x{h}"

        # Preprocess for ResNet18
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std

        # Convert to tensor: (H,W,C) -> (C,H,W)
        img_tensor = torch.from_numpy(img_normalized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)  # (1, 2)

        cos_val = output[0, 0].cpu().item()
        sin_val = output[0, 1].cpu().item()
        angle = sin_cos_to_angle(sin_val, cos_val)

        # Calculate correction angle: rotate opposite to detected angle
        # Normalize to [-180, 180] range for minimal rotation
        correction_angle = -angle
        if correction_angle <= -180:
            correction_angle += 360
        elif correction_angle > 180:
            correction_angle -= 360

        # Skip if correction is very small (already upright)
        if abs(correction_angle) < 1.0:
            aligned = img
            aligned_size = original_size
            rotation_applied = False
        else:
            # Rotate and crop back to original size to maintain focus on meter
            aligned = smart_rotate(img, correction_angle, crop_to_original=True)
            aligned_size = f"{aligned.shape[1]}x{aligned.shape[0]}"
            rotation_applied = True

        # Save aligned image
        output_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(output_path), aligned)

        # Save test visualization
        base_name = img_path.stem
        cv2.imwrite(str(TEST_RESULTS_DIR / f"{base_name}_1_original.jpg"), img)
        cv2.imwrite(str(TEST_RESULTS_DIR / f"{base_name}_2_aligned.jpg"), aligned)

        # Create comparison image
        h1, w1 = img.shape[:2]
        h2, w2 = aligned.shape[:2]
        max_h = max(h1, h2)

        img_v = cv2.resize(img, (int(w1 * max_h / h1), max_h))
        alg_v = cv2.resize(aligned, (int(w2 * max_h / h2), max_h))

        comp = np.hstack([img_v, alg_v])
        cv2.putText(comp, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comp, f"Angle: {angle:.1f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(comp, "ALIGNED", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comp, f"Correction: {correction_angle:.1f}", (w1 + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(str(TEST_RESULTS_DIR / f"{base_name}_3_comparison.jpg"), comp)

        results.append({
            'filename': img_path.name,
            'original_size': original_size,
            'aligned_size': aligned_size,
            'angle': angle,
            'correction_angle': correction_angle,
            'cos': cos_val,
            'sin': sin_val,
            'rotation_applied': rotation_applied
        })

        success_count += 1

    except Exception as e:
        error_count += 1
        print(f"      Error processing {img_path.name}: {e}")

# ============================================
# SAVE RESULTS
# ============================================

print(f"\n[4/4] Results:")

# Save to CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

print("-" * 80)

for i, r in enumerate(results, 1):
    rotation_mark = "[ROTATED]" if r['rotation_applied'] else "[SKIPPED]"
    print(f"  {i:2d}. {r['filename'][:50]:50s} {rotation_mark}")
    print(f"      {r['original_size']} -> {r['aligned_size']}")
    print(f"      Angle: {r['angle']:6.1f} | Correction: {r['correction_angle']:6.1f}")
    print(f"      Cos: {r['cos']:6.3f} | Sin: {r['sin']:6.3f}")

# Statistics
print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)

print(f"\nTotal images:   {len(image_files)}")
print(f"Success:        {success_count}")
print(f"Errors:         {error_count}")

if success_count > 0:
    angles = [r['angle'] for r in results]
    corrections = [abs(r['correction_angle']) for r in results]
    rotations = [r for r in results if r['rotation_applied']]

    print(f"\nAngle Statistics:")
    print(f"  Min:  {min(angles):.1f}")
    print(f"  Max:  {max(angles):.1f}")
    print(f"  Mean: {np.mean(angles):.1f}")
    print(f"  Std:  {np.std(angles):.1f}")

    print(f"\nCorrection Statistics:")
    print(f"  Min:  {min(corrections):.1f}")
    print(f"  Max:  {max(corrections):.1f}")
    print(f"  Mean: {np.mean(corrections):.1f}")

    print(f"\nRotation Applied:")
    print(f"  Rotated:  {len(rotations)} ({len(rotations)/len(results)*100:.1f}%)")
    print(f"  Skipped:  {len(results) - len(rotations)} ({(len(results) - len(rotations))/len(results)*100:.1f}%)")
else:
    print("\nNo successful predictions to analyze.")

print(f"\n" + "=" * 80)
print("OUTPUT LOCATIONS")
print("=" * 80)
print(f"Aligned images:  {OUTPUT_DIR}")
print(f"Test results:    {TEST_RESULTS_DIR}")
print(f"Metadata CSV:    {OUTPUT_DIR / 'metadata.csv'}")
print("=" * 80)

print(f"\nNext steps:")
print(f"  1. Review aligned images in: {OUTPUT_DIR}")
print(f"  2. Check comparison images in: {TEST_RESULTS_DIR}")
print(f"  3. Use aligned images for M3 ROI detection")
print(f"  4. Process all images: Set NUM_SAMPLES = None in script")
print("=" * 80)

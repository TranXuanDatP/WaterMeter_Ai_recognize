"""
Analyze pipeline failures - find out why 429 images failed
"""
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torchvision.models as models

# M2 Model
class M2_OrientationModel_Fixed(nn.Module):
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

# Paths
DATA_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
LABELS_FILE = Path(r"F:\Workspace\Project\data\images_4digit2.csv")
M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M2_MODEL = r"F:\Workspace\Project\model\m2_angle_model_epoch15_FIXED_COS_SIN.pth"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"

print("="*70)
print("PIPELINE FAILURE ANALYSIS")
print("="*70)

# Load labels
labels_df = pd.read_csv(LABELS_FILE)
all_images = list(DATA_DIR.glob("*.jpg"))

print(f"\n[1] OVERALL STATISTICS")
print(f"-"*70)
print(f"Total images in dataset: {len(all_images)}")
print(f"Total images in labels: {len(labels_df)}")

# Check which images failed
success_df = pd.read_csv(r"F:\Workspace\Project\results\pipeline_fixed_m2\pipeline_results.csv")
success_files = set(success_df['filename'].values)

all_files_with_ext = set([f.name for f in all_images])
failed_files = all_files_with_ext - success_files

print(f"Successful: {len(success_files)}")
print(f"Failed: {len(failed_files)}")
print(f"Success rate: {len(success_files)/len(all_images)*100:.2f}%")

# Test some failed images with M1
print(f"\n[2] TESTING M1 (Watermeter Detection) ON FAILED IMAGES")
print(f"-"*70)

m1_model = YOLO(M1_MODEL)
m1_failures = []
m1_successes = []

# Test first 20 failed images
failed_list = list(failed_files)[:20]

for fname in failed_list:
    img_path = DATA_DIR / fname
    if not img_path.exists():
        m1_failures.append((fname, "File not found"))
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        m1_failures.append((fname, "Cannot read image"))
        continue

    try:
        results = m1_model(img, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            m1_failures.append((fname, "No watermeter detected"))
        else:
            m1_successes.append((fname, len(results[0].boxes)))
    except Exception as e:
        m1_failures.append((fname, f"Error: {str(e)}"))

print(f"M1 Test Results ({len(failed_list)} failed images):")
print(f"  M1 Success (found watermeter): {len(m1_successes)}")
print(f"  M1 Failures: {len(m1_failures)}")

if m1_failures:
    print(f"\nM1 Failure breakdown:")
    for fname, reason in m1_failures[:10]:
        print(f"  {fname[:50]}: {reason}")

# Test some failed images with full pipeline
print(f"\n[3] TESTING FULL PIPELINE ON SOME FAILED IMAGES")
print(f"-"*70)

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cpu')
m2_model = M2_OrientationModel_Fixed(pretrained=False).to(device)
checkpoint = torch.load(M2_MODEL, map_location=device, weights_only=False)
m2_model.load_state_dict(checkpoint['model_state_dict'])
m2_model.eval()

m3_model = YOLO(M3_MODEL)

pipeline_failures = []

for fname in failed_list[:10]:
    img_path = DATA_DIR / fname
    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        pipeline_failures.append((fname, "Cannot read image"))
        continue

    failure_reason = []

    # Test M1
    try:
        m1_results = m1_model(img, verbose=False)
        if len(m1_results) == 0 or len(m1_results[0].boxes) == 0:
            failure_reason.append("M1: No detection")
        else:
            # M1 success, test M2
            boxes = m1_results[0].boxes
            best_idx = boxes.conf.argmax()
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            meter_crop = img[y1:y2, x1:x2]

            # Test M2
            try:
                pil_img = Image.fromarray(cv2.cvtColor(meter_crop, cv2.COLOR_BGR2RGB))
                tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    vec = m2_model(tensor)

                cos_val = vec[0, 0].cpu().item()
                sin_val = vec[0, 1].cpu().item()
                angle_deg = np.degrees(np.arctan2(sin_val, cos_val))

                # Rotate
                height, width = meter_crop.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)

                cos = np.abs(rotation_matrix[0, 0])
                sin = np.abs(rotation_matrix[0, 1])
                new_width = int((height * sin) + (width * cos))
                new_height = int((height * cos) + (width * sin))

                rotation_matrix[0, 2] += (new_width - width) // 2
                rotation_matrix[1, 2] += (new_height - height) // 2

                meter_aligned = cv2.warpAffine(meter_crop, rotation_matrix, (new_width, new_height),
                                               flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=(255, 255, 255))

                # Test M3
                m3_results = m3_model(meter_aligned, verbose=False)
                if len(m3_results) == 0 or len(m3_results[0].boxes) == 0:
                    failure_reason.append("M3: No ROI detection")
                else:
                    failure_reason.append("M3: Success but failed in pipeline?")

            except Exception as e:
                failure_reason.append(f"M2/M3 Error: {str(e)[:50]}")

    except Exception as e:
        failure_reason.append(f"M1 Error: {str(e)[:50]}")

    if failure_reason:
        pipeline_failures.append((fname, failure_reason))

print(f"\nPipeline Test Results:")
for fname, reasons in pipeline_failures:
    print(f"  {fname[:50]}:")
    for r in reasons:
        print(f"    - {r}")

# Check for missing files
print(f"\n[4] CHECKING FILE INTEGRITY")
print(f"-"*70)

missing_in_labels = []
missing_in_dataset = []

for f in all_images:
    if f.name not in labels_df['photo_name'].values:
        missing_in_labels.append(f.name)

for fname in labels_df['photo_name'].values:
    if not (DATA_DIR / fname).exists():
        missing_in_dataset.append(fname)

print(f"Images in dataset but NOT in labels: {len(missing_in_labels)}")
if missing_in_labels:
    print(f"  Examples: {missing_in_labels[:5]}")

print(f"\nImages in labels but NOT in dataset: {len(missing_in_dataset)}")
if missing_in_dataset:
    print(f"  Examples: {missing_in_dataset[:5]}")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print(f"Main failure reasons:")
print(f"  1. M1: No watermeter detection")
print(f"  2. M3: No ROI detection (after successful M2)")
print(f"  3. File corruption or format issues")
print(f"{'='*70}")

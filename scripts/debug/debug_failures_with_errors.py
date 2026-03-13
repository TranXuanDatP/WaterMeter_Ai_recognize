"""
Debug pipeline failures with detailed error tracking
"""
import sys
import traceback
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

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

class M2_SmartRotator_Fixed:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = M2_OrientationModel_Fixed(pretrained=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_angle(self, image: np.ndarray) -> float:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vec = self.model(tensor)

        cos_val = vec[0, 0].cpu().item()
        sin_val = vec[0, 1].cpu().item()
        angle_deg = np.degrees(np.arctan2(sin_val, cos_val))
        return angle_deg

    def smart_rotate(self, image: np.ndarray, angle: float) -> tuple:
        correction_angle = -angle

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)

        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        rotation_matrix[0, 2] += (new_width - width) // 2
        rotation_matrix[1, 2] += (new_height - height) // 2

        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        return rotated, correction_angle

# Paths
DATA_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
LABELS_FILE = Path(r"F:\Workspace\Project\data\images_4digit2.csv")
M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M2_MODEL = r"F:\Workspace\Project\model\m2_angle_model_epoch15_FIXED_COS_SIN.pth"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"

# Load data
labels_df = pd.read_csv(LABELS_FILE)
success_df = pd.read_csv(r"F:\Workspace\Project\results\pipeline_fixed_m2\pipeline_results.csv")
success_files = set(success_df['filename'].values)

all_files_with_ext = set([f.name for f in DATA_DIR.glob("*.jpg")])
failed_files = all_files_with_ext - success_files

print("="*70)
print("DETAILED FAILURE DEBUG WITH ERROR TRACKING")
print("="*70)
print(f"\nTotal failed: {len(failed_files)} images")

# Load models
m1_model = YOLO(M1_MODEL)
m2_rotator = M2_SmartRotator_Fixed(M2_MODEL)
m3_model = YOLO(M3_MODEL)

# Track failures by stage
failures_by_stage = {
    'file_not_found': [],
    'file_read_error': [],
    'm1_no_detection': [],
    'm1_error': [],
    'm2_error': [],
    'm3_no_detection': [],
    'm3_error': [],
    'success': []
}

# Test first 50 failed images
sample_size = min(50, len(failed_files))
failed_list = list(failed_files)[:sample_size]

for fname in failed_list:
    img_path = DATA_DIR / fname

    # Check file exists
    if not img_path.exists():
        failures_by_stage['file_not_found'].append(fname)
        continue

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        failures_by_stage['file_read_error'].append(fname)
        continue

    try:
        # M1
        m1_results = m1_model(img, verbose=False)
        if len(m1_results) == 0 or len(m1_results[0].boxes) == 0:
            failures_by_stage['m1_no_detection'].append(fname)
            continue

        boxes = m1_results[0].boxes
        best_idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

        # Validate bbox
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
            failures_by_stage['m1_error'].append((fname, "Invalid bbox"))
            continue

        meter_crop = img[y1:y2, x1:x2]

        # M2
        try:
            detected_angle = m2_rotator.predict_angle(meter_crop)
            meter_aligned, correction_angle = m2_rotator.smart_rotate(meter_crop, detected_angle)
        except Exception as e:
            failures_by_stage['m2_error'].append((fname, str(e)[:100]))
            continue

        # M3
        try:
            m3_results = m3_model(meter_aligned, verbose=False)
            if len(m3_results) == 0 or len(m3_results[0].boxes) == 0:
                failures_by_stage['m3_no_detection'].append(fname)
                continue

            boxes = m3_results[0].boxes
            best_idx = boxes.conf.argmax()
            cx1, cy1, cx2, cy2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

            # Validate bbox
            if cx2 <= cx1 or cy2 <= cy1 or cx1 < 0 or cy1 < 0:
                failures_by_stage['m3_error'].append((fname, "Invalid bbox"))
                continue

            # If we get here, it should have succeeded
            failures_by_stage['success'].append(fname)

        except Exception as e:
            failures_by_stage['m3_error'].append((fname, str(e)[:100]))

    except Exception as e:
        failures_by_stage['m1_error'].append((fname, str(e)[:100]))

# Print results
print(f"\nBased on {sample_size} sampled failed images:")
print(f"-"*70)

for stage, files in failures_by_stage.items():
    count = len(files)
    pct = count / sample_size * 100
    print(f"{stage}: {count} ({pct:.1f}%)")
    if stage == 'm1_error' and count > 0:
        print(f"  Examples: {files[:3]}")
    if stage == 'm2_error' and count > 0:
        print(f"  Examples: {files[:3]}")
    if stage == 'm3_error' and count > 0:
        print(f"  Examples: {files[:3]}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

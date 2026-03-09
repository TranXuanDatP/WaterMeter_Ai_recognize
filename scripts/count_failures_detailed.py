"""
Count detailed failure statistics
"""
import pandas as pd
from pathlib import Path
import cv2
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
print("DETAILED FAILURE ANALYSIS")
print("="*70)
print(f"\nTotal failed: {len(failed_files)} images")

# Load models
m1_model = YOLO(M1_MODEL)
m3_model = YOLO(M3_MODEL)

device = torch.device('cpu')
m2_model = M2_OrientationModel_Fixed(pretrained=False).to(device)
checkpoint = torch.load(M2_MODEL, map_location=device, weights_only=False)
m2_model.load_state_dict(checkpoint['model_state_dict'])
m2_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Count failures
m1_failures = []
m3_failures = []
other_failures = []

# Sample 100 failed images for analysis
sample_size = min(100, len(failed_files))
failed_list = list(failed_files)[:sample_size]

for fname in failed_list:
    img_path = DATA_DIR / fname
    if not img_path.exists():
        other_failures.append((fname, "File not found"))
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        other_failures.append((fname, "Cannot read"))
        continue

    # Test M1
    try:
        m1_results = m1_model(img, verbose=False)
        if len(m1_results) == 0 or len(m1_results[0].boxes) == 0:
            m1_failures.append(fname)
            continue
    except:
        other_failures.append((fname, "M1 error"))
        continue

    # M1 success, test M2+M3
    try:
        boxes = m1_results[0].boxes
        best_idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        meter_crop = img[y1:y2, x1:x2]

        # M2
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

        # M3
        m3_results = m3_model(meter_aligned, verbose=False)
        if len(m3_results) == 0 or len(m3_results[0].boxes) == 0:
            m3_failures.append(fname)
        else:
            other_failures.append((fname, "Pipeline success but missing?"))

    except Exception as e:
        other_failures.append((fname, f"M2/M3 error: {str(e)[:30]}"))

# Calculate percentages
m1_pct = len(m1_failures) / sample_size * 100
m3_pct = len(m3_failures) / sample_size * 100
other_pct = len(other_failures) / sample_size * 100

print(f"\nBased on {sample_size} sampled failed images:")
print(f"-"*70)
print(f"M1 Failures (No watermeter detected): {len(m1_failures)} ({m1_pct:.1f}%)")
print(f"M3 Failures (No ROI detected): {len(m3_failures)} ({m3_pct:.1f}%)")
print(f"Other Failures: {len(other_failures)} ({other_pct:.1f}%)")

# Extrapolate to full dataset
total_failed = len(failed_files)
est_m1_failures = int(total_failed * m1_pct / 100)
est_m3_failures = int(total_failed * m3_pct / 100)
est_other_failures = total_failed - est_m1_failures - est_m3_failures

print(f"\nEstimated breakdown for all {total_failed} failures:")
print(f"-"*70)
print(f"M1 Failures: ~{est_m1_failures} images")
print(f"M3 Failures: ~{est_m3_failures} images")
print(f"Other Failures: ~{est_other_failures} images")

print(f"\n{'='*70}")
print("ROOT CAUSES")
print(f"{'='*70}")
print(f"\n1. M1 FAILURE ({m1_pct:.1f}% of failures):")
print(f"   - Watermeter not detected by YOLO")
print(f"   - Possible causes:")
print(f"     * Image quality issues (blur, low resolution)")
print(f"     * Watermeter partially out of frame")
print(f"     * Unusual angles or lighting conditions")
print(f"     * Model confidence threshold too high (current: 0.25)")

print(f"\n2. M3 FAILURE ({m3_pct:.1f}% of failures):")
print(f"   - ROI (counter region) not detected after M2 alignment")
print(f"   - Possible causes:")
print(f"     * M2 rotation not perfect (ROI still misaligned)")
print(f"     * Counter region too small or blurred")
print(f"     * Model confidence threshold too high (current: 0.25)")
print(f"     * Some meters have different counter layouts")

print(f"\n3. OTHER FAILURES ({other_pct:.1f}% of failures):")
print(f"   - File corruption, format issues, or processing errors")

print(f"\n{'='*70}")
print("RECOMMENDATIONS")
print(f"{'='*70}")
print(f"1. Lower M1 confidence threshold (0.25 -> 0.15)")
print(f"2. Lower M3 confidence threshold (0.25 -> 0.15)")
print(f"3. Add fallback logic for edge cases")
print(f"4. Collect more training data for difficult cases")
print(f"{'='*70}")

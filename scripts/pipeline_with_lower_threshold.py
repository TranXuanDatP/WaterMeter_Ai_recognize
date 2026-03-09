"""
Full Pipeline with FIXED M2 Model + LOWER THRESHOLDS

Changes:
1. M1 Confidence: 0.25 -> 0.15
2. M3 Confidence: 0.25 -> 0.10
"""
import sys
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import torchvision.models as models

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

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
        print("[M2] Loading FIXED model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = M2_OrientationModel_Fixed(pretrained=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"      Model: {model_path}")
        print(f"      Epoch: {checkpoint.get('epoch')}")
        print(f"      Val Loss: {checkpoint.get('val_loss'):.6f}")
        print(f"      Device: {self.device}")

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


class Config:
    M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
    M2_MODEL = r"F:\Workspace\Project\model\m2_angle_model_epoch15_FIXED_COS_SIN.pth"
    M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"
    M4_MODEL = r"F:\Workspace\Project\model\M4_OCR.pth"

    DATA_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
    LABELS_FILE = Path(r"F:\Workspace\Project\data\images_4digit2.csv")

    OUTPUT_DIR = Path(r"F:\Workspace\Project\results\pipeline_lower_threshold")
    LOG_DIR = OUTPUT_DIR / "logs"

    M1_CROPS_DIR = OUTPUT_DIR / "m1_crops"
    M2_ALIGNED_DIR = OUTPUT_DIR / "m2_aligned"
    M3_ROI_DIR = OUTPUT_DIR / "m3_roi_crops"

    # CHANGED: Lower thresholds!
    M1_CONFIDENCE = 0.15  # Was 0.25
    M3_CONFIDENCE = 0.10  # Was 0.25

    IMG_SIZE = (64, 224)
    CHAR_MAP = "0123456789"


def run_pipeline(config: Config, num_samples: int = None):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    config.M1_CROPS_DIR.mkdir(exist_ok=True)
    config.M2_ALIGNED_DIR.mkdir(exist_ok=True)
    config.M3_ROI_DIR.mkdir(exist_ok=True)

    print("="*70)
    print("FULL PIPELINE WITH FIXED M2 + LOWER THRESHOLDS")
    print("="*70)
    print(f"M1 Confidence: {config.M1_CONFIDENCE} (was 0.25)")
    print(f"M3 Confidence: {config.M3_CONFIDENCE} (was 0.25)")
    print(f"Output: {config.OUTPUT_DIR}")
    print("="*70)
    print()

    print("[1/4] Loading models...")
    m1_model = YOLO(config.M1_MODEL)
    m2_rotator = M2_SmartRotator_Fixed(config.M2_MODEL)
    m3_model = YOLO(config.M3_MODEL)
    print(">> All models loaded\n")

    print("[2/4] Loading data...")
    labels_df = pd.read_csv(config.LABELS_FILE)
    image_files = list(config.DATA_DIR.glob("*.jpg"))
    if num_samples:
        image_files = image_files[:num_samples]
    print(f">> Found {len(image_files)} images\n")

    print("[3/4] Running pipeline...")
    results = []

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            img_name = img_path.name
            label_row = labels_df[labels_df['photo_name'] == img_name]
            if len(label_row) == 0:
                continue

            true_value = label_row.iloc[0]['value']
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # M1
            m1_results = m1_model(img, verbose=False, conf=config.M1_CONFIDENCE)
            if len(m1_results) == 0 or len(m1_results[0].boxes) == 0:
                continue

            boxes = m1_results[0].boxes
            best_idx = boxes.conf.argmax()
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            meter_crop = img[y1:y2, x1:x2]

            # M2
            detected_angle = m2_rotator.predict_angle(meter_crop)
            meter_aligned, correction_angle = m2_rotator.smart_rotate(meter_crop, detected_angle)

            # M3
            m3_results = m3_model(meter_aligned, verbose=False, conf=config.M3_CONFIDENCE)
            if len(m3_results) == 0 or len(m3_results[0].boxes) == 0:
                continue

            boxes = m3_results[0].boxes
            best_idx = boxes.conf.argmax()
            cx1, cy1, cx2, cy2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            roi_crop = meter_aligned[cy1:cy2, cx1:cx2]

            # Save
            base_name = img_path.stem
            cv2.imwrite(str(config.M1_CROPS_DIR / f"{base_name}.jpg"), meter_crop)
            cv2.imwrite(str(config.M2_ALIGNED_DIR / f"{base_name}.jpg"), meter_aligned)
            cv2.imwrite(str(config.M3_ROI_DIR / f"{base_name}.jpg"), roi_crop)

            results.append({
                'filename': img_path.name,
                'true_value': true_value,
                'm1_bbox': [x1, y1, x2, y2],
                'm2_detected_angle': detected_angle,
                'm2_correction_angle': correction_angle,
                'm3_bbox': [cx1, cy1, cx2, cy2]
            })

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    print(f">> Processed {len(results)} images\n")

    print("[4/4] Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.OUTPUT_DIR / "pipeline_results.csv", index=False)

    stats = {
        'total_images': len(image_files),
        'successful': len(results),
        'success_rate': len(results) / len(image_files) if image_files else 0,
        'm1_confidence': config.M1_CONFIDENCE,
        'm3_confidence': config.M3_CONFIDENCE,
        'timestamp': datetime.now().isoformat()
    }

    with open(config.OUTPUT_DIR / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f">> Results saved to {config.OUTPUT_DIR}")

    if results:
        angles = [r['m2_detected_angle'] for r in results]
        print()
        print("="*70)
        print("STATISTICS")
        print("="*70)
        print(f"Total images: {stats['total_images']}")
        print(f"Successful: {stats['successful']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"\nM2 Angles:")
        print(f"  Mean: {np.mean(angles):.2f}°")
        print(f"  Std:  {np.std(angles):.2f}°")
        print(f"  Range: {np.min(angles):.2f}° → {np.max(angles):.2f}°")
        print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline with lower thresholds")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--m1-conf", type=float, default=0.15)
    parser.add_argument("--m3-conf", type=float, default=0.10)

    args = parser.parse_args()

    config = Config()
    config.M1_CONFIDENCE = args.m1_conf
    config.M3_CONFIDENCE = args.m3_conf

    run_pipeline(config, num_samples=args.samples)

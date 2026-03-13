#!/usr/bin/env python3
"""
Test Pipeline on images_4digit_xxxx dataset

M1 -> M2 + Smart Rotate -> M3 -> M3.5 -> M4
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ====================== CONFIGURATION ======================
INPUT_DIR = r"F:\Workspace\Project\data\images_4digit_xxxx"
OUTPUT_DIR = r"F:\Workspace\Project\results\test_images_4digit_xxxx"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")

# Model paths
M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"
M4_MODEL = r"F:\Workspace\Project\model\M4_OCR.pth"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "m1_crops"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "m2_aligned"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "m3_roi"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "m3_5_digits"), exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("TESTING PIPELINE ON images_4digit_xxxx")
print("="*70)
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Device: {DEVICE}")
print(f"Models:")
print(f"  M1: {M1_MODEL}")
print(f"  M2: {M2_MODEL}")
print(f"  M3: {M3_MODEL}")
print(f"  M4: {M4_MODEL}")
print("="*70)


# ====================== M1: WATER METER DETECTION ======================
class M1_WaterMeterDetector:
    """M1: Detect water meter in image using YOLO"""

    def __init__(self, model_path: str, confidence: float = 0.25):
        print("\n[M1] Loading water meter detection model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"      Model loaded: {model_path}")

    def detect(self, image: np.ndarray) -> dict:
        results = self.model(image, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            return {'detected': False, 'bbox': None, 'confidence': 0.0}

        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()

        if confidences[best_idx] < self.confidence:
            return {'detected': False, 'bbox': None, 'confidence': float(confidences[best_idx])}

        box = boxes[best_idx]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        return {
            'detected': True,
            'bbox': (x1, y1, x2, y2),
            'confidence': float(confidences[best_idx])
        }

    def crop(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]


# ====================== M2: ORIENTATION + SMART ROTATE ======================
class M2_OrientationModel(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 512, 1),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.6),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        ca = torch.sigmoid(self.channel_att(x))
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        x = self.gap(x).flatten(1)
        x = self.regressor(x)
        return F.normalize(x, p=2, dim=1)


class M2_SmartRotator:
    def __init__(self, model_path: str, device: torch.device):
        print("\n[M2] Loading orientation + smart rotate model...")
        self.device = device
        self.model = M2_OrientationModel().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"      Model loaded: {model_path}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def predict_angle(self, image: np.ndarray) -> float:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            sin_cos = self.model(tensor)

        sin_val = sin_cos[0, 0].cpu().item()
        cos_val = sin_cos[0, 1].cpu().item()
        angle_rad = np.arctan2(sin_val, cos_val)
        angle_deg = np.degrees(angle_rad)
        angle_deg = (angle_deg + 360) % 360

        return angle_deg

    def smart_rotate(self, image: np.ndarray, angle: float) -> tuple:
        if angle > 180:
            angle -= 360
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        return rotated, angle


# ====================== M3: ROI DETECTION ======================
class M3_ROIDetector:
    def __init__(self, model_path: str, confidence: float = 0.25):
        print("\n[M3] Loading ROI detection model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"      Model loaded: {model_path}")

    def detect(self, image: np.ndarray) -> dict:
        results = self.model(image, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return {'detected': False, 'bbox': None, 'confidence': 0.0}

        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()

        if confidences[best_idx] < self.confidence:
            return {'detected': False, 'bbox': None, 'confidence': float(confidences[best_idx])}

        box = boxes[best_idx]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        return {
            'detected': True,
            'bbox': (x1, y1, x2, y2),
            'confidence': float(confidences[best_idx])
        }

    def crop(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]


# ====================== M3.5: BLACK DIGIT EXTRACTION ======================
class M3_5_DigitExtractor:
    def __init__(self, method: str = "color"):
        print("\n[M3.5] Initializing black digit extractor...")
        self.method = method
        print(f"      Method: {method}")

    def extract(self, roi_image: np.ndarray) -> list:
        if self.method == "color":
            return self._extract_by_color(roi_image)
        else:
            return self._extract_by_projection(roi_image)

    def _extract_by_color(self, image: np.ndarray) -> list:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0

            if 20 < w < 300 and 20 < h < 300 and 0.3 < aspect_ratio < 3.0:
                digit = gray[y:y+h, x:x+w]
                digits.append((digit, (x, y, w, h)))

        digits.sort(key=lambda x: x[1][0])
        return [d[0] for d in digits]


# ====================== M4: CRNN OCR ======================
class M4_CRNNOCR:
    def __init__(self, model_path: str, device: torch.device):
        print("\n[M4] Loading CRNN OCR model...")
        self.device = device
        self.char_map = "0123456789"
        self.label_to_char = {i: c for i, c in enumerate(self.char_map)}
        self.blank_idx = 10
        self._load_model(model_path)
        print(f"      Model loaded: {model_path}")

    def _load_model(self, model_path: str):
        class CRNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                )
                self.rnn = nn.LSTM(1024, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
                self.fc = nn.Linear(512, 11)

            def forward(self, x):
                conv = self.cnn(x)
                b, c, h, w = conv.size()
                conv = conv.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
                out, _ = self.rnn(conv)
                return self.fc(out).permute(1, 0, 2)

        self.model = CRNN().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def recognize(self, image: np.ndarray) -> dict:
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image).convert('L')
        else:
            pil_image = Image.fromarray(image)

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(tensor)

        text, confidence = self._decode(predictions)
        return {'text': text, 'confidence': confidence}

    def _decode(self, predictions: torch.Tensor) -> tuple:
        probs = predictions.softmax(dim=-1)
        pred_indices = predictions.argmax(dim=-1).squeeze().cpu().numpy()

        chars = []
        confidences = []
        prev = None

        for t, idx in enumerate(pred_indices):
            if idx != self.blank_idx and idx != prev:
                chars.append(self.label_to_char[idx])
                confidences.append(probs[t, 0, idx].item())
            prev = idx

        text = ''.join(chars)
        confidence = np.mean(confidences) if confidences else 0.0
        return text, confidence


# ====================== COMPLETE PIPELINE ======================
class CompletePipeline:
    def __init__(self):
        print("\n" + "="*70)
        print("INITIALIZING PIPELINE MODELS")
        print("="*70)

        self.m1 = M1_WaterMeterDetector(M1_MODEL)
        self.m2 = M2_SmartRotator(M2_MODEL, DEVICE)
        self.m3 = M3_ROIDetector(M3_MODEL)
        self.m3_5 = M3_5_DigitExtractor(method="color")
        self.m4 = M4_CRNNOCR(M4_MODEL, DEVICE)

        print("\n" + "="*70)
        print("PIPELINE READY!")
        print("="*70)

    def process_single_image(self, image_path: str, save_intermediates: bool = True) -> dict:
        result = {
            'filename': os.path.basename(image_path),
            'filepath': image_path,
            'success': False,
            'stages': {}
        }

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                result['error'] = 'Could not load image'
                return result

            # ===== M1: Detect Water Meter =====
            m1_result = self.m1.detect(image)
            result['stages']['m1'] = m1_result

            if not m1_result['detected']:
                result['error'] = 'M1: Water meter not detected'
                return result

            m1_crop = self.m1.crop(image, m1_result['bbox'])

            if save_intermediates:
                m1_path = os.path.join(OUTPUT_DIR, "m1_crops", result['filename'])
                cv2.imwrite(m1_path, m1_crop)

            # ===== M2: Orientation + Smart Rotate =====
            angle = self.m2.predict_angle(m1_crop)
            m2_aligned, actual_angle = self.m2.smart_rotate(m1_crop, angle)

            result['stages']['m2'] = {
                'predicted_angle': angle,
                'actual_angle': actual_angle
            }

            if save_intermediates:
                m2_path = os.path.join(OUTPUT_DIR, "m2_aligned", result['filename'])
                cv2.imwrite(m2_path, m2_aligned)

            # ===== M3: Detect ROI =====
            m3_result = self.m3.detect(m2_aligned)
            result['stages']['m3'] = m3_result

            if not m3_result['detected']:
                result['error'] = 'M3: ROI not detected'
                roi_image = m2_aligned
            else:
                roi_image = self.m3.crop(m2_aligned, m3_result['bbox'])

                if save_intermediates:
                    m3_path = os.path.join(OUTPUT_DIR, "m3_roi", result['filename'])
                    cv2.imwrite(m3_path, roi_image)

            # ===== M3.5: Extract Black Digits =====
            digits = self.m3_5.extract(roi_image)
            result['stages']['m3_5'] = {'num_digits': len(digits)}

            if len(digits) == 0:
                result['error'] = 'M3.5: No digits extracted'
                return result

            # ===== M4: OCR =====
            if len(digits) == 4:
                combined = self._combine_digits(digits)
                m4_result = self.m4.recognize(combined)
            else:
                texts = []
                confidences = []
                for digit in digits[:4]:
                    r = self.m4.recognize(digit)
                    texts.append(r['text'])
                    confidences.append(r['confidence'])
                m4_result = {
                    'text': ''.join(texts),
                    'confidence': np.mean(confidences) if confidences else 0.0
                }

            result['stages']['m4'] = m4_result
            result['final_reading'] = m4_result['text']
            result['final_confidence'] = m4_result['confidence']
            result['success'] = True

        except Exception as e:
            result['error'] = f'Exception: {str(e)}'
            import traceback
            result['traceback'] = traceback.format_exc()

        return result

    def _combine_digits(self, digits: list) -> np.ndarray:
        target_height = 64
        resized = []
        for digit in digits:
            if len(digit.shape) == 3:
                digit = cv2.cvtColor(digit, cv2.COLOR_RGB2GRAY)
            h, w = digit.shape
            aspect = w / h
            new_w = int(target_height * aspect)
            resized_digit = cv2.resize(digit, (new_w, target_height))
            resized.append(resized_digit)
        return np.hstack(resized)

    def process_directory(self, input_dir: str, output_csv: str = None):
        image_files = list(Path(input_dir).glob('*.jpg')) + list(Path(input_dir).glob('*.png'))

        print(f"\nFound {len(image_files)} images")
        print("Processing...\n")

        results = []
        success_count = 0
        error_count = 0

        for img_path in tqdm(image_files, desc="Processing"):
            result = self.process_single_image(str(img_path), save_intermediates=True)
            results.append(result)

            if result['success']:
                success_count += 1
                tqdm.write(f"✓ {result['filename']}: {result['final_reading']} ({result['final_confidence']:.3f})")
            else:
                error_count += 1
                tqdm.write(f"✗ {result['filename']}: {result.get('error', 'Unknown error')}")

        # Save results
        if output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")

        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total images: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Success rate: {success_count/len(results)*100:.1f}%")

        # Print some successful readings
        successful_results = [r for r in results if r['success']]
        if successful_results:
            print("\nSample successful readings:")
            for r in successful_results[:10]:
                print(f"  {r['filename']}: {r['final_reading']} (conf: {r['final_confidence']:.3f})")

        print("="*70)

        return results


# ====================== MAIN ======================
if __name__ == "__main__":
    print("\nStarting pipeline test...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    pipeline = CompletePipeline()
    results = pipeline.process_directory(INPUT_DIR, OUTPUT_CSV)

    print("\nTest completed!")

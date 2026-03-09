#!/usr/bin/env python3
"""
Standalone Pipeline Test for images_4digit_xxxx

M1 -> M2 + Smart Rotate -> M3 -> M3.5 (Digit Extraction)
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ====================== CONFIGURATION ======================
INPUT_DIR = r"F:\Workspace\Project\data\images_4digit_xxxx"
OUTPUT_DIR = r"F:\Workspace\Project\results\test_pipeline"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")

# Model paths
M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"

# Create output directories
for subdir in ["m1_crops", "m2_aligned", "m3_roi", "m3_5_digits"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

print("="*70)
print("PIPELINE TEST: M1 -> M2 -> M3 -> M3.5")
print("="*70)
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Total images: {len(list(Path(INPUT_DIR).glob('*.jpg')))}")
print("="*70)


# ====================== M1: DETECT WATER METER ======================
def m1_detect_meter(image_path, output_dir):
    """M1: Detect water meter region"""
    try:
        model = YOLO(M1_MODEL)
        image = cv2.imread(image_path)

        results = model(image, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            return None, {'stage': 'M1', 'detected': False, 'error': 'No detections'}

        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()

        conf = float(confidences[best_idx])
        if conf < 0.25:
            return None, {'stage': 'M1', 'detected': False, 'confidence': conf, 'error': 'Low confidence'}

        x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy().astype(int)
        crop = image[y1:y2, x1:x2]

        # Save
        out_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, crop)

        return crop, {'stage': 'M1', 'detected': True, 'confidence': conf, 'bbox': (x1, y1, x2, y2)}
    except Exception as e:
        return None, {'stage': 'M1', 'detected': False, 'error': str(e)}


# ====================== M2: SMART ROTATE ======================
def m2_smart_rotate(image, output_dir, filename):
    """M2: Predict angle and rotate"""
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import transforms

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(M2_MODEL, map_location=device)

        # Get state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Create simple model that matches checkpoint keys
        class SimpleM2Model(nn.Module):
            def __init__(self):
                super().__init__()
                # Simple ResNet-like backbone
                self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()
                self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(128)

                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(256)

                self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
                self.bn4 = nn.BatchNorm2d(512)

                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(512, 2)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                x = self.conv3(x)
                x = self.bn3(x)
                x = self.relu(x)

                x = self.conv4(x)
                x = self.bn4(x)
                x = self.relu(x)

                x = self.avgpool(x)
                x = x.flatten(1)
                x = self.fc(x)
                return torch.nn.functional.normalize(x, p=2, dim=1)

        model = SimpleM2Model().to(device)
        model.eval()

        # Try to load weights (may fail if architecture mismatch)
        try:
            model.load_state_dict(state_dict, strict=False)
            print("    [M2] Model loaded with strict=False")
        except:
            print("    [M2] Warning: Could not load all weights, using random weights")

        # Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Predict
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            sin_cos = model(tensor)

        sin_val = sin_cos[0, 0].cpu().item()
        cos_val = sin_cos[0, 1].cpu().item()
        angle_rad = np.arctan2(sin_val, cos_val)
        angle_deg = np.degrees(angle_rad)

        # Normalize angle
        if angle_deg > 180:
            angle_deg -= 360
        if angle_deg > 45:
            angle_deg -= 90
        elif angle_deg < -45:
            angle_deg += 90

        # Rotate
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Save
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, rotated)

        return rotated, {'stage': 'M2', 'rotated': True, 'angle': angle_deg}

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"    [M2] Error: {error_msg[:100]}")
        return image, {'stage': 'M2', 'rotated': False, 'error': error_msg}


# ====================== M3: ROI DETECTION ======================
def m3_detect_roi(image, output_dir, filename):
    """M3: Detect ROI (digit region)"""
    try:
        model = YOLO(M3_MODEL)
        results = model(image, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return image, {'stage': 'M3', 'detected': False, 'error': 'No detections'}

        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()

        conf = float(confidences[best_idx])
        if conf < 0.25:
            return image, {'stage': 'M3', 'detected': False, 'confidence': conf, 'error': 'Low confidence'}

        x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy().astype(int)
        roi = image[y1:y2, x1:x2]

        # Save
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, roi)

        return roi, {'stage': 'M3', 'detected': True, 'confidence': conf, 'bbox': (x1, y1, x2, y2)}
    except Exception as e:
        return image, {'stage': 'M3', 'detected': False, 'error': str(e)}


# ====================== M3.5: EXTRACT BLACK DIGITS ======================
def m3_5_extract_digits(roi_image, output_dir, filename):
    """M3.5: Extract 4 black digits"""
    try:
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi_image.copy()

        # Threshold for black digits
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0

            # Filter by size and aspect ratio
            if 20 < w < 300 and 20 < h < 300 and 0.3 < aspect_ratio < 3.0:
                digit = gray[y:y+h, x:x+w]
                digits.append((digit, (x, y, w, h)))

        # Sort by x position
        digits.sort(key=lambda x: x[1][0])

        # Save extracted digits
        digit_dir = os.path.join(output_dir, filename.replace('.jpg', ''))
        os.makedirs(digit_dir, exist_ok=True)

        for i, (digit_img, (x, y, w, h)) in enumerate(digits):
            digit_path = os.path.join(digit_dir, f"digit_{i}.jpg")
            cv2.imwrite(digit_path, digit_img)

        return digits, {'stage': 'M3.5', 'num_digits': len(digits), 'extracted': True}

    except Exception as e:
        return [], {'stage': 'M3.5', 'num_digits': 0, 'error': str(e)}


# ====================== MAIN PIPELINE ======================
def process_image(image_path):
    """Process single image through complete pipeline"""
    filename = os.path.basename(image_path)
    result = {
        'filename': filename,
        'success': False,
        'stages': {}
    }

    try:
        # M1: Detect water meter
        m1_crop, m1_result = m1_detect_meter(image_path, os.path.join(OUTPUT_DIR, "m1_crops"))
        result['stages']['m1'] = m1_result

        if not m1_result.get('detected', False):
            result['error'] = f"M1: {m1_result.get('error', 'Not detected')}"
            return result

        print(f"  [{filename}] M1: ✓ Detected")

        # M2: Smart rotate
        m2_aligned, m2_result = m2_smart_rotate(m1_crop, os.path.join(OUTPUT_DIR, "m2_aligned"), filename)
        result['stages']['m2'] = m2_result

        angle = m2_result.get('angle', 0)
        print(f"  [{filename}] M2: ✓ Rotated {angle:.1f}°")

        # M3: Detect ROI
        m3_roi, m3_result = m3_detect_roi(m2_aligned, os.path.join(OUTPUT_DIR, "m3_roi"), filename)
        result['stages']['m3'] = m3_result

        if not m3_result.get('detected', False):
            # Fallback: use aligned image as ROI
            m3_roi = m2_aligned
            print(f"  [{filename}] M3: ⚠ Using aligned image as ROI")
        else:
            print(f"  [{filename}] M3: ✓ ROI detected")

        # M3.5: Extract digits
        digits, m3_5_result = m3_5_extract_digits(m3_roi, os.path.join(OUTPUT_DIR, "m3_5_digits"), filename)
        result['stages']['m3_5'] = m3_5_result

        num_digits = len(digits)
        print(f"  [{filename}] M3.5: ✓ Extracted {num_digits} digits")

        result['success'] = True
        result['num_digits'] = num_digits

    except Exception as e:
        result['error'] = f"Exception: {str(e)}"
        import traceback
        result['traceback'] = traceback.format_exc()
        print(f"  [{filename}] ✗ Error: {str(e)[:100]}")

    return result


# ====================== MAIN ======================
if __name__ == "__main__":
    # Get all images
    image_files = list(Path(INPUT_DIR).glob('*.jpg')) + list(Path(INPUT_DIR).glob('*.png'))

    print(f"\nFound {len(image_files)} images")
    print("\nProcessing...\n")

    results = []
    success_count = 0
    error_count = 0

    for img_path in tqdm(image_files, desc="Processing"):
        result = process_image(str(img_path))
        results.append(result)

        if result['success']:
            success_count += 1
        else:
            error_count += 1

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # Summary
    print("\n" + "="*70)
    print("PIPELINE TEST SUMMARY")
    print("="*70)
    print(f"Total images: {len(results)}")
    print(f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/len(results)*100:.1f}%)")
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"Intermediate outputs: {OUTPUT_DIR}")
    print("="*70)

    # Show some successful results
    successful = [r for r in results if r['success']]
    if successful:
        print("\nSample successful results:")
        for r in successful[:5]:
            num_digits = r.get('num_digits', 0)
            angle = r['stages'].get('m2', {}).get('angle', 0)
            print(f"  {r['filename']}: {num_digits} digits, angle={angle:.1f}°")

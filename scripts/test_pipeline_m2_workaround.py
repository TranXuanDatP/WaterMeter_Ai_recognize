#!/usr/bin/env python3
"""
Quick Pipeline Test with M2 Workaround

Since M2 model predicts ~280° for upright images, we add a workaround:
- Don't use M2 for now (images from M1 are already aligned)
- Or use a simple angle threshold to skip unnecessary rotations
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Configuration
INPUT_DIR = r"F:\Workspace\Project\data\images_4digit_xxxx"
OUTPUT_DIR = r"F:\Workspace\Project\results\test_pipeline_m2_workaround"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")
MAX_IMAGES = 10

# Model paths
M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"

# Create output directories
for subdir in ["m1_crops", "m2_aligned", "m3_roi", "m3_5_digits"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

print("="*70)
print("PIPELINE TEST - M2 WORKAROUND")
print("="*70)
print("M2 Workaround: Skip ML rotation (images from M1 are already aligned)")
print("="*70)


def m1_detect_meter(image_path, output_dir, filename):
    """M1: Detect water meter region"""
    model = YOLO(M1_MODEL)
    image = cv2.imread(image_path)

    results = model(image, verbose=False)
    if len(results) == 0 or results[0].boxes is None:
        return None, {'detected': False}

    boxes = results[0].boxes
    confidences = boxes.conf.cpu().numpy()
    best_idx = confidences.argmax()

    conf = float(confidences[best_idx])
    if conf < 0.25:
        return None, {'detected': False, 'confidence': conf}

    x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy().astype(int)
    crop = image[y1:y2, x1:x2]

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, crop)

    return crop, {'detected': True, 'confidence': conf, 'bbox': (x1, y1, x2, y2)}


def m2_workaround(image, output_dir, filename):
    """
    M2 Workaround: Skip rotation

    Images from M1 detection are already reasonably aligned.
    M2 model has issues (predicts ~280° for upright images).
    """
    # Just copy the image without rotation
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, image)

    return image, {'rotated': False, 'angle': 0.0, 'method': 'workaround_skip'}


def m3_detect_roi(image, output_dir, filename):
    """M3: Detect ROI (digit region)"""
    model = YOLO(M3_MODEL)
    results = model(image, verbose=False)

    if len(results) == 0 or results[0].boxes is None:
        return image, {'detected': False}

    boxes = results[0].boxes
    confidences = boxes.conf.cpu().numpy()
    best_idx = confidences.argmax()

    conf = float(confidences[best_idx])
    if conf < 0.25:
        return image, {'detected': False, 'confidence': conf}

    x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy().astype(int)
    roi = image[y1:y2, x1:x2]

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, roi)

    return roi, {'detected': True, 'confidence': conf, 'bbox': (x1, y1, x2, y2)}


def m3_5_extract_digits(roi_image, output_dir, filename):
    """M3.5: Extract black digits"""
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi_image.copy()

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

    digit_dir = os.path.join(output_dir, filename.replace('.jpg', ''))
    os.makedirs(digit_dir, exist_ok=True)

    for i, (digit_img, _) in enumerate(digits):
        digit_path = os.path.join(digit_dir, f"digit_{i}.jpg")
        cv2.imwrite(digit_path, digit_img)

    return digits, {'num_digits': len(digits), 'extracted': True}


def process_image(image_path, idx, total):
    """Process single image through pipeline"""
    filename = os.path.basename(image_path)
    result = {
        'filename': filename,
        'success': False,
        'stages': {}
    }

    print(f"\n[{idx+1}/{total}] Processing: {filename}")
    print("-" * 60)

    try:
        # M1: Detect water meter
        print("  [M1] Detecting water meter...")
        m1_crop, m1_result = m1_detect_meter(image_path, os.path.join(OUTPUT_DIR, "m1_crops"), filename)
        result['stages']['m1'] = m1_result

        if not m1_result.get('detected', False):
            result['error'] = f"M1: Not detected (conf={m1_result.get('confidence', 0):.3f})"
            print(f"       X {result['error']}")
            return result

        print(f"       OK Detected (conf={m1_result['confidence']:.3f})")

        # M2: Workaround (skip rotation)
        print("  [M2] Skipping rotation (workaround)...")
        m2_aligned, m2_result = m2_workaround(m1_crop, os.path.join(OUTPUT_DIR, "m2_aligned"), filename)
        result['stages']['m2'] = m2_result
        print(f"       OK Skipped (images already aligned)")

        # M3: Detect ROI
        print("  [M3] Detecting ROI...")
        m3_roi, m3_result = m3_detect_roi(m2_aligned, os.path.join(OUTPUT_DIR, "m3_roi"), filename)
        result['stages']['m3'] = m3_result

        if not m3_result.get('detected', False):
            m3_roi = m2_aligned
            print(f"       ! No ROI detected, using aligned image")
        else:
            print(f"       OK ROI detected (conf={m3_result['confidence']:.3f})")

        # M3.5: Extract digits
        print("  [M3.5] Extracting digits...")
        digits, m3_5_result = m3_5_extract_digits(m3_roi, os.path.join(OUTPUT_DIR, "m3_5_digits"), filename)
        result['stages']['m3_5'] = m3_5_result

        num_digits = len(digits)
        print(f"       OK Extracted {num_digits} digits")

        result['success'] = True
        result['num_digits'] = num_digits

    except Exception as e:
        result['error'] = f"Exception: {str(e)}"
        print(f"       X Error: {str(e)[:100]}")

    return result


if __name__ == "__main__":
    # Get images
    image_files = sorted(list(Path(INPUT_DIR).glob('*.jpg')))[:MAX_IMAGES]

    print(f"\nFound {len(image_files)} images (testing first {MAX_IMAGES})")
    print("\nStarting pipeline...\n")

    results = []
    success_count = 0
    error_count = 0

    for idx, img_path in enumerate(image_files):
        result = process_image(str(img_path), idx, len(image_files))
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
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total images: {len(results)}")
    print(f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/len(results)*100:.1f}%)")
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*70)

    print("\nNOTE: M2 model has issues with angle prediction.")
    print("      This workaround skips M2 rotation.")
    print("      The M2 model needs to be retrained with corrected labels.")
    print("="*70)

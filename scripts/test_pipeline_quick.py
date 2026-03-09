#!/usr/bin/env python3
"""
Quick Pipeline Test on 10 images from images_4digit_xxxx

M1 -> M2 + Smart Rotate -> M3 -> M3.5
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# ====================== CONFIGURATION ======================
INPUT_DIR = r"F:\Workspace\Project\data\images_4digit_xxxx"
OUTPUT_DIR = r"F:\Workspace\Project\results\test_pipeline"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")
MAX_IMAGES = 10  # Test only first 10 images

# Model paths
M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"

# Create output directories
for subdir in ["m1_crops", "m2_aligned", "m3_roi", "m3_5_digits"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

print("="*70)
print("QUICK PIPELINE TEST (10 images)")
print("="*70)
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("="*70)


# ====================== M1: DETECT WATER METER ======================
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

    # Save
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, crop)

    return crop, {'detected': True, 'confidence': conf, 'bbox': (x1, y1, x2, y2)}


# ====================== M2: SMART ROTATE (NO ML - USE OPENCV) ======================
def m2_smart_rotate_simple(image, output_dir, filename):
    """M2: Smart rotate using image processing (no ML model)"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect text lines using morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours or len(contours) < 3:
            # No rotation applied
            return image, {'rotated': False, 'angle': 0.0, 'method': 'none'}

        # Use PCA to find dominant orientation
        all_points = np.vstack([cnt for cnt in contours[:20]])  # Use top 20 contours
        mean, eigenvectors = cv2.PCACompute(all_points.astype(np.float32), mean=None)

        # Calculate angle from principal component
        angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))

        # Adjust to [-45, 45] range
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        # Save
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, rotated)

        return rotated, {'rotated': True, 'angle': angle, 'method': 'opencv_pca'}

    except Exception as e:
        return image, {'rotated': False, 'angle': 0.0, 'error': str(e)}


# ====================== M3: ROI DETECTION ======================
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

    # Save
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, roi)

    return roi, {'detected': True, 'confidence': conf, 'bbox': (x1, y1, x2, y2)}


# ====================== M3.5: EXTRACT BLACK DIGITS ======================
def m3_5_extract_digits(roi_image, output_dir, filename):
    """M3.5: Extract 4 black digits"""
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

    for i, (digit_img, _) in enumerate(digits):
        digit_path = os.path.join(digit_dir, f"digit_{i}.jpg")
        cv2.imwrite(digit_path, digit_img)

    return digits, {'num_digits': len(digits), 'extracted': True}


# ====================== MAIN PIPELINE ======================
def process_image(image_path, idx, total):
    """Process single image through complete pipeline"""
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
            print(f"       ✗ {result['error']}")
            return result

        print(f"       ✓ Detected (conf={m1_result['confidence']:.3f})")

        # M2: Smart rotate (using OpenCV PCA)
        print("  [M2] Smart rotating...")
        m2_aligned, m2_result = m2_smart_rotate_simple(m1_crop, os.path.join(OUTPUT_DIR, "m2_aligned"), filename)
        result['stages']['m2'] = m2_result

        angle = m2_result.get('angle', 0)
        rotated = m2_result.get('rotated', False)
        print(f"       ✓ Rotated {angle:.1f}° ({m2_result.get('method', 'unknown')})")

        # M3: Detect ROI
        print("  [M3] Detecting ROI...")
        m3_roi, m3_result = m3_detect_roi(m2_aligned, os.path.join(OUTPUT_DIR, "m3_roi"), filename)
        result['stages']['m3'] = m3_result

        if not m3_result.get('detected', False):
            # Fallback: use aligned image as ROI
            m3_roi = m2_aligned
            print(f"       ⚠ No ROI detected, using aligned image")
        else:
            print(f"       ✓ ROI detected (conf={m3_result['confidence']:.3f})")

        # M3.5: Extract digits
        print("  [M3.5] Extracting digits...")
        digits, m3_5_result = m3_5_extract_digits(m3_roi, os.path.join(OUTPUT_DIR, "m3_5_digits"), filename)
        result['stages']['m3_5'] = m3_5_result

        num_digits = len(digits)
        print(f"       ✓ Extracted {num_digits} digits")

        result['success'] = True
        result['num_digits'] = num_digits

    except Exception as e:
        result['error'] = f"Exception: {str(e)}"
        print(f"       ✗ Error: {str(e)[:100]}")

    return result


# ====================== MAIN ======================
if __name__ == "__main__":
    # Get images
    image_files = list(Path(INPUT_DIR).glob('*.jpg'))
    image_files = sorted(image_files)[:MAX_IMAGES]  # Limit to 10 images

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

    # Show detailed results
    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        digits = r.get('num_digits', 0)
        angle = r['stages'].get('m2', {}).get('angle', 0)
        error = r.get('error', '')
        print(f"  {status} {r['filename']}: {digits} digits, angle={angle:.1f}° {error}")

    print("\n✓ Test completed! Check outputs at: " + OUTPUT_DIR)

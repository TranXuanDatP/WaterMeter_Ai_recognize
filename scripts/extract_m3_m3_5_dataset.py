#!/usr/bin/env python3
"""
Extract M3 ROI and M3.5 Black Digits from images_4digit_xxxx

Creates datasets in the same format as:
- m3_roi_crops_all
- m4_ocr_dataset_black_digits

Output:
- M3: ROI crops saved to output_m3_roi/
- M3.5: Two formats:
  - Individual digits: output_m3_5_digits/[parent_folder]/digit_0.jpg, digit_1.jpg, ...
  - Word-level: output_m3_5_word/images/ + labels.csv (for CRNN training)
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ====================== CONFIGURATION ======================
INPUT_DIR = r"F:\Workspace\Project\data\images_4digit_xxxx"
OUTPUT_M3 = r"F:\Workspace\Project\data\m3_roi_crops_new"
OUTPUT_M3_5_DIGITS = r"F:\Workspace\Project\data\m3_5_black_digits_individual"
OUTPUT_M3_5_WORD = r"F:\Workspace\Project\data\m3_5_word_dataset"

M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"

# M3.5 Digit extraction parameters
DIGIT_MIN_WIDTH = 20
DIGIT_MAX_WIDTH = 300
DIGIT_MIN_HEIGHT = 20
DIGIT_MAX_HEIGHT = 300
DIGIT_MIN_ASPECT = 0.3
DIGIT_MAX_ASPECT = 3.0
THRESHOLD_VALUE = 120

# Create output directories
os.makedirs(OUTPUT_M3, exist_ok=True)
os.makedirs(OUTPUT_M3_5_DIGITS, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_M3_5_WORD, "images"), exist_ok=True)

print("="*70)
print("M3 ROI + M3.5 BLACK DIGIT EXTRACTION")
print("="*70)
print(f"Input: {INPUT_DIR}")
print(f"M3 Output: {OUTPUT_M3}")
print(f"M3.5 Digits Output: {OUTPUT_M3_5_DIGITS}")
print(f"M3.5 Word Output: {OUTPUT_M3_5_WORD}")
print("="*70)


# ====================== M1: DETECT WATER METER ======================
def m1_detect_meter(model, image_path):
    """M1: Detect water meter region"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None

    results = model(image, verbose=False)
    if len(results) == 0 or results[0].boxes is None:
        return None, None

    boxes = results[0].boxes
    confidences = boxes.conf.cpu().numpy()
    best_idx = confidences.argmax()

    conf = float(confidences[best_idx])
    if conf < 0.25:
        return None, None

    x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy().astype(int)
    crop = image[y1:y2, x1:x2]

    return crop, {'confidence': conf, 'bbox': (x1, y1, x2, y2)}


# ====================== M3: ROI DETECTION ======================
def m3_detect_roi(model, image):
    """M3: Detect ROI (digit region)"""
    results = model(image, verbose=False)

    if len(results) == 0 or results[0].boxes is None:
        return None, None

    boxes = results[0].boxes
    confidences = boxes.conf.cpu().numpy()
    best_idx = confidences.argmax()

    conf = float(confidences[best_idx])
    if conf < 0.25:
        return None, None

    x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy().astype(int)
    roi = image[y1:y2, x1:x2]

    return roi, {'confidence': conf, 'bbox': (x1, y1, x2, y2)}


# ====================== M3.5: EXTRACT BLACK DIGITS ======================
def m3_5_extract_digits(roi_image, min_width=20, max_width=300,
                        min_height=20, max_height=300,
                        min_aspect=0.3, max_aspect=3.0,
                        threshold=120):
    """
    M3.5: Extract individual black digits from ROI

    Returns: list of (digit_image, bbox) tuples sorted by x position
    """
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi_image.copy()

    # Threshold for black digits on white/light background
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip if too small
        if w < min_width or h < min_height:
            continue

        # Skip if too large
        if w > max_width or h > max_height:
            continue

        # Check aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue

        # Extract digit
        digit = gray[y:y+h, x:x+w]

        # Add padding
        padding = 5
        digit_padded = cv2.copyMakeBorder(digit, padding, padding, padding, padding,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))

        digits.append((digit_padded, (x, y, w, h)))

    # Sort by x position (left to right)
    digits.sort(key=lambda x: x[1][0])

    return digits


# ====================== MAIN EXTRACTION ======================
def extract_datasets():
    """Extract M3 and M3.5 datasets"""

    # Load models
    print("\nLoading models...")
    m1_model = YOLO(M1_MODEL)
    m3_model = YOLO(M3_MODEL)
    print("Models loaded")

    # Get all images
    image_files = sorted(list(Path(INPUT_DIR).glob('*.jpg')))
    print(f"\nFound {len(image_files)} images")

    # Results tracking
    m3_results = []
    m3_5_word_results = []

    # Statistics
    stats = {
        'total': len(image_files),
        'm1_success': 0,
        'm3_success': 0,
        'm3_5_success': 0,
        'total_digits': 0
    }

    print("\nExtracting...")
    print("="*70)

    for img_path in tqdm(image_files, desc="Processing"):
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]

        try:
            # M1: Detect water meter
            m1_crop, m1_info = m1_detect_meter(m1_model, img_path)
            if m1_crop is None:
                continue

            stats['m1_success'] += 1

            # M3: Detect ROI
            m3_roi, m3_info = m3_detect_roi(m3_model, m1_crop)

            if m3_roi is None:
                # No ROI detected, use M1 crop
                m3_roi = m1_crop
                m3_conf = 0.0
            else:
                m3_conf = m3_info['confidence']
                stats['m3_success'] += 1

            # Save M3 ROI (format: crop_[filename].jpg)
            m3_filename = f"crop_{filename}"
            m3_path = os.path.join(OUTPUT_M3, m3_filename)
            cv2.imwrite(m3_path, m3_roi)

            m3_results.append({
                'filename': m3_filename,
                'source_image': filename,
                'm3_confidence': m3_conf
            })

            # M3.5: Extract digits
            digits = m3_5_extract_digits(
                m3_roi,
                min_width=DIGIT_MIN_WIDTH,
                max_width=DIGIT_MAX_WIDTH,
                min_height=DIGIT_MIN_HEIGHT,
                max_height=DIGIT_MAX_HEIGHT,
                min_aspect=DIGIT_MIN_ASPECT,
                max_aspect=DIGIT_MAX_ASPECT,
                threshold=THRESHOLD_VALUE
            )

            if len(digits) > 0:
                stats['m3_5_success'] += 1
                stats['total_digits'] += len(digits)

                # Option A: Save individual digits (for character-level training)
                digit_folder = os.path.join(OUTPUT_M3_5_DIGITS, name_without_ext)
                os.makedirs(digit_folder, exist_ok=True)

                for idx, (digit_img, bbox) in enumerate(digits):
                    digit_filename = f"digit_{idx}.jpg"
                    digit_path = os.path.join(digit_folder, digit_filename)
                    cv2.imwrite(digit_path, digit_img)

                # Option B: Save word-level with label (for CRNN training)
                # Create a combined image with all digits horizontally
                if len(digits) >= 1:
                    # Calculate total width
                    total_width = sum(digit.shape[1] for digit, _ in digits)
                    max_height = max(digit.shape[0] for digit, _ in digits)

                    # Create combined image
                    combined = np.ones((max_height, total_width), dtype=np.uint8) * 255

                    # Paste digits
                    x_offset = 0
                    for digit_img, _ in digits:
                        h, w = digit_img.shape
                        # Center vertically
                        y_offset = (max_height - h) // 2
                        combined[y_offset:y_offset+h, x_offset:x_offset+w] = digit_img
                        x_offset += w

                    # Save combined image
                    word_filename = f"crop_{filename}"
                    word_path = os.path.join(OUTPUT_M3_5_WORD, "images", word_filename)
                    cv2.imwrite(word_path, combined)

                    # Create label (unknown text, will need manual labeling)
                    # For now, use placeholder with digit count
                    label = "?" * len(digits)
                    m3_5_word_results.append({
                        'filename': word_filename,
                        'text': label,  # Placeholder - needs manual labeling
                        'num_digits': len(digits)
                    })

        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue

    # Save M3 metadata
    m3_df = pd.DataFrame(m3_results)
    m3_csv_path = os.path.join(OUTPUT_M3, "metadata.csv")
    m3_df.to_csv(m3_csv_path, index=False, encoding='utf-8-sig')
    print(f"\nM3 metadata saved: {m3_csv_path}")

    # Save M3.5 word-level labels (for CRNN)
    m3_5_word_df = pd.DataFrame(m3_5_word_results)
    m3_5_word_csv_path = os.path.join(OUTPUT_M3_5_WORD, "labels.csv")
    m3_5_word_df.to_csv(m3_5_word_csv_path, index=False, encoding='utf-8-sig')
    print(f"M3.5 word labels saved: {m3_5_word_csv_path}")
    print("  Note: Labels are placeholders ('?' * num_digits) and need manual labeling")

    # Print statistics
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Total images processed: {stats['total']}")
    print(f"M1 detection success: {stats['m1_success']} ({stats['m1_success']/stats['total']*100:.1f}%)")
    print(f"M3 ROI detection success: {stats['m3_success']} ({stats['m3_success']/stats['total']*100:.1f}%)")
    print(f"M3.5 digit extraction success: {stats['m3_5_success']} ({stats['m3_5_success']/stats['total']*100:.1f}%)")
    print(f"Total digits extracted: {stats['total_digits']}")
    print(f"Average digits per image: {stats['total_digits']/max(stats['m3_5_success'], 1):.1f}")
    print("="*70)

    print(f"\nOutputs created:")
    print(f"  1. M3 ROI crops: {OUTPUT_M3}/")
    print(f"     - Format: crop_[filename].jpg")
    print(f"     - Metadata: metadata.csv")
    print(f"  2. M3.5 Individual digits: {OUTPUT_M3_5_DIGITS}/")
    print(f"     - Format: [filename]/digit_0.jpg, digit_1.jpg, ...")
    print(f"  3. M3.5 Word-level dataset: {OUTPUT_M3_5_WORD}/")
    print(f"     - Images: images/crop_[filename].jpg")
    print(f"     - Labels: labels.csv (placeholders, needs manual labeling)")
    print("="*70)

    return stats


# ====================== MAIN ======================
if __name__ == "__main__":
    stats = extract_datasets()

    print("\nNext steps:")
    print("  1. Review M3 ROI crops in:", OUTPUT_M3)
    print("  2. Review individual digits in:", OUTPUT_M3_5_DIGITS)
    print("  3. Label M3.5 word-level dataset:", OUTPUT_M3_5_WORD)
    print("     - Edit labels.csv to replace '?' with actual digit readings")
    print("  4. Use labeled dataset for M4 CRNN training")

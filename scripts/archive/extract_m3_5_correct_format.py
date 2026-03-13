#!/usr/bin/env python3
"""
Extract M3 ROI and M3.5 BLACK Digits (CORRECT FORMAT)

M3.5 sẽ:
1. Phân biệt màu đen và đỏ
2. Chỉ extract digits màu đen
3. Resize về cùng height (64px)
4. NÓI NGANG bằng np.hstack() - KHÔNG có padding trắng!

Format giống m4_ocr_dataset_black_digits
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
OUTPUT_M3 = r"F:\Workspace\Project\data\m3_roi_crops_correct"
OUTPUT_M3_5 = r"F:\Workspace\Project\data\m3_5_black_correct_format"

M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"

# Target height for combining digits (same as m4 dataset)
TARGET_HEIGHT = 64

# M3.5 Black digit extraction parameters
DIGIT_MIN_WIDTH = 20
DIGIT_MAX_WIDTH = 300
DIGIT_MIN_HEIGHT = 20
DIGIT_MAX_HEIGHT = 300
DIGIT_MIN_ASPECT = 0.3
DIGIT_MAX_ASPECT = 3.0
BLACK_THRESHOLD = 120

# HSV color ranges
BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 80, 120])
RED_LOWER1 = np.array([0, 100, 100])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 100, 100])
RED_UPPER2 = np.array([180, 255, 255])

# Create output directories
os.makedirs(OUTPUT_M3, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_M3_5, "images"), exist_ok=True)

print("="*70)
print("M3 + M3.5 BLACK DIGITS EXTRACTION (CORRECT FORMAT)")
print("="*70)
print(f"Input: {INPUT_DIR}")
print(f"M3 Output: {OUTPUT_M3}")
print(f"M3.5 Output: {OUTPUT_M3_5}")
print("="*70)
print("Format:")
print("  - Resize digits to height=64")
print("  - Combine horizontally with np.hstack() (NO padding)")
print("  - Same format as m4_ocr_dataset_black_digits")
print("="*70)


# ====================== M1 & M3 MODELS ======================
def m1_detect_meter(model, image_path):
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

    return crop, {'confidence': conf}


def m3_detect_roi(model, image):
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

    return roi, {'confidence': conf}


# ====================== M3.5: BLACK DIGIT EXTRACTION ======================
def is_red_region(hsv_image, x, y, w, h, min_red_ratio=0.3):
    """Check if region is predominantly red"""
    region = hsv_image[y:y+h, x:x+w]

    red_mask1 = cv2.inRange(region, RED_LOWER1, RED_UPPER1)
    red_mask2 = cv2.inRange(region, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = w * h

    return red_pixels / total_pixels > min_red_ratio


def m3_5_extract_black_digits(roi_image):
    """
    Extract ONLY black digits, exclude red digits
    Return: list of digit images
    """
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    else:
        gray = roi_image.copy()
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    # Threshold for black digits
    _, thresh = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_digits = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Size filters
        if w < DIGIT_MIN_WIDTH or h < DIGIT_MIN_HEIGHT:
            continue
        if w > DIGIT_MAX_WIDTH or h > DIGIT_MAX_HEIGHT:
            continue

        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < DIGIT_MIN_ASPECT or aspect_ratio > DIGIT_MAX_ASPECT:
            continue

        # Check if red (exclude)
        if is_red_region(hsv, x, y, w, h):
            continue

        # Extract black digit
        digit = gray[y:y+h, x:x+w]
        black_digits.append((digit, x))

    # Sort by x position
    black_digits.sort(key=lambda x: x[1])

    return [d[0] for d in black_digits]


def combine_digits_hstack(digits, target_height=64):
    """
    Combine digit images using np.hstack() - CORRECT FORMAT

    This is the same method used in m4_ocr_dataset_black_digits
    """
    if len(digits) == 0:
        return None

    # Resize all digits to target height
    resized = []
    for digit in digits:
        h, w = digit.shape
        aspect = w / h
        new_w = int(target_height * aspect)
        resized_digit = cv2.resize(digit, (new_w, target_height))
        resized.append(resized_digit)

    # Concatenate horizontally (NO PADDING)
    combined = np.hstack(resized)

    return combined


# ====================== MAIN EXTRACTION ======================
def extract_datasets():
    """Extract M3 and M3.5 datasets with correct format"""

    # Load models
    print("\nLoading models...")
    m1_model = YOLO(M1_MODEL)
    m3_model = YOLO(M3_MODEL)
    print("Models loaded")

    # Get all images
    image_files = sorted(list(Path(INPUT_DIR).glob('*.jpg')))
    print(f"\nFound {len(image_files)} images")

    # Results tracking
    m3_5_results = []
    stats = {
        'total_processed': 0,
        'black_digits_extracted': 0,
        'images_with_4_digits': 0,
        'images_with_other_count': 0
    }

    print("\nExtracting with CORRECT format (hstack, no padding)...")
    print("="*70)

    for img_path in tqdm(image_files, desc="Processing"):
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]

        try:
            # M1: Detect water meter
            m1_crop, _ = m1_detect_meter(m1_model, img_path)
            if m1_crop is None:
                continue

            # M3: Detect ROI
            m3_roi, _ = m3_detect_roi(m3_model, m1_crop)
            if m3_roi is None:
                continue

            # Save M3 ROI
            m3_filename = f"crop_{filename}"
            m3_path = os.path.join(OUTPUT_M3, m3_filename)
            cv2.imwrite(m3_path, m3_roi)

            # M3.5: Extract black digits
            black_digits = m3_5_extract_black_digits(m3_roi)

            stats['total_processed'] += 1

            if len(black_digits) > 0:
                stats['black_digits_extracted'] += len(black_digits)

                # Combine digits using CORRECT format (hstack)
                combined = combine_digits_hstack(black_digits, target_height=TARGET_HEIGHT)

                if combined is not None:
                    # Save combined image
                    word_filename = f"crop_{filename}"
                    word_path = os.path.join(OUTPUT_M3_5, "images", word_filename)
                    cv2.imwrite(word_path, combined)

                    # Create label
                    label = "?" * len(black_digits)
                    m3_5_results.append({
                        'filename': word_filename,
                        'text': label
                    })

                    # Track statistics
                    if len(black_digits) == 4:
                        stats['images_with_4_digits'] += 1
                    else:
                        stats['images_with_other_count'] += 1

        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue

    # Save labels
    if len(m3_5_results) > 0:
        m3_5_df = pd.DataFrame(m3_5_results)
        labels_path = os.path.join(OUTPUT_M3_5, "labels.csv")
        m3_5_df.to_csv(labels_path, index=False, encoding='utf-8-sig')
        print(f"\nLabels saved: {labels_path}")
    else:
        print("\nWARNING: No images with digits found!")

    # Print statistics
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY (CORRECT FORMAT)")
    print("="*70)
    print(f"Total images processed: {stats['total_processed']}")
    print(f"Total black digits extracted: {stats['black_digits_extracted']}")
    print(f"\nDigit distribution:")
    print(f"  Images with 4 black digits: {stats['images_with_4_digits']}")
    print(f"  Images with other count: {stats['images_with_other_count']}")
    if stats['total_processed'] > 0:
        print(f"  Average black digits per image: {stats['black_digits_extracted']/stats['total_processed']:.1f}")
    print("="*70)

    print(f"\nOutputs created:")
    print(f"  1. M3 ROI crops: {OUTPUT_M3}/")
    print(f"  2. M3.5 Black digits (CORRECT FORMAT): {OUTPUT_M3_5}/")
    print(f"     - Images: images/crop_[filename].jpg")
    print(f"     - Combined using np.hstack() (NO padding)")
    print(f"     - Height normalized to {TARGET_HEIGHT}px")
    print(f"     - Labels: labels.csv")
    print("="*70)

    return stats


if __name__ == "__main__":
    stats = extract_datasets()

    print("\n✓ Extraction complete with CORRECT format!")
    print("  Format matches m4_ocr_dataset_black_digits")
    print("  - Digits resized to height=64")
    print("  - Combined with np.hstack() (no padding)")
    print("  - Ready for M4 CRNN training after manual labeling")

#!/usr/bin/env python3
"""
Extract M3 ROI and M3.5 BLACK Digits Only (exclude red digits)

M3.5 sẽ:
1. Phân biệt màu đen và đỏ bằng HSV color space
2. Chỉ extract 4 digits màu đen
3. Loại bỏ digits màu đỏ
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
OUTPUT_M3_5_DIGITS = r"F:\Workspace\Project\data\m3_5_black_digits_only"
OUTPUT_M3_5_WORD = r"F:\Workspace\Project\data\m3_5_black_word_dataset"

M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"

# M3.5 Black digit extraction parameters
DIGIT_MIN_WIDTH = 20
DIGIT_MAX_WIDTH = 300
DIGIT_MIN_HEIGHT = 20
DIGIT_MAX_HEIGHT = 300
DIGIT_MIN_ASPECT = 0.3
DIGIT_MAX_ASPECT = 3.0
BLACK_THRESHOLD = 120  # For grayscale thresholding

# HSV color ranges for black and red digits
# Black: very low saturation and value
BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 80, 120])

# Red: two ranges in HSV (red wraps around)
RED_LOWER1 = np.array([0, 100, 100])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 100, 100])
RED_UPPER2 = np.array([180, 255, 255])

# Create output directories
os.makedirs(OUTPUT_M3, exist_ok=True)
os.makedirs(OUTPUT_M3_5_DIGITS, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_M3_5_WORD, "images"), exist_ok=True)

print("="*70)
print("M3 ROI + M3.5 BLACK DIGITS ONLY EXTRACTION")
print("="*70)
print(f"Input: {INPUT_DIR}")
print(f"M3 Output: {OUTPUT_M3}")
print(f"M3.5 Black Digits Output: {OUTPUT_M3_5_DIGITS}")
print(f"M3.5 Black Word Output: {OUTPUT_M3_5_WORD}")
print("="*70)
print("M3.5: Chỉ extract 4 digits màu đen, loại bỏ digits màu đỏ")
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


# ====================== M3.5: EXTRACT BLACK DIGITS ONLY ======================
def is_red_region(hsv_image, x, y, w, h, min_red_ratio=0.3):
    """
    Check if a region is predominantly red

    Args:
        hsv_image: Image in HSV color space
        x, y, w, h: Bounding box coordinates
        min_red_ratio: Minimum ratio of red pixels to consider as red digit

    Returns:
        True if region is predominantly red
    """
    # Extract region
    region = hsv_image[y:y+h, x:x+w]

    # Create masks for red
    red_mask1 = cv2.inRange(region, RED_LOWER1, RED_UPPER1)
    red_mask2 = cv2.inRange(region, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Count red pixels
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = w * h

    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0

    return red_ratio > min_red_ratio


def m3_5_extract_black_digits_only(roi_image,
                                   min_width=20, max_width=300,
                                   min_height=20, max_height=300,
                                   min_aspect=0.3, max_aspect=3.0,
                                   threshold=120,
                                   target_digits=4):
    """
    M3.5: Extract ONLY black digits, exclude red digits

    Returns: list of (digit_image, bbox) tuples sorted by x position
    """
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    else:
        gray = roi_image.copy()
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    # Threshold for black digits on white/light background
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_digits = []
    red_digits = []

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

        # Check if region is red or black
        if is_red_region(hsv, x, y, w, h):
            # It's a red digit - skip it
            digit = gray[y:y+h, x:x+w]
            red_digits.append((digit, (x, y, w, h)))
        else:
            # It's a black digit - keep it
            digit = gray[y:y+h, x:x+w]

            # Add padding
            padding = 5
            digit_padded = cv2.copyMakeBorder(digit, padding, padding, padding, padding,
                                           cv2.BORDER_CONSTANT, value=(255, 255, 255))

            black_digits.append((digit_padded, (x, y, w, h)))

    # Sort by x position (left to right)
    black_digits.sort(key=lambda x: x[1][0])

    # Try to get exactly target_digits (4) black digits
    # If we have more, take the 4 most central/largest
    # If we have fewer, return what we have

    if len(black_digits) > target_digits:
        # Prioritize larger digits that are well-positioned
        black_digits.sort(key=lambda x: x[1][2] * x[1][3], reverse=True)  # Sort by area
        black_digits = black_digits[:target_digits]
        black_digits.sort(key=lambda x: x[1][0])  # Re-sort by x position

    return black_digits, len(red_digits)


# ====================== MAIN EXTRACTION ======================
def extract_datasets():
    """Extract M3 and M3.5 datasets (black digits only)"""

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
    m3_5_stats = {
        'total_processed': 0,
        'black_digits_extracted': 0,
        'red_digits_found': 0,
        'images_with_4_digits': 0,
        'images_with_fewer_digits': 0,
        'images_with_more_digits': 0
    }

    print("\nExtracting BLACK digits only (excluding red digits)...")
    print("="*70)

    for img_path in tqdm(image_files, desc="Processing"):
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]

        try:
            # M1: Detect water meter
            m1_crop, m1_info = m1_detect_meter(m1_model, img_path)
            if m1_crop is None:
                continue

            # M3: Detect ROI
            m3_roi, m3_info = m3_detect_roi(m3_model, m1_crop)

            if m3_roi is None:
                # No ROI detected, use M1 crop
                m3_roi = m1_crop
                m3_conf = 0.0
            else:
                m3_conf = m3_info['confidence']

            # Save M3 ROI
            m3_filename = f"crop_{filename}"
            m3_path = os.path.join(OUTPUT_M3, m3_filename)
            cv2.imwrite(m3_path, m3_roi)

            m3_results.append({
                'filename': m3_filename,
                'source_image': filename,
                'm3_confidence': m3_conf
            })

            # M3.5: Extract BLACK digits only
            black_digits, num_red = m3_5_extract_black_digits_only(
                m3_roi,
                min_width=DIGIT_MIN_WIDTH,
                max_width=DIGIT_MAX_WIDTH,
                min_height=DIGIT_MIN_HEIGHT,
                max_height=DIGIT_MAX_HEIGHT,
                min_aspect=DIGIT_MIN_ASPECT,
                max_aspect=DIGIT_MAX_ASPECT,
                threshold=BLACK_THRESHOLD,
                target_digits=4
            )

            m3_5_stats['total_processed'] += 1
            m3_5_stats['red_digits_found'] += num_red

            if len(black_digits) > 0:
                m3_5_stats['black_digits_extracted'] += len(black_digits)

                # Save individual black digits
                digit_folder = os.path.join(OUTPUT_M3_5_DIGITS, name_without_ext)
                os.makedirs(digit_folder, exist_ok=True)

                for idx, (digit_img, bbox) in enumerate(black_digits):
                    digit_filename = f"digit_{idx}.jpg"
                    digit_path = os.path.join(digit_folder, digit_filename)
                    cv2.imwrite(digit_path, digit_img)

                # Create word-level image with black digits only
                total_width = sum(digit.shape[1] for digit, _ in black_digits)
                max_height = max(digit.shape[0] for digit, _ in black_digits)

                combined = np.ones((max_height, total_width), dtype=np.uint8) * 255

                x_offset = 0
                for digit_img, _ in black_digits:
                    h, w = digit_img.shape
                    y_offset = (max_height - h) // 2
                    combined[y_offset:y_offset+h, x_offset:x_offset+w] = digit_img
                    x_offset += w

                word_filename = f"crop_{filename}"
                word_path = os.path.join(OUTPUT_M3_5_WORD, "images", word_filename)
                cv2.imwrite(word_path, combined)

                # Create label
                label = "?" * len(black_digits)
                m3_5_word_results.append({
                    'filename': word_filename,
                    'text': label,
                    'num_black_digits': len(black_digits),
                    'num_red_digits': num_red
                })

                # Track statistics
                if len(black_digits) == 4:
                    m3_5_stats['images_with_4_digits'] += 1
                elif len(black_digits) < 4:
                    m3_5_stats['images_with_fewer_digits'] += 1
                else:
                    m3_5_stats['images_with_more_digits'] += 1

        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue

    # Save results
    m3_df = pd.DataFrame(m3_results)
    m3_csv_path = os.path.join(OUTPUT_M3, "metadata.csv")
    m3_df.to_csv(m3_csv_path, index=False, encoding='utf-8-sig')

    m3_5_word_df = pd.DataFrame(m3_5_word_results)
    m3_5_word_csv_path = os.path.join(OUTPUT_M3_5_WORD, "labels.csv")
    m3_5_word_df.to_csv(m3_5_word_csv_path, index=False, encoding='utf-8-sig')

    # Print statistics
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY (BLACK DIGITS ONLY)")
    print("="*70)
    print(f"Total images processed: {m3_5_stats['total_processed']}")
    print(f"Total black digits extracted: {m3_5_stats['black_digits_extracted']}")
    print(f"Total red digits found (excluded): {m3_5_stats['red_digits_found']}")
    print(f"\nDigit distribution:")
    print(f"  Images with exactly 4 black digits: {m3_5_stats['images_with_4_digits']}")
    print(f"  Images with fewer than 4 digits: {m3_5_stats['images_with_fewer_digits']}")
    print(f"  Images with more than 4 digits: {m3_5_stats['images_with_more_digits']}")
    if m3_5_stats['total_processed'] > 0:
        print(f"  Average black digits per image: {m3_5_stats['black_digits_extracted']/m3_5_stats['total_processed']:.1f}")
        print(f"  Average red digits per image: {m3_5_stats['red_digits_found']/m3_5_stats['total_processed']:.1f}")
    print("="*70)

    print(f"\nOutputs created:")
    print(f"  1. M3 ROI crops: {OUTPUT_M3}/")
    print(f"     - Format: crop_[filename].jpg")
    print(f"  2. M3.5 Black digits only: {OUTPUT_M3_5_DIGITS}/")
    print(f"     - Format: [filename]/digit_0.jpg, digit_1.jpg, ...")
    print(f"     - Chỉ chứa digits MÀU ĐEN, đã loại bỏ digits màu đỏ")
    print(f"  3. M3.5 Black word-level: {OUTPUT_M3_5_WORD}/")
    print(f"     - Images: images/crop_[filename].jpg")
    print(f"     - Labels: labels.csv")
    print("="*70)

    return m3_5_stats


if __name__ == "__main__":
    stats = extract_datasets()

    print("\nNext steps:")
    print("  1. Review black digit extraction in:", OUTPUT_M3_5_DIGITS)
    print("  2. Verify red digits are excluded")
    print("  3. Label M3.5 word dataset:", OUTPUT_M3_5_WORD)
    print("     - Edit labels.csv to replace '?' with actual digit readings")
    print("  4. Use labeled dataset for M4 CRNN training")

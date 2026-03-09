"""
M5: Crop Only Black Digits from ROI Images
Extract only the 4 black digits (integer part), exclude red digit (decimal part)
"""
import os
import sys
import codecs
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Configuration
INPUT_DIR = Path(r"F:\Workspace\Project\data\m3_roi_crops_all")
OUTPUT_DIR = Path(r"F:\Workspace\Project\data\m5_black_digits")
LABELS_FILE = r"F:\Workspace\Project\data\data_4digit.csv"
RESULTS_CSV = Path(r"F:\Workspace\Project\data\m5_crop_results.csv")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("M5: CROP BLACK DIGITS (INTEGER PART) FROM ROI IMAGES")
print("=" * 70)
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

def detect_red_digit_region(img):
    """
    Detect the red digit region on the right side
    Returns: x-coordinate where red digits start
    """
    h, w = img.shape[:2]

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color range (two ranges because red wraps around in HSV)
    # Red 1: 0-10 degrees
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    # Red 2: 170-180 degrees
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # Fallback: use 80% of width
        return int(w * 0.8)

    # Get all red regions
    red_regions = []
    for cnt in contours:
        x, y, w_red, h_red = cv2.boundingRect(cnt)
        if w_red > 5 and h_red > 10:  # Filter small noise
            red_regions.append((x, y, w_red, h_red))

    if len(red_regions) == 0:
        # Fallback: use 80% of width
        return int(w * 0.8)

    # Find the rightmost red region (decimal digits)
    red_regions.sort(key=lambda r: r[0], reverse=True)
    rightmost_red = red_regions[0]

    # Red digit boundary is at the start of the rightmost red region
    red_x_start = rightmost_red[0]

    return red_x_start

def crop_black_digits(img_path, output_dir):
    """
    Crop only the black digits (integer part) from ROI image
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            'filename': img_path.name,
            'status': 'error',
            'error': 'Could not read image'
        }

    h, w = img.shape[:2]

    # Detect red digit region
    red_x_start = detect_red_digit_region(img)

    # Crop only black digits (left part, exclude red digits)
    # Add small margin on right
    crop_x_end = min(red_x_start, w)
    crop_x_end = max(crop_x_end - 5, int(w * 0.75))  # Ensure we keep at least 75%

    black_digits = img[:, :crop_x_end]

    if black_digits.size == 0:
        return {
            'filename': img_path.name,
            'status': 'error',
            'error': 'Empty crop'
        }

    # Save cropped image
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), black_digits)

    crop_h, crop_w = black_digits.shape[:2]

    return {
        'filename': img_path.name,
        'status': 'success',
        'original_size': f'{w}x{h}',
        'crop_size': f'{crop_w}x{crop_h}',
        'red_x_start': red_x_start,
        'crop_ratio': crop_x_end / w
    }

# ==========================================
# MAIN PROCESSING
# ==========================================

# Get all ROI images
image_files = sorted(list(INPUT_DIR.glob('*.jpg')))
print(f"\n[SCAN] Found {len(image_files)} ROI images")

if len(image_files) == 0:
    print("[ERROR] No ROI images found!")
    sys.exit(1)

# Process images
print(f"\n[PROCESS] Cropping black digits...")
print("-" * 70)

results = []
success_count = 0
error_count = 0

for img_path in tqdm(image_files, desc="Cropping black digits"):
    result = crop_black_digits(img_path, OUTPUT_DIR)
    results.append(result)

    if result['status'] == 'success':
        success_count += 1
    else:
        error_count += 1

# Save results
df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Total images: {len(image_files)}")
print(f"Success: {success_count} ({success_count/len(image_files)*100:.1f}%)")
print(f"Errors: {error_count} ({error_count/len(image_files)*100:.1f}%)")
print(f"\nBlack digit images saved to: {OUTPUT_DIR}")
print(f"Results saved to: {RESULTS_CSV}")

# Statistics for successful crops
if success_count > 0:
    success_df = df[df['status'] == 'success']
    print("\n" + "=" * 70)
    print("CROP STATISTICS")
    print("=" * 70)

    # Crop ratio statistics
    print(f"\nCrop ratio (width kept / original width):")
    print(f"  Mean: {success_df['crop_ratio'].mean():.3f}")
    print(f"  Std: {success_df['crop_ratio'].std():.3f}")
    print(f"  Min: {success_df['crop_ratio'].min():.3f}")
    print(f"  Max: {success_df['crop_ratio'].max():.3f}")
    print(f"  Median: {success_df['crop_ratio'].median():.3f}")

    # Red X position statistics
    print(f"\nRed digit X position (pixels from left):")
    print(f"  Mean: {success_df['red_x_start'].mean():.1f}")
    print(f"  Std: {success_df['red_x_start'].std():.1f}")
    print(f"  Min: {success_df['red_x_start'].min():.1f}")
    print(f"  Max: {success_df['red_x_start'].max():.1f}")

print("\n" + "=" * 70)
print("✅ M5 CROP BLACK DIGITS COMPLETED!")
print("=" * 70)

# Show sample results
print(f"\n[SAMPLE RESULTS] (First 10)")
print("-" * 70)
for i, result in enumerate(results[:10], 1):
    status_emoji = "✅" if result['status'] == 'success' else "❌"
    print(f"  {i:2d}. {result['filename'][:60]:60s} {status_emoji}")
    if result['status'] == 'success':
        print(f"      {result['original_size']} → {result['crop_size']} ({result['crop_ratio']*100:.1f}%)")
    else:
        print(f"      Status: {result['status']} - {result.get('error', 'Unknown')}")

print(f"\n💡 Next steps:")
print(f"   1. Review cropped images in: {OUTPUT_DIR}")
print(f"   2. Run updated prepare_dataset.py to create OCR dataset")
print(f"   3. Upload to Colab and train OCR model")

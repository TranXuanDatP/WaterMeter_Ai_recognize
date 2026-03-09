"""
Test M5: Crop Black Digits - Sample Images
Visualize the cropping process on sample images
"""
import os
import sys
import codecs
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Configuration
INPUT_DIR = Path(r"F:\Workspace\Project\data\m3_roi_crops_all")
OUTPUT_DIR = Path(r"F:\Workspace\Project\data\m5_test_samples")
SAMPLE_IMAGES = [
    "crop_meter4_00000_0001e09f7ad5442a832f7b5efb74bf2c.jpg",
    "crop_meter4_00003_000d167bd188450caa39df42ec57fd15.jpg",
    "crop_meter4_00011_0083838ef93d4c7fae8782c755835707.jpg",
]

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_red_digit_region(img):
    """
    Detect the red digit region on the right side
    Returns: x-coordinate where red digits start
    """
    h, w = img.shape[:2]

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color range (two ranges because red wraps around in HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    return red_mask

def visualize_crop(img_name, input_dir, output_dir):
    """
    Visualize the cropping process
    """
    img_path = input_dir / img_name
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"[ERROR] Could not read: {img_name}")
        return

    h, w = img.shape[:2]

    # Detect red region
    red_mask = detect_red_digit_region(img)

    # Find contours in red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find red x position
    red_x_start = int(w * 0.8)  # Default

    if len(contours) > 0:
        red_regions = []
        for cnt in contours:
            x, y, w_red, h_red = cv2.boundingRect(cnt)
            if w_red > 5 and h_red > 10:
                red_regions.append((x, y, w_red, h_red))

        if len(red_regions) > 0:
            red_regions.sort(key=lambda r: r[0], reverse=True)
            red_x_start = red_regions[0][0]

    # Crop black digits
    crop_x_end = min(red_x_start, w)
    crop_x_end = max(crop_x_end - 5, int(w * 0.75))
    black_digits = img[:, :crop_x_end]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # Original image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title(f'Original ({w}x{h})')
    axes[0, 0].axis('off')

    # Red mask
    axes[0, 1].imshow(red_mask, cmap='gray')
    axes[0, 1].set_title('Red Mask (Decimal Digits)')
    axes[0, 1].axis('off')

    # Original with crop line
    img_with_line = img_rgb.copy()
    cv2.line(img_with_line, (crop_x_end, 0), (crop_x_end, h), (255, 0, 0), 3)
    axes[1, 0].imshow(img_with_line)
    axes[1, 0].set_title(f'Crop Line at x={crop_x_end} ({crop_x_end/w*100:.1f}%)')
    axes[1, 0].axis('off')

    # Cropped black digits
    black_rgb = cv2.cvtColor(black_digits, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(black_rgb)
    crop_h, crop_w = black_digits.shape[:2]
    axes[1, 1].set_title(f'Cropped Black Digits ({crop_w}x{crop_h})')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"test_{img_name}", dpi=150, bbox_inches='tight')
    plt.close()

    # Save cropped image
    cv2.imwrite(str(output_dir / f"cropped_{img_name}"), black_digits)

    print(f"\n{'='*70}")
    print(f"Image: {img_name}")
    print(f"{'='*70}")
    print(f"Original size: {w}x{h}")
    print(f"Red X start: {red_x_start}")
    print(f"Crop X end: {crop_x_end}")
    print(f"Crop ratio: {crop_x_end/w*100:.1f}%")
    print(f"Cropped size: {crop_w}x{crop_h}")
    print(f"✅ Saved: test_{img_name}, cropped_{img_name}")

# ==========================================
# MAIN PROCESSING
# ==========================================

print("=" * 70)
print("M5: TEST CROP BLACK DIGITS - SAMPLE IMAGES")
print("=" * 70)

for img_name in SAMPLE_IMAGES:
    visualize_crop(img_name, INPUT_DIR, OUTPUT_DIR)

print("\n" + "=" * 70)
print("✅ TEST COMPLETED!")
print("=" * 70)
print(f"\n📁 Results saved to: {OUTPUT_DIR}")
print(f"   - test_*.png: Visualization images")
print(f"   - cropped_*.jpg: Cropped black digit images")
print(f"\n💡 Please review the visualization images to verify cropping quality")

#!/usr/bin/env python3
"""
Visualize black digits extraction to verify red digits are excluded
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
M3_DIR = r"F:\Workspace\Project\data\m3_roi_crops_new"
M3_5_BLACK_DIR = r"F:\Workspace\Project\data\m3_5_black_digits_only"
OUTPUT_DIR = r"F:\Workspace\Project\results\black_digits_visualization"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("VISUALIZE BLACK DIGITS EXTRACTION")
print("="*70)

# Get samples
m3_files = sorted(list(Path(M3_DIR).glob('*.jpg')))
m3_5_black_folders = sorted([f for f in Path(M3_5_BLACK_DIR).iterdir() if f.is_dir()])

print(f"\nFound {len(m3_files)} M3 ROI crops")
print(f"Found {len(m3_5_black_folders)} M3.5 black digit folders")

# Visualize 5 random samples
num_samples = min(5, len(m3_5_black_folders))
sample_folders = random.sample(m3_5_black_folders, num_samples) if num_samples > 0 else []

for idx, folder in enumerate(sample_folders):
    folder_name = folder.name
    print(f"\n[{idx+1}/{num_samples}] Visualizing: {folder_name}")

    # Find corresponding M3 ROI
    m3_file = Path(M3_DIR) / f"crop_{folder_name}.jpg"

    # Get black digit files
    digit_files = sorted(list(folder.glob('digit_*.jpg')))
    num_black_digits = len(digit_files)

    # Create figure
    fig = plt.figure(figsize=(18, 10))

    # Row 1: M3 ROI (original with all digits)
    ax1 = plt.subplot(3, 1, 1)
    if m3_file.exists():
        m3_img = cv2.imread(str(m3_file))
        # Convert to RGB for matplotlib
        m3_rgb = cv2.cvtColor(m3_img, cv2.COLOR_BGR2RGB)

        # Highlight black vs red regions
        hsv = cv2.cvtColor(m3_img, cv2.COLOR_BGR2HSV)

        # Red masks
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Create overlay
        overlay = m3_rgb.copy()
        overlay[red_mask > 0] = [255, 0, 0]  # Mark red pixels as blue in RGB

        ax1.imshow(m3_rgb)
        ax1.set_title(f'M3 ROI: {m3_file.name} (Original - contains both black & red digits)',
                     fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'M3 ROI not found', ha='center', va='center', fontsize=14)
        ax1.set_title('M3 ROI: Not found', fontsize=12)
    ax1.axis('off')

    # Row 2: Black digits extracted
    ax2 = plt.subplot(3, 1, 2)
    # Combine all black digits horizontally
    if num_black_digits > 0:
        combined_width = sum(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).shape[1] for f in digit_files)
        max_height = max(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).shape[0] for f in digit_files)

        combined = np.ones((max_height, combined_width), dtype=np.uint8) * 255
        x_offset = 0

        for digit_file in digit_files:
            digit = cv2.imread(str(digit_file), cv2.IMREAD_GRAYSCALE)
            h, w = digit.shape
            y_offset = (max_height - h) // 2
            combined[y_offset:y_offset+h, x_offset:x_offset+w] = digit
            x_offset += w

        ax2.imshow(combined, cmap='gray')
        ax2.set_title(f'Black Digits Extracted: {num_black_digits} digits (Red digits EXCLUDED)',
                     fontsize=12, fontweight='bold', color='green')
    else:
        ax2.text(0.5, 0.5, 'No black digits extracted', ha='center', va='center', fontsize=14)
        ax2.set_title('No black digits extracted', fontsize=12)
    ax2.axis('off')

    # Row 3: Individual black digits
    num_show = min(8, num_black_digits)
    for i in range(num_show):
        ax = plt.subplot(3, num_show, num_show * 2 + i + 1)
        digit_img = cv2.imread(str(digit_files[i]), cv2.IMREAD_GRAYSCALE)
        ax.imshow(digit_img, cmap='gray')
        ax.set_title(f'Digit {i}', fontsize=10, color='green')
        ax.axis('off')

    # Hide unused subplots
    for i in range(num_show, 8):
        ax = plt.subplot(3, 8, i + 9)
        ax.axis('off')

    title_text = f'Sample {idx+1}: {folder_name[:40]}... | '
    title_text += f'Black: {num_black_digits}, Red: Excluded'

    plt.suptitle(title_text, fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"viz_black_digits_{idx+1}.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Black digits extracted: {num_black_digits}")
    print(f"  Saved: {save_path}")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"\nVisualizations saved to: {OUTPUT_DIR}")
print("\nEach sample shows:")
print("  - Top: M3 ROI (original with both black & red digits)")
print("  - Middle: Combined black digits only (red EXCLUDED)")
print("  - Bottom: Individual black digits")
print("="*70)
print("\nVerification checklist:")
print("  [ ] Red digits are excluded from extraction")
print("  [ ] Only black digits are extracted")
print("  [ ] Target: 4 black digits per image")
print("  [ ] Quality: Digits are clear and readable")
print("="*70)

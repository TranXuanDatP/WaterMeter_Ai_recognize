#!/usr/bin/env python3
"""
Visualize extracted M3 and M3.5 data for review
"""

import os
import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
M3_DIR = r"F:\Workspace\Project\data\m3_roi_crops_new"
M3_5_DIGITS_DIR = r"F:\Workspace\Project\data\m3_5_black_digits_individual"
M3_5_WORD_DIR = r"F:\Workspace\Project\data\m3_5_word_dataset"
OUTPUT_DIR = r"F:\Workspace\Project\results\extraction_visualization"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("VISUALIZE EXTRACTED M3 & M3.5 DATA")
print("="*70)

# Get samples
m3_files = sorted(list(Path(M3_DIR).glob('*.jpg')))
m3_5_digit_folders = sorted([f for f in Path(M3_5_DIGITS_DIR).iterdir() if f.is_dir()])

print(f"\nFound {len(m3_files)} M3 ROI crops")
print(f"Found {len(m3_5_digit_folders)} M3.5 digit folders")

# Visualize 5 random samples
num_samples = min(5, len(m3_5_digit_folders))
sample_folders = random.sample(m3_5_digit_folders, num_samples) if num_samples > 0 else []

for idx, folder in enumerate(sample_folders):
    folder_name = folder.name
    print(f"\n[{idx+1}/{num_samples}] Visualizing: {folder_name}")

    # Find corresponding M3 ROI
    m3_file = Path(M3_DIR) / f"crop_{folder_name}.jpg"

    # Find corresponding word-level image
    word_file = Path(M3_5_WORD_DIR) / "images" / f"crop_{folder_name}.jpg"

    # Get digit files
    digit_files = sorted(list(folder.glob('digit_*.jpg')))
    num_digits = len(digit_files)

    # Calculate grid size based on number of digits
    num_cols = min(12, num_digits)
    if num_digits > 12:
        num_rows_digits = 2
        num_cols = 12
    else:
        num_rows_digits = 1

    # Create figure with dynamic size
    fig_height = 4 + num_rows_digits * 3
    fig = plt.figure(figsize=(20, fig_height))
    gs = fig.add_gridspec(2 + num_rows_digits, 4, hspace=0.3, wspace=0.2)

    # Top row: M3 ROI
    ax1 = fig.add_subplot(gs[0, :])
    if m3_file.exists():
        m3_img = cv2.imread(str(m3_file))
        ax1.imshow(cv2.cvtColor(m3_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'M3 ROI: {m3_file.name}', fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'M3 ROI not found', ha='center', va='center', fontsize=14)
        ax1.set_title('M3 ROI: Not found', fontsize=12)
    ax1.axis('off')

    # Middle row: Word-level image
    ax2 = fig.add_subplot(gs[1, :])
    if word_file.exists():
        word_img = cv2.imread(str(word_file))
        ax2.imshow(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'M3.5 Word-level: {word_file.name} ({num_digits} digits)', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Word image not found', ha='center', va='center', fontsize=14)
        ax2.set_title('M3.5 Word-level: Not found', fontsize=12)
    ax2.axis('off')

    # Bottom rows: Individual digits
    for i, digit_file in enumerate(digit_files[:12]):  # First row: up to 12 digits
        row = 2 + (i // 4) if num_rows_digits > 1 else 2
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        digit_img = cv2.imread(str(digit_file), cv2.IMREAD_GRAYSCALE)
        ax.imshow(digit_img, cmap='gray')
        ax.set_title(f'Digit {i}', fontsize=10)
        ax.axis('off')

    # Second row for digits if more than 12
    if num_digits > 12:
        for i in range(12, min(24, num_digits)):
            row = 3
            col = (i - 12) % 4
            ax = fig.add_subplot(gs[row, col])

            digit_img = cv2.imread(str(digit_files[i]), cv2.IMREAD_GRAYSCALE)
            ax.imshow(digit_img, cmap='gray')
            ax.set_title(f'Digit {i}', fontsize=10)
            ax.axis('off')

    # Hide unused subplots in last row
    max_cols = 4
    if num_rows_digits > 1:
        used_cols = (num_digits - 12) % 4 if num_digits > 12 else num_digits % 4
        if used_cols != 0:
            for col in range(used_cols, max_cols):
                ax = fig.add_subplot(gs[3, col])
                ax.axis('off')

    plt.suptitle(f'Extraction Sample {idx+1}: {folder_name[:50]}...',
                 fontsize=14, fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, f"viz_sample_{idx+1}.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Digits extracted: {len(digit_files)}")
    print(f"  Saved: {save_path}")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"\nVisualizations saved to: {OUTPUT_DIR}")
print("\nEach sample shows:")
print("  - Top: M3 ROI crop")
print("  - Middle: M3.5 Word-level image (combined digits)")
print("  - Bottom: Individual extracted digits")
print("="*70)
print("\nNext steps:")
print("  1. Review visualizations to check extraction quality")
print("  2. If good, proceed with labeling M3.5 word dataset")
print("  3. If bad, adjust extraction parameters in extract_m3_m3_5_dataset.py")
print("="*70)

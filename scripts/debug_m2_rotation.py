#!/usr/bin/env python3
"""
Debug M2 Orientation - Check predictions and fix logic
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_DIR = r"F:\Workspace\Project\data\m2_crops"
OUTPUT_DIR = r"F:\Workspace\Project\results\m2_debug"
MAX_IMAGES = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("M2 ORIENTATION DEBUG")
print("="*70)

# Load one image to test
image_files = list(Path(INPUT_DIR).glob('*.jpg'))[:MAX_IMAGES]

for img_path in image_files:
    filename = os.path.basename(img_path)
    print(f"\n{'='*70}")
    print(f"Testing: {filename}")
    print(f"{'='*70}")

    # Load original
    img_original = cv2.imread(str(img_path))
    print(f"Original shape: {img_original.shape}")

    # Display original
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title(f"Original - {filename}")
    plt.axis('off')

    # Test different rotation angles to see which is correct
    angles_to_test = [0, -90, -180, -270]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for idx, test_angle in enumerate(angles_to_test):
        ax = axes[idx // 2, idx % 2]

        # Rotate with test angle
        h, w = img_original.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, test_angle, 1.0)
        rotated = cv2.warpAffine(img_original, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
        ax.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Rotated {test_angle}°")
        ax.axis('off')

    plt.suptitle(f"Testing Different Rotations - {filename}", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"debug_{filename}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved debug visualization: {save_path}")

    # Analyze what should be correct rotation
    print("\nManual Analysis:")
    print("  - Original: Check orientation")
    print("  - Try rotating -90°, -180°, -270° to see which is upright")
    print("  - Compare with the model's prediction")

print("\n" + "="*70)
print("DEBUG COMPLETE")
print("="*70)
print(f"\nCheck outputs at: {OUTPUT_DIR}")

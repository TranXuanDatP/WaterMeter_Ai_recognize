"""
Create Before/After Visualization for CORRECTED M2 Rotation

Shows sample images comparing original vs CORRECTED rotated results.
"""
import sys
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Paths
ORIGINAL_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
ROTATED_DIR = Path(r"F:\Workspace\Project\results\m2_rotated_corrected\aligned_images")
RESULTS_CSV = Path(r"F:\Workspace\Project\results\m2_rotated_corrected\rotation_results.csv")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\m2_rotation_corrected_visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("M2 ROTATION VISUALIZATION - CORRECTED VERSION")
print("=" * 80)

# Load rotation results
results_df = pd.read_csv(RESULTS_CSV)
print(f"\nLoaded {len(results_df)} rotation results")

# Select samples with significant rotation (|correction_angle| > 30)
significant_rotations = results_df[
    (results_df['correction_angle'].abs() > 30) &
    (results_df['status'] == 'success')
]

if len(significant_rotations) > 20:
    sample_df = significant_rotations.sample(n=20, random_state=42)
else:
    sample_df = results_df[results_df['status'] == 'success'].sample(n=min(20, len(results_df)), random_state=42)

print(f"Selected {len(sample_df)} samples with significant rotation")

print(f"\nGenerating before/after comparisons...")

for idx, row in sample_df.iterrows():
    img_name = row['filename']
    detected_angle = row['detected_angle']
    correction_angle = row['correction_angle']

    # Load images
    original_path = ORIGINAL_DIR / img_name
    rotated_path = ROTATED_DIR / img_name

    original_img = cv2.imread(str(original_path))
    rotated_img = cv2.imread(str(rotated_path))

    if original_img is None or rotated_img is None:
        continue

    # Resize for display (maintain aspect ratio)
    max_height = 300

    h1, w1 = original_img.shape[:2]
    h2, w2 = rotated_img.shape[:2]

    if h1 > max_height:
        scale1 = max_height / h1
        original_img = cv2.resize(original_img, (int(w1 * scale1), max_height))

    if h2 > max_height:
        scale2 = max_height / h2
        rotated_img = cv2.resize(rotated_img, (int(w2 * scale2), max_height))

    # Add labels
    original_labeled = cv2.copyMakeBorder(original_img, 50, 10, 10, 10,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(original_labeled, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)
    angle_text1 = f"Detected: {detected_angle:.1f} deg"
    cv2.putText(original_labeled, angle_text1, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2)

    rotated_labeled = cv2.copyMakeBorder(rotated_img, 50, 10, 10, 10,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(rotated_labeled, "ROTATED (CORRECTED)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 200, 0), 2)
    angle_text2 = f"Correction: {correction_angle:.1f} deg"
    cv2.putText(rotated_labeled, angle_text2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2)

    # Concatenate side by side
    combined = np.hstack([original_labeled, rotated_labeled])

    # Add filename at top
    combined = cv2.copyMakeBorder(combined, 60, 10, 10, 10,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(combined, img_name[:40], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2)

    # Save
    output_path = OUTPUT_DIR / f"corrected_{idx:03d}_{img_name}"
    cv2.imwrite(str(output_path), combined)

    print(f"  Saved: {output_path.name}")

print(f"\n{'=' * 80}")
print(f"VISUALIZATION COMPLETE!")
print(f"{'=' * 80}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nGenerated {len(sample_df)} comparison images showing:")
print(f"  - Left: ORIGINAL image with detected angle")
print(f"  - Right: ROTATED image with correction angle (CORRECTED LOGIC)")
print(f"  - Top: Filename for reference")
print(f"\nCORRECTED LOGIC:")
print(f"  Normalize angle to [-180, 180]")
print(f"  Rotate counter-clockwise by angle (correction = -angle)")
print(f"  This matches the original metadata pattern!")
print(f"{'=' * 80}")

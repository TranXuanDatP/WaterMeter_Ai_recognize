"""
Create Before/After Visualization for M2 Rotation

Shows sample images comparing original vs rotated results.
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
ROTATED_DIR = Path(r"F:\Workspace\Project\results\m2_rotated_data_4digit2\aligned_images")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\m2_rotation_visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("M2 ROTATION VISUALIZATION - Before/After")
print("=" * 80)

# Load rotation results
results_df = pd.read_csv(r"F:\Workspace\Project\results\m2_rotated_data_4digit2\rotation_results.csv")
print(f"\nLoaded {len(results_df)} rotation results")

# Select interesting samples (with significant rotation)
significant_rotations = results_df[
    (results_df['actual_rotation'].abs() > 10) &
    (results_df['status'] == 'success')
]

if len(significant_rotations) > 20:
    sample_df = significant_rotations.sample(n=20, random_state=42)
else:
    sample_df = results_df.sample(n=min(20, len(results_df)), random_state=42)

print(f"Selected {len(sample_df)} samples for visualization")

print(f"\nGenerating before/after comparisons...")

for idx, row in sample_df.iterrows():
    img_name = row['filename']
    angle = row['actual_rotation']

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
    original_labeled = cv2.copyMakeBorder(original_img, 40, 10, 10, 10,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(original_labeled, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    rotated_labeled = cv2.copyMakeBorder(rotated_img, 40, 10, 10, 10,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
    angle_text = f"ROTATED: {angle:.1f} deg"
    cv2.putText(rotated_labeled, angle_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # Concatenate side by side
    combined = np.hstack([original_labeled, rotated_labeled])

    # Add filename at top
    combined = cv2.copyMakeBorder(combined, 50, 10, 10, 10,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(combined, img_name[:40], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2)

    # Save
    output_path = OUTPUT_DIR / f"comparison_{idx:03d}_{img_name}"
    cv2.imwrite(str(output_path), combined)

    print(f"  Saved: {output_path.name}")

# Create summary grid
print(f"\nCreating summary grid...")

# Select 6 best samples for grid
grid_samples = sample_df.head(6)
grid_cols = 2
grid_rows = 3
cell_height = 400
cell_width = 900

grid_img = np.ones((grid_rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8) * 255

for idx, (_, row) in enumerate(grid_samples.iterrows()):
    row_idx = idx // grid_cols
    col_idx = idx % grid_cols

    img_name = row['filename']
    angle = row['actual_rotation']

    original_path = ORIGINAL_DIR / img_name
    rotated_path = ROTATED_DIR / img_name

    original_img = cv2.imread(str(original_path))
    rotated_img = cv2.imread(str(rotated_path))

    if original_img is None or rotated_img is None:
        continue

    # Resize to fit
    h1, w1 = original_img.shape[:2]
    h2, w2 = rotated_img.shape[:2]

    target_h = 320
    original_resized = cv2.resize(original_img, (int(w1 * target_h / h1), target_h))
    rotated_resized = cv2.resize(rotated_img, (int(w2 * target_h / h2), target_h))

    # Add labels
    original_labeled = cv2.copyMakeBorder(original_resized, 30, 5, 5, 5,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(original_labeled, "ORIGINAL", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)

    rotated_labeled = cv2.copyMakeBorder(rotated_resized, 30, 5, 5, 5,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(rotated_labeled, f"ROTATED ({angle:.1f})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

    # Combine
    cell = np.hstack([original_labeled, rotated_labeled])
    cell = cv2.copyMakeBorder(cell, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(cell, img_name[:30], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Resize to fit grid cell
    cell = cv2.resize(cell, (cell_width - 20, cell_height - 20))

    y_start = row_idx * cell_height + 10
    y_end = y_start + cell.shape[0]
    x_start = col_idx * cell_width + 10
    x_end = x_start + cell.shape[1]

    grid_img[y_start:y_end, x_start:x_end] = cell

# Save summary grid
summary_path = OUTPUT_DIR / "summary_grid.png"
cv2.imwrite(str(summary_path), grid_img)

# Create statistics visualization
print(f"\nCreating angle distribution chart...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Plot angle distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
angles = results_df['actual_rotation'].values
ax1.hist(angles, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Rotation Angle (degrees)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Rotation Angles', fontsize=14, fontweight='bold')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No rotation')
ax1.axvline(np.mean(angles), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(angles):.2f}')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Box plot
box = ax2.boxplot([angles], labels=['Rotation Angles'], patch_artist=True)
box['boxes'][0].set_facecolor('lightblue')
box['boxes'][0].set_alpha(0.7)
ax2.set_ylabel('Angle (degrees)', fontsize=12)
ax2.set_title('Rotation Angle Statistics', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Add statistics text
stats_text = f"""Statistics:
  Mean:   {np.mean(angles):.2f} deg
  Std:    {np.std(angles):.2f} deg
  Median: {np.median(angles):.2f} deg
  Min:    {np.min(angles):.2f} deg
  Max:    {np.max(angles):.2f} deg"""

ax2.text(0.15, 0.05, stats_text, transform=ax2.transAxes,
         fontsize=11, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
chart_path = OUTPUT_DIR / "angle_distribution.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'=' * 80}")
print(f"VISUALIZATION COMPLETE!")
print(f"{'=' * 80}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nGenerated files:")
print(f"  - {len(sample_df)} before/after comparison images")
print(f"  - 1 summary grid (6 samples)")
print(f"  - 1 angle distribution chart")
print(f"\nTo view:")
print(f"  1. Open folder: {OUTPUT_DIR}")
print(f"  2. View summary_grid.png for overview")
print(f"  3. View comparison_*.jpg for detailed samples")
print(f"  4. View angle_distribution.png for statistics")
print(f"{'=' * 80}")

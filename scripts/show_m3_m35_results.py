"""
Display M3 and M3.5 Results Visualization

Shows sample images from M3 (ROI Detection) and M3.5 (Black Digit Extraction) stages.
"""
import sys
import cv2
import numpy as np
import random
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Paths
M3_DIR = Path(r"F:\Workspace\Project\results\full_pipeline_data_4digit2\m3_roi_crops")
M3_5_DIR = Path(r"F:\Workspace\Project\results\full_pipeline_data_4digit2\m3_5_black_digits")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\m3_m35_visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("M3 & M3.5 RESULTS VISUALIZATION")
print("=" * 80)

# Get image lists
m3_images = list(M3_DIR.glob('*.jpg')) + list(M3_DIR.glob('*.png'))
m3_5_images = list(M3_5_DIR.glob('*.jpg')) + list(M3_5_DIR.glob('*.png'))

print(f"\nM3 (ROI Detection): {len(m3_images)} images")
print(f"M3.5 (Black Digits): {len(m3_5_images)} images")

# Select random samples
num_samples = 10
sample_indices = random.sample(range(len(m3_images)), min(num_samples, len(m3_images)))

print(f"\nGenerating visualization for {len(sample_indices)} random samples...")

for idx, img_idx in enumerate(sample_indices):
    img_name = m3_images[img_idx].name

    # Load images
    m3_img = cv2.imread(str(m3_images[img_idx]))
    m3_5_img = cv2.imread(str(M3_5_DIR / img_name))

    if m3_img is None or m3_5_img is None:
        continue

    # Resize for display
    h1, w1 = m3_img.shape[:2]
    h2, w2 = m3_5_img.shape[:2]

    # Make same height
    target_height = 200
    m3_resized = cv2.resize(m3_img, (int(w1 * target_height / h1), target_height))
    m3_5_resized = cv2.resize(m3_5_img, (int(w2 * target_height / h2), target_height))

    # Add labels
    m3_labeled = cv2.copyMakeBorder(m3_resized, 30, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(m3_labeled, "M3: ROI Detection", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    m3_5_labeled = cv2.copyMakeBorder(m3_5_resized, 30, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(m3_5_labeled, "M3.5: Black Digits", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Concatenate horizontally
    combined = np.hstack([m3_labeled, m3_5_labeled])

    # Add filename
    combined = cv2.copyMakeBorder(combined, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(combined, f"Sample {idx+1}: {img_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Save
    output_path = OUTPUT_DIR / f"comparison_{idx+1:02d}_{img_name}"
    cv2.imwrite(str(output_path), combined)

    print(f"  Saved: {output_path.name}")

# Create summary image (grid layout)
print(f"\nCreating summary grid...")
grid_cols = 2
grid_rows = min(5, len(sample_indices))
cell_height = 280
cell_width = 800

grid_img = np.ones((grid_rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8) * 255

for idx, img_idx in enumerate(sample_indices[:grid_rows * grid_cols]):
    row = idx // grid_cols
    col = idx % grid_cols

    img_name = m3_images[img_idx].name

    m3_img = cv2.imread(str(m3_images[img_idx]))
    m3_5_img = cv2.imread(str(M3_5_DIR / img_name))

    if m3_img is None or m3_5_img is None:
        continue

    h1, w1 = m3_img.shape[:2]
    h2, w2 = m3_5_img.shape[:2]

    m3_resized = cv2.resize(m3_img, (int(w1 * 200 / h1), 200))
    m3_5_resized = cv2.resize(m3_5_img, (int(w2 * 200 / h2), 200))

    # Add side by side
    cell = np.hstack([m3_resized, m3_5_resized])

    # Add border and label
    cell = cv2.copyMakeBorder(cell, 60, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(cell, img_name[:30], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(cell, "M3 ROI", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(cell, "M3.5 Digits", (cell.shape[1]//2 + 10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Place in grid
    y_start = row * cell_height
    y_end = y_start + cell.shape[0]
    x_start = col * cell_width
    x_end = x_start + cell.shape[1]

    # Resize to fit cell
    cell = cv2.resize(cell, (cell_width - 20, cell_height - 20))
    grid_img[y_start+10:y_end-10, x_start+10:x_end-10] = cell

# Save summary grid
summary_path = OUTPUT_DIR / "summary_grid.png"
cv2.imwrite(str(summary_path), grid_img)

print(f"\n{'=' * 80}")
print(f"VISUALIZATION COMPLETE!")
print(f"{'=' * 80}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nGenerated files:")
print(f"  - {len(sample_indices)} comparison images (M3 vs M3.5 side-by-side)")
print(f"  - 1 summary grid image")
print(f"\nTo view images:")
print(f"  1. Open folder: {OUTPUT_DIR}")
print(f"  2. View summary_grid.png for overview")
print(f"  3. View comparison_*.jpg for detailed samples")
print(f"{'=' * 80}")

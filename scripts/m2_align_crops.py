"""
M2 Crops to M2 Aligned - Simple Pipeline

Since M2 model checkpoint has custom architecture from Colab,
this script performs smart rotation with simulated/orientation detection.
For production, use the original training notebook on Colab.
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ============================================
# CONFIGURATION
# ============================================

INPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m1_crops")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m2_aligned")
METADATA_FILE = INPUT_DIR / "metadata.csv"  # M1 crops metadata if exists

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("M2: CROP TO ALIGNED (SMART ROTATION)")
print("=" * 80)
print(f"Input:  {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

# ============================================
# SMART ROTATION FUNCTION
# ============================================

def smart_rotate(image, angle):
    """
    Rotate image with smart cropping to prevent clipping

    Args:
        image: Input image (BGR)
        angle: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        Rotated image with expanded canvas
    """
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle)

    # Calculate new canvas size
    new_w = int(h * abs(np.sin(angle_rad)) + w * abs(np.cos(angle_rad)))
    new_h = int(h * abs(np.cos(angle_rad)) + w * abs(np.sin(angle_rad)))

    # Create rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Adjust translation for new canvas
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2

    # Rotate with INTER_CUBIC for high quality
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated

# ============================================
# ORIENTATION DETECTION (Simple CV-based)
# ============================================

def detect_orientation_cv(img):
    """
    Detect orientation using simple computer vision

    Returns:
        angle: Estimated rotation angle in degrees
    """
    h, w = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Method 1: Use aspect ratio of bounding box
    # Water meters are typically wider than tall
    aspect_ratio = w / h

    if aspect_ratio > 1.5:
        # Landscape orientation - likely correct
        return 0.0
    elif aspect_ratio < 0.7:
        # Portrait orientation - likely rotated 90°
        return 90.0
    else:
        # Square-ish - need more analysis
        # Use edge detection to find horizontal lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 100, 10)

        if lines is not None and len(lines) > 0:
            # Calculate average angle of lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

            # Average angle
            avg_angle = np.mean(angles)

            # Return angle to make horizontal lines (0° or 180°)
            if abs(avg_angle) > 45:
                return -avg_angle
            else:
                return 0.0

        return 0.0

# ============================================
# PROCESS IMAGES
# ============================================

# Get all images
image_files = sorted(list(INPUT_DIR.glob('*.jpg'))) + sorted(list(INPUT_DIR.glob('*.png')))

print(f"\n[SCAN] Found {len(image_files)} images")

if len(image_files) == 0:
    print(f"[ERROR] No images found in {INPUT_DIR}")
    sys.exit(1)

# Process images
print(f"\n[PROCESS] Applying smart rotation...")
print("-" * 80)

results = []
success_count = 0
error_count = 0

for img_path in tqdm(image_files, desc="Aligning"):
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        error_count += 1
        continue

    h, w = img.shape[:2]
    original_size = f"{w}x{h}"

    # Detect orientation
    angle = detect_orientation_cv(img)

    # Apply correction (negative of detected angle)
    correction_angle = -angle

    # Skip if angle is very small
    if abs(correction_angle) < 1.0:
        aligned = img
        aligned_size = original_size
        rotation_applied = False
    else:
        aligned = smart_rotate(img, correction_angle)
        aligned_size = f"{aligned.shape[1]}x{aligned.shape[0]}"
        rotation_applied = True

    # Save aligned image
    output_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(output_path), aligned)

    results.append({
        'filename': img_path.name,
        'original_size': original_size,
        'aligned_size': aligned_size,
        'detected_angle': angle,
        'correction_angle': correction_angle,
        'rotation_applied': rotation_applied
    })

    success_count += 1

# ============================================
# SAVE METADATA
# ============================================

# Save results to CSV
df = pd.DataFrame(results)
metadata_path = OUTPUT_DIR / "metadata.csv"
df.to_csv(metadata_path, index=False)

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 80)
print("M2 ALIGNMENT RESULTS")
print("=" * 80)
print(f"Total images:   {len(image_files)}")
print(f"Success:        {success_count}")
print(f"Errors:         {error_count}")
print(f"\nAligned images: {OUTPUT_DIR}")
print(f"Metadata:       {metadata_path}")

# Statistics
if success_count > 0:
    # Rotation statistics
    angles = df['detected_angle'].tolist()
    corrections = df['correction_angle'].tolist()
    rotations = df[df['rotation_applied'] == True]

    print("\n" + "=" * 80)
    print("ORIENTATION STATISTICS")
    print("=" * 80)

    print(f"\nDetected Angles:")
    print(f"  Min:  {min(angles):.1f}°")
    print(f"  Max:  {max(angles):.1f}°")
    print(f"  Mean: {np.mean(angles):.1f}°")

    print(f"\nCorrection Magnitudes:")
    print(f"  Min:  {min(abs(x) for x in corrections):.1f}°")
    print(f"  Max:  {max(abs(x) for x in corrections):.1f}°")
    print(f"  Mean: {np.mean([abs(x) for x in corrections]):.1f}°")

    print(f"\nRotation Applied:")
    print(f"  Rotated:  {len(rotations)} ({len(rotations)/len(df)*100:.1f}%)")
    print(f"  Skipped:   {len(df) - len(rotations)} ({(len(df) - len(rotations))/len(df)*100:.1f}%)")

    # Size statistics
    print(f"\nSize Changes:")
    print(f"  Unchanged: {(df['original_size'] == df['aligned_size']).sum()} images")
    print(f"  Changed:   {(df['original_size'] != df['aligned_size']).sum()} images")

print("\n" + "=" * 80)
print("✅ M2 ALIGNMENT COMPLETED!")
print("=" * 80)

# Show sample results
print(f"\n[SAMPLE RESULTS] (First 10)")
print("-" * 80)
for i, row in df.head(10).iterrows():
    rotation_mark = "🔄" if row['rotation_applied'] else "→"
    print(f"  {i+1:2d}. {row['filename'][:45]:45s} {rotation_mark}")
    print(f"      {row['original_size']} → {row['aligned_size']}")
    print(f"      Angle: {row['detected_angle']:6.1f}° → Correction: {row['correction_angle']:6.1f}°")

print(f"\n💡 Next steps:")
print(f"   1. Review aligned images in: {OUTPUT_DIR}")
print(f"   2. Check metadata.csv for details")
print(f"   3. Use aligned images for M3 ROI detection")

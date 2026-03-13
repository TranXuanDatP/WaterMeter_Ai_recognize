"""
Test Meter Rotation with Crop to Original Size

This script demonstrates the smart_rotate function with crop_to_original option,
which rotates the image and crops it back to maintain focus on the meter.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.image_rotation import smart_rotate

# Test image
INPUT_IMAGE = Path(r"F:\Workspace\Project\results\test_pipeline\m1_crops\meter4_00000_validate_00000_00385501ab4d419fa7b0bdf0d9f8451f.jpg")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\meter_rotation_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("METER ROTATION WITH CROP TEST")
print("=" * 80)

# Load test image
img = cv2.imread(str(INPUT_IMAGE))
if img is None:
    print(f"ERROR: Could not load image from {INPUT_IMAGE}")
    sys.exit(1)

h, w = img.shape[:2]
print(f"\nOriginal image: {w}x{h}")

# Test different rotation strategies
test_angles = [15, 23.1, 45, 90]

print("\n" + "=" * 80)
print("COMPARISON: With vs Without Crop")
print("=" * 80)

for angle in test_angles:
    print(f"\n[Angle: {angle}°]")

    # Strategy 1: Smart rotate WITHOUT crop (expanded canvas)
    rotated_expanded = smart_rotate(img, angle, expand_canvas=True, crop_to_original=False)
    print(f"  Expanded canvas: {rotated_expanded.shape[1]}x{rotated_expanded.shape[0]}")

    # Strategy 2: Smart rotate WITH crop (original size)
    rotated_cropped = smart_rotate(img, angle, expand_canvas=True, crop_to_original=True)
    print(f"  Cropped to original: {rotated_cropped.shape[1]}x{rotated_cropped.shape[0]}")

    # Strategy 3: Smart rotate WITH crop and padding (zoom out a bit)
    rotated_padded = smart_rotate(img, angle, expand_canvas=True, crop_to_original=True, crop_padding=10)
    print(f"  Cropped with padding: {rotated_padded.shape[1]}x{rotated_padded.shape[0]}")

    # Create comparison visualization
    # Resize all to same height for display
    target_h = 200

    # Original
    img_resized = cv2.resize(img, (int(w * target_h / h), target_h))

    # Expanded (may be larger or smaller, resize to target_h)
    scale_exp = target_h / rotated_expanded.shape[0]
    exp_resized = cv2.resize(rotated_expanded, (int(rotated_expanded.shape[1] * scale_exp), target_h))

    # Cropped (should be original size)
    crop_resized = cv2.resize(rotated_cropped, (int(rotated_cropped.shape[1] * target_h / rotated_cropped.shape[0]), target_h))

    # Padded
    pad_resized = cv2.resize(rotated_padded, (int(rotated_padded.shape[1] * target_h / rotated_padded.shape[0]), target_h))

    # Add labels
    cv2.putText(img_resized, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(exp_resized, "EXPANDED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(crop_resized, "CROPPED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(pad_resized, "PADDED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Concatenate horizontally
    comparison = np.hstack([img_resized, exp_resized, crop_resized, pad_resized])

    # Save
    output_path = OUTPUT_DIR / f"comparison_{angle}deg.jpg"
    cv2.imwrite(str(output_path), comparison)
    print(f"  Saved: {output_path.name}")

# Test with negative angles (clockwise rotation)
print("\n" + "=" * 80)
print("CLOCKWISE ROTATION TEST")
print("=" * 80)

for angle in [-15, -23.1, -45]:
    print(f"\n[Angle: {angle}°]")

    rotated_cropped = smart_rotate(img, angle, expand_canvas=True, crop_to_original=True)
    print(f"  Rotated and cropped: {rotated_cropped.shape[1]}x{rotated_cropped.shape[0]}")

    # Create comparison
    target_h = 250
    img_resized = cv2.resize(img, (int(w * target_h / h), target_h))
    rot_resized = cv2.resize(rotated_cropped, (int(rotated_cropped.shape[1] * target_h / rotated_cropped.shape[0]), target_h))

    cv2.putText(img_resized, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(rot_resized, f"{angle} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    comparison = np.hstack([img_resized, rot_resized])
    output_path = OUTPUT_DIR / f"clockwise_{abs(angle)}deg.jpg"
    cv2.imwrite(str(output_path), comparison)
    print(f"  Saved: {output_path.name}")

# Test crop_padding parameter
print("\n" + "=" * 80)
print("CROP PADDING TEST (23.1 degrees)")
print("=" * 80)

angle = 23.1
padding_values = [-20, -10, 0, 10, 20]

comparison_images = []
for padding in padding_values:
    rotated = smart_rotate(img, angle, expand_canvas=True, crop_to_original=True, crop_padding=padding)

    # Resize all to same height for display
    target_h = 250
    scale = target_h / rotated.shape[0]
    rotated_resized = cv2.resize(rotated, (int(rotated.shape[1] * scale), target_h))

    # Add label
    labeled = rotated_resized.copy()
    label_text = f"Padding: {padding}"
    cv2.putText(labeled, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    comparison_images.append(labeled)
    print(f"  Padding {padding:3d}: {rotated.shape[1]}x{rotated.shape[0]}")

# Concatenate all
padding_comparison = np.hstack(comparison_images)
output_path = OUTPUT_DIR / "padding_comparison.jpg"
cv2.imwrite(str(output_path), padding_comparison)
print(f"  Saved: {output_path.name}")

print("\n" + "=" * 80)
print("TEST COMPLETED!")
print("=" * 80)
print(f"\nAll output images saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*.jpg")):
    print(f"  - {file.name}")

print("\n" + "=" * 80)
print("USAGE RECOMMENDATION FOR METER IMAGES:")
print("=" * 80)
print("""
For water meter alignment, use:

    from src.utils.image_rotation import smart_rotate

    # Option 1: Rotate and crop to original size (RECOMMENDED)
    aligned = smart_rotate(img, angle, crop_to_original=True)

    # Option 2: Rotate with small padding to keep more context
    aligned = smart_rotate(img, angle, crop_to_original=True, crop_padding=10)

    # Option 3: Rotate without cropping (keeps full image)
    aligned = smart_rotate(img, angle, crop_to_original=False)

The crop_to_original=True option maintains the original image size
and keeps the focus on the meter, which is ideal for:
- M2 alignment
- M3 ROI detection
- M4/M5 processing
""")

"""
Demo script for smart rotation utilities

This script demonstrates how to use the smart rotation functions
with actual images.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.image_rotation import (
    smart_rotate,
    rotate_with_crop,
    auto_rotate,
    get_minimal_rotation_angle
)

# Test image
INPUT_IMAGE = Path(r"F:\Workspace\Project\results\test_pipeline\m1_crops\meter4_00000_validate_00000_00385501ab4d419fa7b0bdf0d9f8451f.jpg")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\smart_rotate_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SMART ROTATION UTILITIES DEMO")
print("=" * 80)

# Load test image
img = cv2.imread(str(INPUT_IMAGE))
if img is None:
    print(f"ERROR: Could not load image from {INPUT_IMAGE}")
    sys.exit(1)

h, w = img.shape[:2]
print(f"\nOriginal image: {w}x{h}")

# Test 1: Basic smart rotation
print("\n[Test 1] Basic Smart Rotation")
print("-" * 40)
angle = 23.1
rotated = smart_rotate(img, -angle)
print(f"Rotated by {-angle} degrees")
print(f"New size: {rotated.shape[1]}x{rotated.shape[0]}")
cv2.imwrite(str(OUTPUT_DIR / "1_basic_rotation.jpg"), rotated)

# Test 2: Rotation with different interpolation methods
print("\n[Test 2] Interpolation Methods Comparison")
print("-" * 40)

methods = {
    'INTER_NEAREST': cv2.INTER_NEAREST,
    'INTER_LINEAR': cv2.INTER_LINEAR,
    'INTER_CUBIC': cv2.INTER_CUBIC
}

for name, method in methods.items():
    rotated = smart_rotate(img, -23.1, interpolation=method)
    output_path = OUTPUT_DIR / f"2_interp_{name.lower()}.jpg"
    cv2.imwrite(str(output_path), rotated)
    print(f"  {name}: saved to {output_path.name}")

# Test 3: Rotation with crop back to original size
print("\n[Test 3] Rotation with Crop")
print("-" * 40)
rotated_cropped = rotate_with_crop(img, -23.1, crop_to_original=True)
print(f"Rotated and cropped: {rotated_cropped.shape[1]}x{rotated_cropped.shape[0]}")
cv2.imwrite(str(OUTPUT_DIR / "3_rotated_cropped.jpg"), rotated_cropped)

# Test 4: Multiple angles comparison
print("\n[Test 4] Multiple Angles Comparison")
print("-" * 40)
angles = [0, 10, 23.1, 45, 90, 180]

# Create a grid comparison
grid_images = []
for angle in angles:
    rotated = smart_rotate(img, angle)
    # Resize all to same height for display
    target_h = 200
    scale = target_h / rotated.shape[0]
    target_w = int(rotated.shape[1] * scale)
    resized = cv2.resize(rotated, (target_w, target_h))

    # Add angle label
    labeled = resized.copy()
    cv2.putText(labeled, f"{angle} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    grid_images.append(labeled)

# Concatenate all images
grid = np.hstack(grid_images)
cv2.imwrite(str(OUTPUT_DIR / "4_angles_grid.jpg"), grid)
print(f"Grid comparison saved: {len(angles)} angles tested")

# Test 5: Minimal rotation calculation
print("\n[Test 5] Minimal Rotation Calculation")
print("-" * 40)
test_angles = [350, 356.2, 345.9, 10, 23.1, 180, 270]

print(f"{'Current Angle':<15} {'To Upright (0°)':<15}")
for angle in test_angles:
    correction = get_minimal_rotation_angle(angle, 0)
    print(f"  {angle:7.1f}°      -> {correction:7.1f}°")

# Test 6: Auto-rotate examples
print("\n[Test 6] Auto-Rotate Examples")
print("-" * 40)

examples = [
    (350, "Near upright (350° -> -10° rotation)"),
    (23.1, "Rotated 23.1° -> -23.1° correction"),
    (90, "90° rotation -> -90° correction"),
    (180, "Upside down (180° -> 180° rotation)")
]

for current_angle, description in examples:
    aligned = auto_rotate(img, current_angle, target_angle=0.0)
    filename = f"6_auto_rotate_{current_angle}.jpg"
    cv2.imwrite(str(OUTPUT_DIR / filename), aligned)
    print(f"  {filename}: {description}")

# Test 7: Create before/after comparison
print("\n[Test 7] Before/After Comparison")
print("-" * 40)

# Create test images at different angles
test_angles_for_comparison = [15, 30, 45, 90]
comparison_images = []

for test_angle in test_angles_for_comparison:
    # Rotate image to create "distorted" version
    distorted = smart_rotate(img, test_angle)

    # Auto-rotate back
    corrected = auto_rotate(distorted, test_angle, target_angle=0.0)

    # Resize for comparison
    target_h = 250
    scale1 = target_h / img.shape[0]
    scale2 = target_h / distorted.shape[0]
    scale3 = target_h / corrected.shape[0]

    img_resized = cv2.resize(img, (int(img.shape[1] * scale1), target_h))
    distorted_resized = cv2.resize(distorted, (int(distorted.shape[1] * scale2), target_h))
    corrected_resized = cv2.resize(corrected, (int(corrected.shape[1] * scale3), target_h))

    # Add labels
    cv2.putText(img_resized, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(distorted_resized, f"{test_angle} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(corrected_resized, "CORRECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Concatenate
    comparison = np.hstack([img_resized, distorted_resized, corrected_resized])
    cv2.imwrite(str(OUTPUT_DIR / f"7_comparison_{test_angle}deg.jpg"), comparison)
    print(f"  Saved: 7_comparison_{test_angle}deg.jpg")

print("\n" + "=" * 80)
print("DEMO COMPLETED!")
print("=" * 80)
print(f"\nAll output images saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*.jpg")):
    print(f"  - {file.name}")

print("\n" + "=" * 80)
print("Usage Examples:")
print("-" * 80)
print("""
# Basic usage:
from src.utils.image_rotation import smart_rotate
aligned = smart_rotate(img, -23.1)

# Auto-rotate from detected angle to upright:
from src.utils.image_rotation import auto_rotate
aligned = auto_rotate(img, current_angle=23.1, target_angle=0.0)

# Get minimal rotation angle:
from src.utils.image_rotation import get_minimal_rotation_angle
correction = get_minimal_rotation_angle(350, 0)  # Returns 10.0
""")

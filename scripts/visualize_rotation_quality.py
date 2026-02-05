"""
Visualize Rotation Quality with Grid Overlay

This script creates a visual assessment of rotation quality by:
1. Detecting the meter box corners
2. Drawing horizontal/vertical reference lines
3. Measuring angle deviation of meter edges
4. Creating before/after comparison with overlay
"""

import cv2
import numpy as np
from pathlib import Path
import math

# Add project path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.m2_orientation_alignment.inference import M2Inference


def draw_grid_overlay(image: np.ndarray, grid_spacing: int = 100, color: tuple = (0, 255, 0), thickness: int = 1):
    """
    Draw grid overlay on image to visualize alignment.

    Args:
        image: Input image
        grid_spacing: Spacing between grid lines
        color: Line color (BGR)
        thickness: Line thickness

    Returns:
        Image with grid overlay
    """
    overlay = image.copy()
    h, w = image.shape[:2]

    # Vertical lines
    for x in range(0, w, grid_spacing):
        cv2.line(overlay, (x, 0), (x, h), color, thickness)

    # Horizontal lines
    for y in range(0, h, grid_spacing):
        cv2.line(overlay, (0, y), (w, y), color, thickness)

    # Blend
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return result


def detect_meter_edges(image: np.ndarray) -> list:
    """
    Detect potential meter box edges using contour analysis.

    Returns:
        List of (angle, length) tuples for detected edges
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for large rectangular contours (potential meter box)
    edge_angles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:  # Skip small contours
            continue

        # Approximate contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Check if rectangular (4 corners)
        if len(approx) == 4:
            # Get edges
            for i in range(4):
                pt1 = approx[i][0]
                pt2 = approx[(i+1) % 4][0]

                # Calculate angle
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                angle = math.degrees(math.atan2(dy, dx)) % 180

                # Edge length
                length = math.sqrt(dx**2 + dy**2)

                if length > 50:  # Only long edges
                    edge_angles.append((angle, length))

    return edge_angles


def calculate_edge_deviation(image: np.ndarray) -> dict:
    """
    Calculate deviation of detected edges from horizontal/vertical.

    Returns:
        Dictionary with deviation metrics
    """
    edges = detect_meter_edges(image)

    if not edges:
        return {
            'num_edges': 0,
            'horizontal_deviation': 0.0,
            'vertical_deviation': 0.0,
            'avg_deviation': 0.0
        }

    # Classify edges as horizontal (near 0° or 180°) or vertical (near 90°)
    horizontal_deviations = []
    vertical_deviations = []

    for angle, length in edges:
        # Use symmetry: angle and 180-angle are the same line
        angle_normalized = angle % 180
        if angle_normalized > 90:
            angle_normalized = 180 - angle_normalized

        if angle_normalized < 45:
            # Horizontal edge
            horizontal_deviations.append(angle_normalized)
        else:
            # Vertical edge
            vertical_deviations.append(abs(angle_normalized - 90))

    # Calculate weighted average (by length)
    horizontal_dev = np.average(horizontal_deviations) if horizontal_deviations else 0
    vertical_dev = np.average(vertical_deviations) if vertical_deviations else 0

    # Overall average
    all_deviations = horizontal_deviations + vertical_deviations
    avg_deviation = np.average(all_deviations) if all_deviations else 0

    return {
        'num_edges': len(edges),
        'horizontal_deviation': horizontal_dev,
        'vertical_deviation': vertical_dev,
        'avg_deviation': avg_deviation,
        'horizontal_count': len(horizontal_deviations),
        'vertical_count': len(vertical_deviations)
    }


def visualize_rotation_quality(
    original: np.ndarray,
    aligned: np.ndarray,
    detected_angle: float,
    correction_angle: float,
    deviation_before: dict,
    deviation_after: dict,
    name: str,
    save_path: Path
):
    """
    Create visualization showing rotation quality.
    """
    # Resize for display
    target_h = 400

    orig_resized = cv2.resize(original, (int(original.shape[1] * target_h / original.shape[0]), target_h))
    aligned_resized = cv2.resize(aligned, (int(aligned.shape[1] * target_h / aligned.shape[0]), target_h))

    # Add grid overlay
    orig_grid = draw_grid_overlay(orig_resized, grid_spacing=50, color=(0, 255, 255), thickness=1)
    aligned_grid = draw_grid_overlay(aligned_resized, grid_spacing=50, color=(0, 255, 0), thickness=1)

    # Stack horizontally
    comparison = np.hstack([orig_grid, aligned_grid])

    # Add labels
    cv2.putText(comparison, "BEFORE - Yellow Grid", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(comparison, "AFTER - Green Grid", (orig_resized.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Add angle info
    angle_text = f"Detected: {detected_angle:.1f} | Correction: {correction_angle:.1f}"
    cv2.putText(comparison, angle_text, (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add deviation info
    y_offset = 110
    cv2.putText(comparison, "Edge Analysis:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    y_offset += 30
    cv2.putText(comparison, f"Before: {deviation_before['num_edges']} edges, Dev={deviation_before['avg_deviation']:.1f}",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    y_offset += 30
    cv2.putText(comparison, f"After: {deviation_after['num_edges']} edges, Dev={deviation_after['avg_deviation']:.1f}",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Quality indicator
    y_offset += 40
    if deviation_after['avg_deviation'] < 3:
        status = "EXCELLENT"
        color = (0, 255, 0)
    elif deviation_after['avg_deviation'] < 7:
        status = "GOOD"
        color = (0, 255, 255)
    elif deviation_after['avg_deviation'] < 15:
        status = "ACCEPTABLE"
        color = (0, 165, 255)
    else:
        status = "NEEDS IMPROVEMENT"
        color = (0, 0, 255)

    cv2.putText(comparison, f"Quality: {status}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imwrite(str(save_path), comparison)


def analyze_rotation_quality(
    m2_model_path: str = "model/orientation.pth",
    num_samples: int = 10,
    output_dir: str = "rotation_quality_results"
):
    """
    Analyze rotation quality on test images.

    Args:
        m2_model_path: Path to M2 model
        num_samples: Number of samples to analyze
        output_dir: Output directory for results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load M2 model
    print("Loading M2 model...")
    m2_inference = M2Inference(m2_model_path, device='cpu')
    print(">> Model loaded\n")

    # Get test images
    test_dir = Path("data/m2_crops")
    if not test_dir.exists():
        test_dir = Path("data/images_4digit")

    test_images = list(test_dir.glob("*.jpg"))[:num_samples]

    if not test_images:
        print(f"No images found in {test_dir}")
        return

    print(f"Analyzing {len(test_images)} images...")
    print("=" * 80)

    results = []

    for i, img_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        print(f"  Size: {w}x{h}")

        # Calculate deviation BEFORE rotation
        deviation_before = calculate_edge_deviation(img)

        # Apply M2 rotation
        m2_result = m2_inference.align_with_info(img)
        aligned = m2_result['aligned_image']
        detected_angle = m2_result['detected_angle']
        correction_angle = m2_result['correction_angle']

        print(f"  Detected angle: {detected_angle:.2f}°")
        print(f"  Correction: {correction_angle:.2f}°")

        # Calculate deviation AFTER rotation
        deviation_after = calculate_edge_deviation(aligned)

        print(f"\n  BEFORE: {deviation_before['num_edges']} edges detected")
        print(f"    Horizontal deviation: {deviation_before['horizontal_deviation']:.2f}°")
        print(f"    Vertical deviation: {deviation_before['vertical_deviation']:.2f}°")
        print(f"    Average: {deviation_before['avg_deviation']:.2f}°")

        print(f"\n  AFTER: {deviation_after['num_edges']} edges detected")
        print(f"    Horizontal deviation: {deviation_after['horizontal_deviation']:.2f}°")
        print(f"    Vertical deviation: {deviation_after['vertical_deviation']:.2f}°")
        print(f"    Average: {deviation_after['avg_deviation']:.2f}°")

        # Store results
        results.append({
            'image': img_path.name,
            'detected_angle': detected_angle,
            'correction_angle': correction_angle,
            'deviation_before': deviation_before,
            'deviation_after': deviation_after
        })

        # Visualize
        vis_path = output_path / f"{img_path.stem}_quality.jpg"
        visualize_rotation_quality(
            img, aligned, detected_angle, correction_angle,
            deviation_before, deviation_after,
            img_path.name, vis_path
        )

        print(f"  >> Saved: {vis_path.name}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        deviations_before = [r['deviation_before']['avg_deviation'] for r in results if r['deviation_before']['num_edges'] > 0]
        deviations_after = [r['deviation_after']['avg_deviation'] for r in results if r['deviation_after']['num_edges'] > 0]

        if deviations_before and deviations_after:
            print(f"\nAverage Edge Deviation:")
            print(f"  BEFORE: {np.mean(deviations_before):.2f}° ± {np.std(deviations_before):.2f}°")
            print(f"  AFTER:  {np.mean(deviations_after):.2f}° ± {np.std(deviations_after):.2f}°")
            print(f"  Improvement: {np.mean(deviations_before) - np.mean(deviations_after):.2f}°")

        # Quality distribution
        excellent = sum(1 for r in results if r['deviation_after']['avg_deviation'] < 3)
        good = sum(1 for r in results if 3 <= r['deviation_after']['avg_deviation'] < 7)
        acceptable = sum(1 for r in results if 7 <= r['deviation_after']['avg_deviation'] < 15)
        poor = sum(1 for r in results if r['deviation_after']['avg_deviation'] >= 15)

        print(f"\nQuality Distribution:")
        print(f"  Excellent (< 3°): {excellent}/{len(results)}")
        print(f"  Good (3-7°): {good}/{len(results)}")
        print(f"  Acceptable (7-15°): {acceptable}/{len(results)}")
        print(f"  Poor (> 15°): {poor}/{len(results)}")

        # Worst cases
        valid_results = [r for r in results if r['deviation_after']['num_edges'] > 0]
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda r: r['deviation_after']['avg_deviation'], reverse=True)

            print(f"\nTop 3 Worst Cases:")
            for i, r in enumerate(sorted_results[:3]):
                print(f"  {i+1}. {r['image']}")
                print(f"     Deviation after: {r['deviation_after']['avg_deviation']:.1f}°")

    print(f"\n>> Results saved to: {output_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize rotation quality")
    parser.add_argument("--m2-model", type=str, default="model/orientation.pth",
                       help="Path to M2 model")
    parser.add_argument("--samples", type=int, default=10,
                       help="Number of samples to analyze")
    parser.add_argument("--output-dir", type=str, default="rotation_quality_results",
                       help="Output directory for results")

    args = parser.parse_args()

    print("=" * 80)
    print("M2 Rotation Quality Analysis")
    print("=" * 80)

    analyze_rotation_quality(
        m2_model_path=args.m2_model,
        num_samples=args.samples,
        output_dir=args.output_dir
    )

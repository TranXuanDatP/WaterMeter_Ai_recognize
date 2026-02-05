"""
Analyze Rotation Error in M2

This script analyzes the rotation accuracy of M2:
1. Detects angle on test images
2. Measures "straightness" of edges after alignment
3. Calculates deviation from horizontal/vertical
4. Visualizes before/after with edge detection
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import math

# Add project path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.m2_orientation_alignment.inference import M2Inference


def calculate_edge_angles(image: np.ndarray, num_edges: int = 10):
    """
    Calculate angles of dominant edges in image using Canny edge detection.

    Returns:
        List of edge angles in degrees (0-180)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

    if lines is None:
        return []

    # Extract angles
    angles = []
    for line in lines[:num_edges]:
        rho, theta = line[0]
        angle_deg = math.degrees(theta)

        # Normalize to 0-180
        if angle_deg < 0:
            angle_deg += 180

        angles.append(angle_deg)

    return angles


def calculate_straightness_score(image: np.ndarray) -> dict:
    """
    Calculate how "straight" the image is by measuring edge alignment.

    Returns:
        Dictionary with straightness metrics
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=80)

    if lines is None or len(lines) < 3:
        return {
            'num_lines': 0,
            'horizontal_score': 0.0,
            'vertical_score': 0.0,
            'avg_deviation': 90.0,
            'max_deviation': 90.0
        }

    # Calculate angles
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = math.degrees(theta) % 180

        # For horizontal/vertical measurement, use symmetry
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        angles.append(angle_deg)

    # Calculate metrics
    # Horizontal: angles near 0 degrees
    # Vertical: angles near 90 degrees
    horizontal_angles = [a for a in angles if a < 45]
    vertical_angles = [a for a in angles if a >= 45]

    # Score: percentage of lines that are close to 0 or 90
    horizontal_score = len([a for a in horizontal_angles if a < 10]) / max(len(horizontal_angles), 1)
    vertical_score = len([a for a in vertical_angles if abs(a - 90) < 10]) / max(len(vertical_angles), 1)

    # Average deviation from perfect horizontal/vertical
    deviations = []
    for a in angles:
        if a < 45:
            deviations.append(a)  # Deviation from 0
        else:
            deviations.append(abs(a - 90))  # Deviation from 90

    avg_deviation = np.mean(deviations) if deviations else 90
    max_deviation = np.max(deviations) if deviations else 90

    return {
        'num_lines': len(angles),
        'horizontal_score': horizontal_score * 100,  # Percentage
        'vertical_score': vertical_score * 100,  # Percentage
        'avg_deviation': avg_deviation,  # Degrees
        'max_deviation': max_deviation  # Degrees
    }


def visualize_rotation_analysis(
    original: np.ndarray,
    aligned: np.ndarray,
    detected_angle: float,
    correction_angle: float,
    straightness_before: dict,
    straightness_after: dict,
    name: str,
    save_path: Path
):
    """
    Visualize rotation analysis with edge overlays.
    """
    # Edge detection for visualization
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    edges_orig = cv2.Canny(gray_orig, 50, 150)
    edges_aligned = cv2.Canny(gray_aligned, 50, 150)

    # Convert edges to BGR for overlay
    edges_orig_bgr = cv2.cvtColor(edges_orig, cv2.COLOR_GRAY2BGR)
    edges_aligned_bgr = cv2.cvtColor(edges_aligned, cv2.COLOR_GRAY2BGR)

    # Resize for display
    target_h = 400

    orig_resized = cv2.resize(original, (int(original.shape[1] * target_h / original.shape[0]), target_h))
    aligned_resized = cv2.resize(aligned, (int(aligned.shape[1] * target_h / aligned.shape[0]), target_h))
    edges_orig_resized = cv2.resize(edges_orig_bgr, (int(edges_orig_bgr.shape[1] * target_h / edges_orig_bgr.shape[0]), target_h))
    edges_aligned_resized = cv2.resize(edges_aligned_bgr, (int(edges_aligned_bgr.shape[1] * target_h / edges_aligned_bgr.shape[0]), target_h))

    # Create comparison: original | aligned | orig edges | aligned edges
    row1 = np.hstack([orig_resized, aligned_resized])
    row2 = np.hstack([edges_orig_resized, edges_aligned_resized])

    padding = np.zeros((20, row1.shape[1], 3), dtype=np.uint8)
    comparison = np.vstack([row1, padding, row2])

    # Add labels
    cv2.putText(comparison, "BEFORE ROTATION", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "AFTER ROTATION", (orig_resized.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Add angle info
    angle_text = f"Detected: {detected_angle:.1f} | Correction: {correction_angle:.1f}"
    cv2.putText(comparison, angle_text, (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Add straightness info
    before_text = f"Before: Dev={straightness_before['avg_deviation']:.1f}deg (Max: {straightness_before['max_deviation']:.1f})"
    after_text = f"After: Dev={straightness_after['avg_deviation']:.1f}deg (Max: {straightness_after['max_deviation']:.1f})"

    y_offset = target_h + 50
    cv2.putText(comparison, before_text, (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(comparison, after_text, (10, y_offset + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Color code: green if avg deviation < 5, yellow if < 10, red if >= 10
    color_after = (0, 255, 0) if straightness_after['avg_deviation'] < 5 else \
                  (0, 255, 255) if straightness_after['avg_deviation'] < 10 else (0, 0, 255)

    status_text = f"Status: {'EXCELLENT' if straightness_after['avg_deviation'] < 5 else 'GOOD' if straightness_after['avg_deviation'] < 10 else 'NEEDS IMPROVEMENT'}"
    cv2.putText(comparison, status_text, (10, y_offset + 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_after, 2)

    cv2.imwrite(str(save_path), comparison)


def analyze_rotation_error(
    m2_model_path: str = "model/orientation.pth",
    num_samples: int = 10,
    output_dir: str = "rotation_analysis_results"
):
    """
    Analyze rotation error on test images.

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

    # Get test images (use M1 crops for better analysis)
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
        print(f"  Original: {w}x{h}")

        # Calculate straightness BEFORE rotation
        straightness_before = calculate_straightness_score(img)

        # Apply M2 rotation
        m2_result = m2_inference.align_with_info(img)
        aligned = m2_result['aligned_image']
        detected_angle = m2_result['detected_angle']
        correction_angle = m2_result['correction_angle']

        print(f"  Detected angle: {detected_angle:.2f}°")
        print(f"  Correction: {correction_angle:.2f}°")
        print(f"  Aligned: {aligned.shape[1]}x{aligned.shape[0]}")

        # Calculate straightness AFTER rotation
        straightness_after = calculate_straightness_score(aligned)

        print(f"\n  BEFORE rotation:")
        print(f"    Lines detected: {straightness_before['num_lines']}")
        print(f"    Avg deviation: {straightness_before['avg_deviation']:.2f}°")
        print(f"    Max deviation: {straightness_before['max_deviation']:.2f}°")

        print(f"\n  AFTER rotation:")
        print(f"    Lines detected: {straightness_after['num_lines']}")
        print(f"    Avg deviation: {straightness_after['avg_deviation']:.2f}°")
        print(f"    Max deviation: {straightness_after['max_deviation']:.2f}°")

        # Store results
        results.append({
            'image': img_path.name,
            'detected_angle': detected_angle,
            'correction_angle': correction_angle,
            'straightness_before': straightness_before,
            'straightness_after': straightness_after
        })

        # Visualize
        vis_path = output_path / f"{img_path.stem}_analysis.jpg"
        visualize_rotation_analysis(
            img, aligned, detected_angle, correction_angle,
            straightness_before, straightness_after,
            img_path.name, vis_path
        )

        print(f"  >> Saved: {vis_path.name}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if results:
        # Calculate statistics
        deviations_before = [r['straightness_before']['avg_deviation'] for r in results]
        deviations_after = [r['straightness_after']['avg_deviation'] for r in results]

        print(f"\nAverage Deviation from Horizontal/Vertical:")
        print(f"  BEFORE rotation: {np.mean(deviations_before):.2f}° ± {np.std(deviations_before):.2f}°")
        print(f"  AFTER rotation:  {np.mean(deviations_after):.2f}° ± {np.std(deviations_after):.2f}°")

        # Count excellent/good/poor
        excellent = sum(1 for r in results if r['straightness_after']['avg_deviation'] < 5)
        good = sum(1 for r in results if 5 <= r['straightness_after']['avg_deviation'] < 10)
        poor = sum(1 for r in results if r['straightness_after']['avg_deviation'] >= 10)

        print(f"\nQuality Distribution:")
        print(f"  Excellent (< 5°): {excellent}/{len(results)} ({excellent/len(results)*100:.1f}%)")
        print(f"  Good (5-10°): {good}/{len(results)} ({good/len(results)*100:.1f}%)")
        print(f"  Poor (> 10°): {poor}/{len(results)} ({poor/len(results)*100:.1f}%)")

        # Show worst cases
        sorted_results = sorted(results, key=lambda r: r['straightness_after']['avg_deviation'], reverse=True)

        print(f"\nTop 3 Worst Cases:")
        for i, r in enumerate(sorted_results[:3]):
            print(f"  {i+1}. {r['image']}")
            print(f"     Detected: {r['detected_angle']:.1f}°, Deviation after: {r['straightness_after']['avg_deviation']:.1f}°")

    print(f"\n>> All results saved to: {output_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze rotation error in M2")
    parser.add_argument("--m2-model", type=str, default="model/orientation.pth",
                       help="Path to M2 model")
    parser.add_argument("--samples", type=int, default=10,
                       help="Number of samples to analyze")
    parser.add_argument("--output-dir", type=str, default="rotation_analysis_results",
                       help="Output directory for results")

    args = parser.parse_args()

    print("=" * 80)
    print("M2 Rotation Error Analysis")
    print("=" * 80)

    analyze_rotation_error(
        m2_model_path=args.m2_model,
        num_samples=args.samples,
        output_dir=args.output_dir
    )

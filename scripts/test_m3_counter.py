"""
Test M3 Counter ROI Detection - Visualize Results

This script tests M3 counter detection on sample images and visualizes:
- Original upright image (from M2)
- Detected bounding box
- Extracted counter ROI
- Side-by-side comparison
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.m3_counter_roi import M3CounterExtractor


def test_m3_on_samples(
    num_samples: int = 5,
    output_dir: str = "test_m3_results"
):
    """
    Test M3 counter detection on sample images.

    Args:
        num_samples: Number of samples to test
        output_dir: Directory to save results
    """
    # Initialize M3
    print("Loading M3 Counter Extractor...")
    extractor = M3CounterExtractor(detection_method="auto")
    print(">> M3 loaded\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get test images (use upright images from M2)
    test_dir = Path("data/m2_upright_gt/val/images")
    if not test_dir.exists():
        test_dir = Path("data/m2_crops")

    test_images = list(test_dir.glob("*.jpg"))[:num_samples]

    if not test_images:
        print(f"No images found in {test_dir}")
        return

    print(f"Testing on {len(test_images)} images from {test_dir}")
    print("="*60)

    results = []

    for i, img_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] Processing: {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  XX Failed to load")
            continue

        print(f"  Original size: {img.shape[1]}x{img.shape[0]}")

        # Extract counter with M3
        roi, bbox, confidence, method = extractor.extract_counter(
            img,
            return_bbox=True,
            return_confidence=True,
        )

        x, y, w, h = bbox

        print(f"  Detected by: {method}")
        print(f"  BBox: x={x}, y={y}, w={w}, h={h}")
        print(f"  ROI size: {roi.shape[1]}x{roi.shape[0]}")
        print(f"  Confidence: {confidence:.2f}")

        # Save results
        base_name = img_path.stem

        # Original
        orig_path = output_path / f"{base_name}_1_original.jpg"
        cv2.imwrite(str(orig_path), img)

        # ROI
        roi_path = output_path / f"{base_name}_2_roi.jpg"
        cv2.imwrite(str(roi_path), roi)

        # Create visualization with bounding box
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Add labels
        label = f"Counter ({method}, conf={confidence:.2f})"
        cv2.putText(vis, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        vis_path = output_path / f"{base_name}_3_detection.jpg"
        cv2.imwrite(str(vis_path), vis)

        # Create side-by-side comparison
        # Resize both to same height for comparison
        target_h = 400

        # Original with bbox (resized)
        vis_resized = cv2.resize(vis, (int(vis.shape[1] * target_h / vis.shape[0]), target_h))

        # ROI (resized)
        roi_resized = cv2.resize(roi, (int(roi.shape[1] * target_h / roi.shape[0]), target_h))

        # Concatenate
        comparison = np.hstack([vis_resized, roi_resized])

        # Add labels
        cv2.putText(comparison, "ORIGINAL + BBOX", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"BBox: ({x},{y}) {w}x{h}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, f"Conf: {confidence:.2f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(comparison, "EXTRACTED ROI", (vis_resized.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"Size: {roi.shape[1]}x{roi.shape[0]}",
                   (vis_resized.shape[1] + 10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        comp_path = output_path / f"{base_name}_4_comparison.jpg"
        cv2.imwrite(str(comp_path), comparison)

        print(f"  >> Saved to: {output_path}")
        print(f"     - {base_name}_1_original.jpg")
        print(f"     - {base_name}_2_roi.jpg")
        print(f"     - {base_name}_3_detection.jpg")
        print(f"     - {base_name}_4_comparison.jpg")

        results.append({
            'image': img_path.name,
            'method': method,
            'bbox': bbox,
            'roi_size': (roi.shape[1], roi.shape[0]),
            'confidence': confidence,
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total processed: {len(results)}")

    if results:
        # Method distribution
        methods = [r['method'] for r in results]
        method_counts = {m: methods.count(m) for m in set(methods)}
        print(f"\nMethod distribution:")
        for method, count in method_counts.items():
            print(f"  - {method}: {count}/{len(results)}")

        # Confidence stats
        confidences = [r['confidence'] for r in results]
        avg_conf = np.mean(confidences)
        min_conf = np.min(confidences)
        max_conf = np.max(confidences)

        print(f"\nConfidence statistics:")
        print(f"  Average: {avg_conf:.2f}")
        print(f"  Min: {min_conf:.2f}")
        print(f"  Max: {max_conf:.2f}")

        # ROI size stats
        roi_widths = [r['roi_size'][0] for r in results]
        roi_heights = [r['roi_size'][1] for r in results]

        print(f"\nROI size statistics:")
        print(f"  Width: {np.mean(roi_widths):.0f} avg, {np.min(roi_widths)}-{np.max(roi_widths)} range")
        print(f"  Height: {np.mean(roi_heights):.0f} avg, {np.min(roi_heights)}-{np.max(roi_heights)} range")

        # Count high confidence detections
        high_conf = sum(1 for c in confidences if c > 0.8)
        print(f"\nHigh confidence detections (>0.8): {high_conf}/{len(results)}")

    print(f"\n>> All results saved to: {output_path.absolute()}")
    print("\nYou can view the comparison images to verify detection quality.")


def test_m3_interactive(image_path: str = None):
    """
    Interactive test with matplotlib visualization.

    Args:
        image_path: Path to single test image
    """
    import matplotlib.pyplot as plt

    # Initialize M3
    print("Loading M3 Counter Extractor...")
    extractor = M3CounterExtractor(detection_method="auto")
    print(">> M3 loaded\n")

    # Get image
    if image_path is None:
        # Use first image from val set
        test_dir = Path("data/m2_upright_gt/val/images")
        if not test_dir.exists():
            test_dir = Path("data/m2_crops")

        images = list(test_dir.glob("*.jpg"))
        if not images:
            print("No test images found")
            return

        image_path = str(images[0])

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return

    print(f"Testing: {Path(image_path).name}")
    print(f"Original size: {img.shape[1]}x{img.shape[0]}")

    # Extract counter
    roi, bbox, confidence, method = extractor.extract_counter(
        img,
        return_bbox=True,
        return_confidence=True,
    )

    x, y, w, h = bbox

    print(f"Method: {method}")
    print(f"BBox: ({x}, {y}, {w}, {h})")
    print(f"ROI size: {roi.shape[1]}x{roi.shape[0]}")
    print(f"Confidence: {confidence:.2f}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original\n{img.shape[1]}x{img.shape[0]}")
    axes[0, 0].axis('off')

    # Detection with bbox
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(vis, f"Counter ({method})", (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    axes[0, 1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Detection ({method})\nConfidence: {confidence:.2f}")
    axes[0, 1].axis('off')

    # Extracted ROI
    axes[1, 0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Extracted ROI\n{roi.shape[1]}x{roi.shape[0]}")
    axes[1, 0].axis('off')

    # ROI enhanced (for OCR preview)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    roi_thresh = cv2.adaptiveThreshold(
        roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    axes[1, 1].imshow(roi_thresh, cmap='gray')
    axes[1, 1].set_title("ROI Preprocessed (Thresh)\nFor OCR Preview")
    axes[1, 1].axis('off')

    plt.suptitle(f"M3 Counter Detection Test - {Path(image_path).name}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\n>> Displayed visualization")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test M3 counter detection visualization")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of samples to test")
    parser.add_argument("--output-dir", type=str, default="test_m3_results",
                       help="Output directory for results")
    parser.add_argument("--interactive", action="store_true",
                       help="Show interactive matplotlib visualization")
    parser.add_argument("--image", type=str,
                       help="Single image to test (for interactive mode)")

    args = parser.parse_args()

    if args.interactive:
        test_m3_interactive(args.image)
    else:
        test_m3_on_samples(
            num_samples=args.samples,
            output_dir=args.output_dir
        )

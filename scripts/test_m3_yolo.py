"""
Test M3 Counter Detection (YOLOv8) - Visualize Results

This script tests M3 counter detection using YOLOv8 model and visualizes:
- Original upright image (from M2)
- Detected counter bounding box
- Extracted counter ROI
- Side-by-side comparison
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def test_m3_on_samples(
    model_path: str = "model/detect_array.pt",
    num_samples: int = 5,
    output_dir: str = "test_m3_results"
):
    """
    Test M3 counter detection on sample images.

    Args:
        model_path: Path to M3 YOLOv8 model
        num_samples: Number of samples to test
        output_dir: Directory to save results
    """
    # Initialize M3
    print("Loading M3 YOLOv8 model...")
    model = YOLO(model_path)
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
    print("=" * 60)

    results = []

    for i, img_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] Processing: {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  XX Failed to load")
            continue

        print(f"  Original size: {img.shape[1]}x{img.shape[0]}")

        # Detect counter with M3
        detection_results = model(img, verbose=False)

        if len(detection_results) == 0 or len(detection_results[0].boxes) == 0:
            print(f"  XX No counter detected")
            continue

        # Get counter detection
        boxes = detection_results[0].boxes
        best_idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        conf = float(boxes.conf[best_idx])

        # Extract counter ROI
        roi = img[y1:y2, x1:x2]

        print(f"  Confidence: {conf:.2f}")
        print(f"  BBox: ({x1}, {y1}, {x2}, {y2})")
        print(f"  ROI size: {roi.shape[1]}x{roi.shape[0]}")

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
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add labels
        label = f"Counter (conf={conf:.2f})"
        cv2.putText(vis, label, (x1, y1 - 10),
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
        cv2.putText(comparison, f"BBox: ({x1},{y1}) {x2-x1}x{y2-y1}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, f"Conf: {conf:.2f}", (10, 110),
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
            'bbox': (x1, y1, x2-x1, y2-y1),
            'roi_size': (roi.shape[1], roi.shape[0]),
            'confidence': conf,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed: {len(results)}")

    if results:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test M3 counter detection (YOLOv8)")
    parser.add_argument("--model", type=str, default="model/detect_array.pt",
                       help="Path to M3 model")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of samples to test")
    parser.add_argument("--output-dir", type=str, default="test_m3_results",
                       help="Output directory for results")

    args = parser.parse_args()

    test_m3_on_samples(
        model_path=args.model,
        num_samples=args.samples,
        output_dir=args.output_dir
    )

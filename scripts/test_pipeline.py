"""
End-to-End Test: M1 → M2 → M3 Pipeline

This script tests the complete pipeline:
1. M1: Detect and crop watermeter from original image
2. M2: Align/rotate to upright orientation
3. M3: Detect counter region from aligned image
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch


def test_pipeline(
    original_images_dir: str,
    m1_model_path: str = "model/detect_watermeter.pt",
    m2_model_path: str = "model/orientation.pth",
    m3_model_path: str = "model/detect_array.pt",
    num_samples: int = 3,
    output_dir: str = "test_pipeline_results"
):
    """
    Test complete M1 → M2 → M3 pipeline.

    Args:
        original_images_dir: Path to original watermeter images
        m1_model_path: Path to M1 watermeter detection model
        m2_model_path: Path to M2 orientation model
        m3_model_path: Path to M3 counter detection model
        num_samples: Number of samples to test
        output_dir: Output directory for results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load models
    print("Loading models...")
    print(f"  M1: {m1_model_path}")
    m1_model = YOLO(m1_model_path)

    print(f"  M2: {m2_model_path}")
    # Load M2 model using M2Inference class
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.m2_orientation_alignment.inference import M2Inference

    m2_inference = M2Inference(m2_model_path, device='cpu')
    m2_model = m2_inference.model

    print(f"  M3: {m3_model_path}")
    m3_model = YOLO(m3_model_path)
    print(">> All models loaded\n")

    # Get test images
    test_dir = Path(original_images_dir)
    test_images = list(test_dir.glob("*.jpg"))[:num_samples]

    if not test_images:
        print(f"No images found in {test_dir}")
        return

    print(f"Testing {len(test_images)} images through M1 → M2 → M3 pipeline")
    print("=" * 60)

    for i, img_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] Processing: {img_path.name}")

        # Load original image
        original = cv2.imread(str(img_path))
        if original is None:
            print(f"  XX Failed to load")
            continue

        h_orig, w_orig = original.shape[:2]
        print(f"  Original: {w_orig}x{h_orig}")

        # ========== M1: Watermeter Detection ==========
        print("\n  [M1] Watermeter Detection...")
        m1_results = m1_model(original, verbose=False)

        if len(m1_results) == 0 or len(m1_results[0].boxes) == 0:
            print(f"    XX No watermeter detected")
            continue

        # Get best detection
        boxes = m1_results[0].boxes
        best_idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        conf_m1 = float(boxes.conf[best_idx])

        # Crop watermeter
        meter_crop = original[y1:y2, x1:x2]

        print(f"    BBox: ({x1}, {y1}, {x2}, {y2}), Conf: {conf_m1:.2f}")
        print(f"    Crop: {meter_crop.shape[1]}x{meter_crop.shape[0]}")

        # Save M1 result
        m1_vis = original.copy()
        cv2.rectangle(m1_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(m1_vis, f"M1: Watermeter (conf={conf_m1:.2f})", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ========== M2: Orientation Alignment ==========
        print("\n  [M2] Orientation Alignment...")

        # Use M2Inference to align
        m2_result = m2_inference.align_with_info(meter_crop)
        meter_aligned = m2_result['aligned_image']
        detected_angle = m2_result['detected_angle']
        correction_angle = m2_result['correction_angle']
        print(f"    Aligned: {meter_aligned.shape[1]}x{meter_aligned.shape[0]}")

        # ========== M3: Counter Detection ==========
        print("\n  [M3] Counter Detection...")
        m3_results = m3_model(meter_aligned, verbose=False)

        if len(m3_results) == 0 or len(m3_results[0].boxes) == 0:
            print(f"    XX No counter detected")
            # Still save intermediate results
            save_pipeline_results(
                original, m1_vis, meter_crop, meter_aligned, None,
                img_path.name, output_path, i
            )
            continue

        # Get counter detection
        boxes = m3_results[0].boxes
        best_idx = boxes.conf.argmax()
        cx1, cy1, cx2, cy2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        conf_m3 = float(boxes.conf[best_idx])

        # Extract counter ROI
        counter_roi = meter_aligned[cy1:cy2, cx1:cx2]

        print(f"    BBox: ({cx1}, {cy1}, {cx2}, {cy2}), Conf: {conf_m3:.2f}")
        print(f"    Counter ROI: {counter_roi.shape[1]}x{counter_roi.shape[0]}")

        # Save results
        save_pipeline_results(
            original, m1_vis, meter_crop, meter_aligned,
            (cx1, cy1, cx2, cy2, conf_m3, counter_roi),
            img_path.name, output_path, i
        )

        print(f"\n  >> Pipeline successful!")
        print(f"     Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print(f"Pipeline test complete!")
    print(f"Results saved to: {output_path.absolute()}")


def save_pipeline_results(
    original: np.ndarray,
    m1_vis: np.ndarray,
    meter_crop: np.ndarray,
    meter_aligned: np.ndarray,
    counter_result: tuple,
    name: str,
    output_path: Path,
    idx: int
):
    """
    Save pipeline results as visualizations.

    Args:
        original: Original image
        m1_vis: Visualization with M1 detection
        meter_crop: M1 cropped meter
        meter_aligned: M2 aligned meter
        counter_result: (cx1, cy1, cx2, cy2, conf, roi) or None
        name: Image name
        output_path: Output directory
        idx: Sample index
    """
    base_name = f"{idx+1:02d}_{name}"

    # Save individual steps
    cv2.imwrite(str(output_path / f"{base_name}_0_original.jpg"), original)
    cv2.imwrite(str(output_path / f"{base_name}_1_m1_detection.jpg"), m1_vis)
    cv2.imwrite(str(output_path / f"{base_name}_2_m1_crop.jpg"), meter_crop)
    cv2.imwrite(str(output_path / f"{base_name}_3_m2_aligned.jpg"), meter_aligned)

    # Create M3 visualization
    if counter_result is not None:
        cx1, cy1, cx2, cy2, conf_m3, counter_roi = counter_result
        m3_vis = meter_aligned.copy()
        cv2.rectangle(m3_vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
        cv2.putText(m3_vis, f"M3: Counter (conf={conf_m3:.2f})", (cx1, cy1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(str(output_path / f"{base_name}_4_m3_detection.jpg"), m3_vis)
        cv2.imwrite(str(output_path / f"{base_name}_5_m3_roi.jpg"), counter_roi)

        # Create pipeline summary
        create_pipeline_summary(
            original, m1_vis, meter_crop, meter_aligned, m3_vis, counter_roi,
            output_path / f"{base_name}_pipeline_summary.jpg"
        )
    else:
        # M3 failed - show what we have
        create_pipeline_summary_no_m3(
            original, m1_vis, meter_crop, meter_aligned,
            output_path / f"{base_name}_pipeline_summary.jpg"
        )


def create_pipeline_summary(
    original: np.ndarray,
    m1_vis: np.ndarray,
    meter_crop: np.ndarray,
    meter_aligned: np.ndarray,
    m3_vis: np.ndarray,
    counter_roi: np.ndarray,
    save_path: Path
):
    """Create a summary visualization of the complete pipeline."""
    # Resize all to same dimensions (width x height)
    target_w, target_h = 400, 300

    orig_resized = cv2.resize(original, (target_w, target_h))
    m1_resized = cv2.resize(m1_vis, (target_w, target_h))
    crop_resized = cv2.resize(meter_crop, (target_w, target_h))
    aligned_resized = cv2.resize(meter_aligned, (target_w, target_h))
    m3_resized = cv2.resize(m3_vis, (target_w, target_h))
    roi_resized = cv2.resize(counter_roi, (target_w, target_h))

    # Create 2x3 grid
    row1 = np.hstack([orig_resized, m1_resized, crop_resized])
    row2 = np.hstack([aligned_resized, m3_resized, roi_resized])

    # Add padding between rows
    padding = np.zeros((20, row1.shape[1], 3), dtype=np.uint8)
    summary = np.vstack([row1, padding, row2])

    # Add labels
    cv2.putText(summary, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(summary, "M1: DETECTION", (target_w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(summary, "M1: CROP", (target_w * 2 + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(summary, "M2: ALIGNED", (10, target_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(summary, "M3: DETECTION", (target_w + 10, target_h + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(summary, "M3: COUNTER ROI", (target_w * 2 + 10, target_h + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(str(save_path), summary)


def create_pipeline_summary_no_m3(
    original: np.ndarray,
    m1_vis: np.ndarray,
    meter_crop: np.ndarray,
    meter_aligned: np.ndarray,
    save_path: Path
):
    """Create summary when M3 fails."""
    target_h = 300

    orig_resized = cv2.resize(original, (int(original.shape[1] * target_h / original.shape[0]), target_h))
    m1_resized = cv2.resize(m1_vis, (int(m1_vis.shape[1] * target_h / m1_vis.shape[0]), target_h))
    crop_resized = cv2.resize(meter_crop, (int(meter_crop.shape[1] * target_h / meter_crop.shape[0]), target_h))
    aligned_resized = cv2.resize(meter_aligned, (int(meter_aligned.shape[1] * target_h / meter_aligned.shape[0]), target_h))

    row1 = np.hstack([orig_resized, m1_resized])
    row2 = np.hstack([crop_resized, aligned_resized])

    padding = np.zeros((20, row1.shape[1], 3), dtype=np.uint8)
    summary = np.vstack([row1, padding, row2])

    cv2.putText(summary, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(summary, "M1: DETECTION", (orig_resized.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(summary, "M1: CROP", (10, target_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(summary, "M2: ALIGNED (M3 FAILED)", (crop_resized.shape[1] + 10, target_h + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(str(save_path), summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test M1 → M2 → M3 pipeline")
    parser.add_argument("--input", type=str, default="data/images_4digit",
                       help="Path to original watermeter images")
    parser.add_argument("--m1-model", type=str, default="model/detect_watermeter.pt",
                       help="Path to M1 model")
    parser.add_argument("--m2-model", type=str, default="model/orientation.pth",
                       help="Path to M2 model")
    parser.add_argument("--m3-model", type=str, default="model/detect_array.pt",
                       help="Path to M3 model")
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of samples to test")
    parser.add_argument("--output-dir", type=str, default="test_pipeline_results",
                       help="Output directory for results")

    args = parser.parse_args()

    print("=" * 60)
    print("M1 → M2 → M3 Pipeline Test")
    print("=" * 60)

    test_pipeline(
        original_images_dir=args.input,
        m1_model_path=args.m1_model,
        m2_model_path=args.m2_model,
        m3_model_path=args.m3_model,
        num_samples=args.samples,
        output_dir=args.output_dir
    )

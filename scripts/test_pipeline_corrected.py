"""
End-to-End Test: M1 → M2 → M3 Pipeline with CORRECTED M2 Logic

This script tests the complete pipeline with corrected M2 smart rotation:
1. M1: Detect and crop watermeter from original image
2. M2: Align/rotate using CORRECTED smart rotate logic
3. M3: Detect counter region from aligned image
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from PIL import Image
from torchvision import transforms

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.m2_orientation.model import M2_OrientationModel


class M2_CorrectedRotator:
    """M2 Orientation with CORRECTED smart rotate logic"""

    def __init__(self, model_path: str, device: torch.device):
        print("[M2] Loading orientation model with CORRECTED logic...")
        self.device = device
        self.model = M2_OrientationModel().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        print(f"      Model: {model_path}")
        print(f"      Device: {device}")
        print(f"      Epoch: {checkpoint.get('epoch', 'N/A')}")

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def predict_angle(self, image: np.ndarray) -> float:
        """Predict rotation angle"""
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Transform
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            sin_cos = self.model(tensor)

        # Convert to angle
        sin_val = sin_cos[0, 0].cpu().item()
        cos_val = sin_cos[0, 1].cpu().item()
        angle_rad = np.arctan2(sin_val, cos_val)
        angle_deg = np.degrees(angle_rad)
        angle_deg = (angle_deg + 360) % 360

        return angle_deg

    def smart_rotate(self, image: np.ndarray, angle: float) -> tuple:
        """
        Smart rotate image using CORRECTED logic

        Logic (matching original metadata pattern):
        1. Normalize angle to [-180, 180]
        2. Correction angle = -angle_norm (rotate counter-clockwise)
        3. Rotate by correction angle

        Args:
            image: Input image
            angle: Detected angle in [0, 360]

        Returns:
            (rotated_image, correction_angle_used)
        """
        # Normalize angle to [-180, 180]
        angle_norm = angle if angle <= 180 else angle - 180

        # Correction angle: rotate counter-clockwise to bring to 0
        correction_angle = -angle_norm

        # Rotate
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)

        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        rotation_matrix[0, 2] += (new_width - width) // 2
        rotation_matrix[1, 2] += (new_height - height) // 2

        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        return rotated, correction_angle

    def align_with_info(self, image: np.ndarray) -> dict:
        """
        Align image and return detailed info

        Returns:
            dict with:
                - aligned_image: rotated image
                - detected_angle: original detected angle [0, 360]
                - correction_angle: angle used for rotation [-180, 180]
        """
        detected_angle = self.predict_angle(image)
        aligned_image, correction_angle = self.smart_rotate(image, detected_angle)

        return {
            'aligned_image': aligned_image,
            'detected_angle': detected_angle,
            'correction_angle': correction_angle
        }


def test_pipeline(
    original_images_dir: str,
    m1_model_path: str,
    m2_model_path: str,
    m3_model_path: str,
    num_samples: int = 3,
    output_dir: str = "test_pipeline_corrected"
):
    """
    Test complete M1 → M2 → M3 pipeline with CORRECTED M2 logic.

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

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    print("=" * 80)
    print("M1 → M2 → M3 Pipeline Test with CORRECTED M2 Logic")
    print("=" * 80)
    print("\nLoading models...")
    print(f"  M1: {m1_model_path}")
    m1_model = YOLO(m1_model_path)

    print(f"  M2: {m2_model_path}")
    m2_rotator = M2_CorrectedRotator(m2_model_path, device)

    print(f"  M3: {m3_model_path}")
    m3_model = YOLO(m3_model_path)
    print(">> All models loaded\n")

    # Get test images
    test_dir = Path(original_images_dir)
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

    if num_samples > 0:
        test_images = test_images[:num_samples]

    if not test_images:
        print(f"No images found in {test_dir}")
        return

    print(f"Testing {len(test_images)} images through M1 → M2 → M3 pipeline")
    print("=" * 80)

    results = []

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
        cv2.putText(m1_vis, f"M1: {conf_m1:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ========== M2: Orientation Alignment (CORRECTED) ==========
        print("\n  [M2] Orientation Alignment (CORRECTED LOGIC)...")

        # Use M2_CorrectedRotator to align
        m2_result = m2_rotator.align_with_info(meter_crop)
        meter_aligned = m2_result['aligned_image']
        detected_angle = m2_result['detected_angle']
        correction_angle = m2_result['correction_angle']

        print(f"    Detected Angle: {detected_angle:.2f}°")
        print(f"    Correction Angle: {correction_angle:.2f}°")
        print(f"    Aligned: {meter_aligned.shape[1]}x{meter_aligned.shape[0]}")

        # Create M2 visualization
        m2_vis = meter_crop.copy()
        cv2.putText(m2_vis, f"Detected: {detected_angle:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(m2_vis, f"Correction: {correction_angle:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        # ========== M3: Counter Detection ==========
        print("\n  [M3] Counter Detection...")
        m3_results = m3_model(meter_aligned, verbose=False)

        if len(m3_results) == 0 or len(m3_results[0].boxes) == 0:
            print(f"    XX No counter detected")
            # Still save intermediate results
            save_pipeline_results(
                original, m1_vis, m2_vis, meter_crop, meter_aligned, None,
                img_path.name, output_path, i
            )
            results.append({
                'filename': img_path.name,
                'm1_conf': conf_m1,
                'm2_detected': detected_angle,
                'm2_correction': correction_angle,
                'm3_detected': False,
                'm3_conf': 0.0
            })
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
            original, m1_vis, m2_vis, meter_crop, meter_aligned,
            (cx1, cy1, cx2, cy2, conf_m3, counter_roi),
            img_path.name, output_path, i
        )

        results.append({
            'filename': img_path.name,
            'm1_conf': conf_m1,
            'm2_detected': detected_angle,
            'm2_correction': correction_angle,
            'm3_detected': True,
            'm3_conf': conf_m3
        })

        print(f"\n  >> Pipeline successful!")

    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    m1_success = sum(1 for r in results if r['m1_conf'] > 0)
    m3_success = sum(1 for r in results if r['m3_detected'])

    print(f"\nTotal images:     {len(results)}")
    print(f"M1 Success:       {m1_success} ({m1_success/len(results)*100:.1f}%)")
    print(f"M3 Success:       {m3_success} ({m3_success/len(results)*100:.1f}%)")

    if results:
        m2_angles = [r['m2_correction'] for r in results]
        print(f"\nM2 Correction Angles:")
        print(f"  Mean:     {np.mean(m2_angles):.2f}°")
        print(f"  Std:      {np.std(m2_angles):.2f}°")
        print(f"  Min:      {np.min(m2_angles):.2f}°")
        print(f"  Max:      {np.max(m2_angles):.2f}°")

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path.absolute()}")
    print("=" * 80)


def save_pipeline_results(
    original: np.ndarray,
    m1_vis: np.ndarray,
    m2_vis: np.ndarray,
    meter_crop: np.ndarray,
    meter_aligned: np.ndarray,
    counter_result: tuple,
    name: str,
    output_path: Path,
    idx: int
):
    """
    Save pipeline results as visualizations.
    """
    base_name = f"{idx+1:02d}_{name}"

    # Save individual steps
    cv2.imwrite(str(output_path / f"{base_name}_0_original.jpg"), original)
    cv2.imwrite(str(output_path / f"{base_name}_1_m1_detection.jpg"), m1_vis)
    cv2.imwrite(str(output_path / f"{base_name}_2_m2_info.jpg"), m2_vis)
    cv2.imwrite(str(output_path / f"{base_name}_3_m1_crop.jpg"), meter_crop)
    cv2.imwrite(str(output_path / f"{base_name}_4_m2_aligned.jpg"), meter_aligned)

    # Create M3 visualization
    if counter_result is not None:
        cx1, cy1, cx2, cy2, conf_m3, counter_roi = counter_result
        m3_vis = meter_aligned.copy()
        cv2.rectangle(m3_vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
        cv2.putText(m3_vis, f"M3: {conf_m3:.2f}", (cx1, cy1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(str(output_path / f"{base_name}_5_m3_detection.jpg"), m3_vis)
        cv2.imwrite(str(output_path / f"{base_name}_6_m3_roi.jpg"), counter_roi)

        # Create pipeline summary
        create_pipeline_summary(
            original, m1_vis, m2_vis, meter_aligned, m3_vis, counter_roi,
            output_path / f"{base_name}_pipeline_summary.jpg"
        )
    else:
        # M3 failed - show what we have
        create_pipeline_summary_no_m3(
            original, m1_vis, m2_vis, meter_crop, meter_aligned,
            output_path / f"{base_name}_pipeline_summary.jpg"
        )


def create_pipeline_summary(
    original: np.ndarray,
    m1_vis: np.ndarray,
    m2_vis: np.ndarray,
    meter_crop: np.ndarray,
    meter_aligned: np.ndarray,
    m3_vis: np.ndarray,
    counter_roi: np.ndarray,
    save_path: Path
):
    """Create a summary visualization of the complete pipeline."""
    # Resize all to same dimensions
    target_h = 250

    orig_resized = cv2.resize(original, (int(original.shape[1] * target_h / original.shape[0]), target_h))
    m1_resized = cv2.resize(m1_vis, (int(m1_vis.shape[1] * target_h / m1_vis.shape[0]), target_h))
    m2_resized = cv2.resize(m2_vis, (int(m2_vis.shape[1] * target_h / m2_vis.shape[0]), target_h))
    crop_resized = cv2.resize(meter_crop, (int(meter_crop.shape[1] * target_h / meter_crop.shape[0]), target_h))
    aligned_resized = cv2.resize(meter_aligned, (int(meter_aligned.shape[1] * target_h / meter_aligned.shape[0]), target_h))
    m3_resized = cv2.resize(m3_vis, (int(m3_vis.shape[1] * target_h / m3_vis.shape[0]), target_h))
    roi_resized = cv2.resize(counter_roi, (int(counter_roi.shape[1] * target_h / counter_roi.shape[0]), target_h))

    # Create 2x3 grid
    row1 = np.hstack([orig_resized, m1_resized, m2_resized])
    row2 = np.hstack([crop_resized, aligned_resized, m3_resized])

    # Add padding between rows
    padding = np.zeros((20, row1.shape[1], 3), dtype=np.uint8)
    summary = np.vstack([row1, padding, row2])

    # Add labels
    cv2.putText(summary, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(summary, "M1: DETECTION", (orig_resized.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(summary, "M2: INFO", (orig_resized.shape[1] + m1_resized.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(summary, "M1: CROP", (10, target_h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(summary, "M2: ALIGNED", (crop_resized.shape[1] + 10, target_h + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(summary, "M3: DETECTION", (crop_resized.shape[1] + aligned_resized.shape[1] + 10, target_h + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(str(save_path), summary)


def create_pipeline_summary_no_m3(
    original: np.ndarray,
    m1_vis: np.ndarray,
    m2_vis: np.ndarray,
    meter_crop: np.ndarray,
    meter_aligned: np.ndarray,
    save_path: Path
):
    """Create summary when M3 fails."""
    target_h = 250

    # Resize all images to same height
    orig_resized = cv2.resize(original, (int(original.shape[1] * target_h / original.shape[0]), target_h))
    m1_resized = cv2.resize(m1_vis, (int(m1_vis.shape[1] * target_h / m1_vis.shape[0]), target_h))
    m2_resized = cv2.resize(m2_vis, (int(m2_vis.shape[1] * target_h / m2_vis.shape[0]), target_h))
    crop_resized = cv2.resize(meter_crop, (int(meter_crop.shape[1] * target_h / meter_crop.shape[0]), target_h))
    aligned_resized = cv2.resize(meter_aligned, (int(meter_aligned.shape[1] * target_h / meter_aligned.shape[0]), target_h))

    # Create 3 rows with consistent width
    row1 = np.hstack([orig_resized, m1_resized])
    row2 = np.hstack([m2_resized, crop_resized])
    # For row3, duplicate aligned to match row2 width
    row3 = np.hstack([aligned_resized, aligned_resized])

    # Pad all rows to same width
    max_width = max(row1.shape[1], row2.shape[1], row3.shape[1])
    if row1.shape[1] < max_width:
        pad_width = max_width - row1.shape[1]
        row1 = np.hstack([row1, np.zeros((target_h, pad_width, 3), dtype=np.uint8)])
    if row2.shape[1] < max_width:
        pad_width = max_width - row2.shape[1]
        row2 = np.hstack([row2, np.zeros((target_h, pad_width, 3), dtype=np.uint8)])
    if row3.shape[1] < max_width:
        pad_width = max_width - row3.shape[1]
        row3 = np.hstack([row3, np.zeros((target_h, pad_width, 3), dtype=np.uint8)])

    padding = np.zeros((15, max_width, 3), dtype=np.uint8)
    summary = np.vstack([row1, padding, row2, padding, row3])

    cv2.putText(summary, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(summary, "M1: DETECTION", (orig_resized.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(summary, "M2: INFO", (10, target_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(summary, "M1: CROP", (m2_resized.shape[1] + 10, target_h + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(summary, "M2: ALIGNED (M3 FAILED)", (10, target_h * 2 + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(str(save_path), summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test M1 → M2 → M3 pipeline with CORRECTED M2 logic")
    parser.add_argument("--input", type=str,
                       default=r"F:\Workspace\Project\data\raw_images",
                       help="Path to original watermeter images")
    parser.add_argument("--m1-model", type=str,
                       default=r"F:\Workspace\Project\model\M1_DetectWatermeter.pt",
                       help="Path to M1 model")
    parser.add_argument("--m2-model", type=str,
                       default=r"F:\Workspace\Project\model\M2_Orientation.pth",
                       help="Path to M2 model")
    parser.add_argument("--m3-model", type=str,
                       default=r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt",
                       help="Path to M3 model")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of samples to test (0 = all)")
    parser.add_argument("--output-dir", type=str,
                       default=r"F:\Workspace\Project\results\test_pipeline_corrected",
                       help="Output directory for results")

    args = parser.parse_args()

    test_pipeline(
        original_images_dir=args.input,
        m1_model_path=args.m1_model,
        m2_model_path=args.m2_model,
        m3_model_path=args.m3_model,
        num_samples=args.samples,
        output_dir=args.output_dir
    )

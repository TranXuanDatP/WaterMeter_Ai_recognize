#!/usr/bin/env python3
"""
Complete Pipeline M1 -> M2 + Smart Rotate -> M3 -> M3.5 -> M4 + Beam Search

Enhanced version with DETAILED METRICS LOGGING for each image:
- Confidence scores (M1, M3)
- Execution time per stage
- M3.5 preprocessing threshold
- Error tracking per stage
- Export to detailed CSV + JSON

Input: Raw images
Output: Final reading with confidence & detailed metrics
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import json
from tqdm import tqdm
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import M3.5 module
from src.m3_5_digit_extraction import M3_5_DigitExtractor

# Import M2 Orientation Model
from src.m2_orientation.model import M2_OrientationModel

# Import Beam Search Decoder
from src.m4_crnn_reading.beam_search_decoder import create_decoder

# Import Detailed Metrics Logger
from src.common.logging_base import PerImageMetricsCollector

# Fix encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ====================== CONFIGURATION ======================
class Config:
    # Model paths
    M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
    M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
    M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"
    M4_MODEL = r"F:\Workspace\Project\model\M4_OCR.pth"

    # Input/Output
    DATA_DIR = Path(r"F:\Workspace\Project\data\raw_images")
    LABELS_FILE = Path(r"F:\Workspace\Project\data\labels.csv")
    OUTPUT_DIR = Path(r"F:\Workspace\Project\results\pipeline_detailed")

    # Stage outputs
    M1_CROPS_DIR = OUTPUT_DIR / "m1_crops"
    M2_ALIGNED_DIR = OUTPUT_DIR / "m2_aligned"
    M3_ROI_DIR = OUTPUT_DIR / "m3_roi_crops"
    M3_5_DIGITS_DIR = OUTPUT_DIR / "m3_5_black_digits"

    # Thresholds
    M1_CONFIDENCE = 0.15
    M2_CONFIDENCE = 0.5
    M3_CONFIDENCE = 0.1
    M4_CONFIDENCE = 0.7

    # Beam Search Decoder Configuration
    BEAM_SEARCH_METHOD = 'beam'
    BEAM_WIDTH = 10

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================== M1: WATER METER DETECTION ======================
class M1_WaterMeterDetector:
    """M1: Detect water meter in image using YOLO"""

    def __init__(self, model_path: str, confidence: float = 0.15):
        print("[M1] Loading water meter detection model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"      Model: {model_path}")
        print(f"      Confidence threshold: {confidence}")

    def detect(self, image: np.ndarray) -> dict:
        """
        Detect water meter in image

        Returns:
            dict with detection results including confidence
        """
        start_time = time.time()

        results = self.model(image, verbose=False)

        elapsed_ms = (time.time() - start_time) * 1000

        if len(results) == 0 or results[0].boxes is None:
            return {
                'detected': False,
                'confidence': 0.0,
                'bbox': None,
                'time_ms': elapsed_ms
            }

        boxes = results[0].boxes

        # Get best detection by confidence
        if len(boxes) == 0:
            return {
                'detected': False,
                'confidence': 0.0,
                'bbox': None,
                'time_ms': elapsed_ms
            }

        best_idx = boxes.conf.argmax()
        confidence = float(boxes.conf[best_idx])
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

        return {
            'detected': True,
            'confidence': confidence,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'time_ms': elapsed_ms
        }


# ====================== M2: ORIENTATION CORRECTION ======================
class M2_OrientationCorrector:
    """M2: Correct orientation using angle regression model"""

    def __init__(self, model_path: str, device: torch.device):
        print("[M2] Loading orientation correction model...")
        self.device = device
        self.model = M2_OrientationModel(dropout=0.4).to(device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"      Model: {model_path}")
        print(f"      Device: {device}")

    def predict_angle(self, image: np.ndarray) -> float:
        """Predict rotation angle for image"""
        start_time = time.time()

        # Convert PIL if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)

        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            sin_cos = self.model(input_tensor)

        # Extract angle
        sin_val = sin_cos[0, 0].cpu().item()
        cos_val = sin_cos[0, 1].cpu().item()
        angle_rad = np.arctan2(sin_val, cos_val)
        angle_deg = np.degrees(angle_rad)
        angle_deg = (angle_deg + 360) % 360

        elapsed_ms = (time.time() - start_time) * 1000

        return angle_deg, elapsed_ms

    def smart_rotate(self, image: np.ndarray, angle: float) -> tuple:
        """
        Rotate image by angle

        Returns:
            tuple: (rotated_image, correction_angle, time_ms)
        """
        start_time = time.time()

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Calculate correction angle (negative of detected angle)
        correction_angle = -angle

        # Rotate
        M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

        elapsed_ms = (time.time() - start_time) * 1000

        return rotated, correction_angle, elapsed_ms


# ====================== M3: ROI DETECTION ======================
class M3_ROIDetector:
    """M3: Detect digit ROI using YOLO"""

    def __init__(self, model_path: str, confidence: float = 0.1):
        print("[M3] Loading ROI detection model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"      Model: {model_path}")
        print(f"      Confidence threshold: {confidence}")

    def detect(self, image: np.ndarray) -> dict:
        """
        Detect digit ROI in image

        Returns:
            dict with detection results including confidence
        """
        start_time = time.time()

        results = self.model(image, verbose=False)

        elapsed_ms = (time.time() - start_time) * 1000

        if len(results) == 0 or results[0].boxes is None:
            return {
                'detected': False,
                'confidence': 0.0,
                'bbox': None,
                'time_ms': elapsed_ms
            }

        boxes = results[0].boxes

        if len(boxes) == 0:
            return {
                'detected': False,
                'confidence': 0.0,
                'bbox': None,
                'time_ms': elapsed_ms
            }

        best_idx = boxes.conf.argmax()
        confidence = float(boxes.conf[best_idx])
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

        return {
            'detected': True,
            'confidence': confidence,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'time_ms': elapsed_ms
        }


# ====================== M3.5: BLACK DIGIT EXTRACTION ======================
class M3_5_DigitExtractorEnhanced(M3_5_DigitExtractor):
    """Enhanced M3.5 with timing and threshold tracking"""

    def extract_with_metrics(self, image: np.ndarray) -> dict:
        """
        Extract black digits with metrics

        Returns:
            dict with extracted image and metrics
        """
        start_time = time.time()

        # Call parent extract
        result = super().extract(image)

        elapsed_ms = (time.time() - start_time) * 1000

        # Get threshold used (check if available)
        threshold = getattr(self, 'last_threshold', 127)

        # Count digits (estimate by contours)
        num_digits = self._estimate_digit_count(result) if result is not None else 0

        return {
            'image': result,
            'threshold': threshold,
            'num_digits': num_digits,
            'time_ms': elapsed_ms
        }

    def _estimate_digit_count(self, image: np.ndarray) -> int:
        """Estimate number of digits by counting contours"""
        if image is None:
            return 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        digit_contours = [c for c in contours if cv2.contourArea(c) > 50]

        return len(digit_contours)


# ====================== M4: OCR WITH BEAM SEARCH ======================
class M4_OCRReader:
    """M4: CRNN OCR with Beam Search Decoder"""

    def __init__(self, model_path: str, device: torch.device,
                 beam_method: str = 'beam', beam_width: int = 10):
        print("[M4] Loading OCR model with Beam Search...")
        self.device = device
        self.beam_method = beam_method
        self.beam_width = beam_width

        # Load model
        self.model = CRNN(num_chars=11).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Create decoder
        self.decoder = create_decoder(
            method=beam_method,
            chars="0123456789",
            blank_idx=10,
            beam_width=beam_width
        )

        print(f"      Model: {model_path}")
        print(f"      Beam method: {beam_method}")
        print(f"      Beam width: {beam_width}")

    def predict(self, image: np.ndarray) -> dict:
        """
        Predict text from image with metrics

        Returns:
            dict with prediction and confidence
        """
        start_time = time.time()

        # Preprocess
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        resized = cv2.resize(gray, (224, 64))
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)

        logits = outputs[:, 0, :]

        # Decode with beam search
        decoded_text = self.decoder.decode(logits)

        if isinstance(decoded_text, list) and len(decoded_text) > 0:
            predicted_text = decoded_text[0]
        else:
            predicted_text = str(decoded_text)

        # Filter to digits only
        predicted_text = ''.join([c for c in predicted_text if c.isdigit()])

        elapsed_ms = (time.time() - start_time) * 1000

        # Calculate confidence (simplified - use softmax of best path)
        confidence = self._calculate_beam_confidence(logits, predicted_text)

        return {
            'text': predicted_text,
            'confidence': confidence,
            'time_ms': elapsed_ms
        }

    def _calculate_beam_confidence(self, logits: torch.Tensor, text: str) -> float:
        """Calculate confidence score for beam search result"""
        # Simplified: use max softmax probability
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        return float(max_probs.mean().item())


# Import CRNN model
from src.m4_crnn_reading.model import CRNN


# ====================== MAIN PIPELINE ======================
def main():
    print("=" * 80)
    print("PIPELINE WITH DETAILED METRICS LOGGING")
    print("=" * 80)
    print(f"\nOutput directory: {Config.OUTPUT_DIR}")
    print(f"Device: {Config.DEVICE}")
    print("=" * 80)

    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.M1_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    Config.M2_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    Config.M3_ROI_DIR.mkdir(parents=True, exist_ok=True)
    Config.M3_5_DIGITS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize metrics collector
    metrics = PerImageMetricsCollector(output_dir=Config.OUTPUT_DIR)

    print("\n[1/6] Loading models...")

    # Load models
    m1_detector = M1_WaterMeterDetector(Config.M1_MODEL, Config.M1_CONFIDENCE)
    m2_corrector = M2_OrientationCorrector(Config.M2_MODEL, Config.DEVICE)
    m3_detector = M3_ROIDetector(Config.M3_MODEL, Config.M3_CONFIDENCE)
    m3_5_extractor = M3_5_DigitExtractorEnhanced()
    m4_reader = M4_OCRReader(Config.M4_MODEL, Config.DEVICE,
                             Config.BEAM_SEARCH_METHOD, Config.BEAM_WIDTH)

    print("\n[2/6] Loading data...")
    labels_df = pd.read_csv(Config.LABELS_FILE)
    image_files = list(Config.DATA_DIR.glob("*.jpg"))
    print(f">> Found {len(image_files)} images\n")

    print("[3/6] Running pipeline with detailed metrics...")

    for img_path in tqdm(image_files, desc="Processing"):
        img_name = img_path.name

        # Get label
        label_row = labels_df[labels_df['photo_name'] == img_name]
        if len(label_row) == 0:
            continue

        true_value = str(label_row.iloc[0]['value'])

        # Start metrics for this image
        metrics.start_image(img_name)

        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                metrics.add_error('read', 'Failed to read image')
                metrics.finalize_image(true_value, False)
                continue

            # === M1: Water Meter Detection ===
            m1_result = m1_detector.detect(img)

            if not m1_result['detected']:
                metrics.add_error('M1', 'No water meter detected')
                metrics.finalize_image(true_value, False)
                continue

            metrics.add_m1_result(
                bbox=m1_result['bbox'],
                confidence=m1_result['confidence'],
                time_ms=m1_result['time_ms']
            )

            x1, y1, x2, y2 = m1_result['bbox']
            meter_crop = img[y1:y2, x1:x2]

            # === M2: Orientation Correction ===
            detected_angle, m2_pred_time = m2_corrector.predict_angle(meter_crop)
            meter_aligned, correction_angle, m2_rot_time = m2_corrector.smart_rotate(meter_crop, detected_angle)

            m2_total_time = m2_pred_time + m2_rot_time
            metrics.add_m2_result(
                detected_angle=detected_angle,
                correction_angle=correction_angle,
                time_ms=m2_total_time
            )

            # === M3: ROI Detection ===
            m3_result = m3_detector.detect(meter_aligned)

            if not m3_result['detected']:
                metrics.add_error('M3', 'No ROI detected')
                metrics.finalize_image(true_value, False)
                continue

            metrics.add_m3_result(
                bbox=m3_result['bbox'],
                confidence=m3_result['confidence'],
                time_ms=m3_result['time_ms']
            )

            cx1, cy1, cx2, cy2 = m3_result['bbox']
            roi_crop = meter_aligned[cy1:cy2, cx1:cx2]

            # === M3.5: Black Digit Extraction ===
            m3_5_result = m3_5_extractor.extract_with_metrics(roi_crop)

            metrics.add_m3_5_result(
                threshold=m3_5_result['threshold'],
                num_digits=m3_5_result['num_digits'],
                time_ms=m3_5_result['time_ms']
            )

            black_digits = m3_5_result['image']

            # === M4: OCR ===
            m4_result = m4_reader.predict(black_digits)

            metrics.add_m4_result(
                predicted=m4_result['text'],
                beam_confidence=m4_result['confidence'],
                time_ms=m4_result['time_ms']
            )

            # Save intermediate results
            base_name = img_path.stem
            cv2.imwrite(str(Config.M1_CROPS_DIR / f"{base_name}.jpg"), meter_crop)
            cv2.imwrite(str(Config.M2_ALIGNED_DIR / f"{base_name}.jpg"), meter_aligned)
            cv2.imwrite(str(Config.M3_ROI_DIR / f"{base_name}.jpg"), roi_crop)
            cv2.imwrite(str(Config.M3_5_DIGITS_DIR / f"{base_name}.jpg"), black_digits)

            # Check correctness
            is_correct = (true_value == m4_result['text'])
            metrics.finalize_image(true_value=true_value, correct=is_correct)

        except Exception as e:
            metrics.add_error('pipeline', str(e))
            metrics.finalize_image(true_value, False)
            print(f"\nError processing {img_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n>> Processed {len(metrics.results)} images\n")

    print("[4/6] Saving detailed results...")
    csv_path = metrics.save_csv("pipeline_detailed_results.csv")
    print(f">> CSV saved: {csv_path}")

    print("\n[5/6] Saving metrics JSON...")
    json_path = metrics.save_json("pipeline_metrics.json")
    print(f">> JSON saved: {json_path}")

    print("\n[6/6] Generating analysis report...")

    # Generate analysis
    summary = metrics._compute_summary()

    print(f"\n{'='*80}")
    print("PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total images: {summary.get('total_images', 0)}")
    print(f"Successful: {summary.get('successful', 0)}")
    print(f"Errors: {summary.get('errors', 0)}")
    print(f"Correct predictions: {summary.get('correct_predictions', 0)}")
    print(f"Accuracy: {summary.get('accuracy', 0):.2%}")

    if 'm1_confidence' in summary:
        m1_conf = summary['m1_confidence']
        print(f"\nM1 Confidence: mean={m1_conf['mean']:.3f}, min={m1_conf['min']:.3f}, max={m1_conf['max']:.3f}")

    if 'm3_confidence' in summary:
        m3_conf = summary['m3_confidence']
        print(f"M3 Confidence: mean={m3_conf['mean']:.3f}, min={m3_conf['min']:.3f}, max={m3_conf['max']:.3f}")

    if 'timing' in summary:
        timing = summary['timing']
        print(f"\nTiming (ms): mean={timing['total_mean_ms']:.1f}, min={timing['total_min_ms']:.1f}, max={timing['total_max_ms']:.1f}")

    if 'errors_by_stage' in summary:
        print(f"\nErrors by stage:")
        for stage, count in summary['errors_by_stage'].items():
            print(f"  {stage}: {count}")

    print(f"{'='*80}")

    # Export low confidence images list
    low_m3 = metrics.get_low_confidence_images('m3', threshold=0.3)
    if low_m3:
        low_conf_file = Config.OUTPUT_DIR / "low_m3_confidence.txt"
        with open(low_conf_file, 'w') as f:
            for img in low_m3:
                f.write(f"{img}\n")
        print(f"\n>> Low M3 confidence images saved: {low_conf_file}")

    # Export error images list
    error_imgs = metrics.get_error_images()
    if error_imgs:
        error_file = Config.OUTPUT_DIR / "error_images.txt"
        with open(error_file, 'w') as f:
            for img in error_imgs:
                f.write(f"{img}\n")
        print(f">> Error images saved: {error_file}")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()

"""
Full Pipeline Test on data_4digit2 Dataset

This script runs the complete pipeline (M1 -> M2 -> M3 -> M3.5 -> M4)
on the data_4digit2 dataset.

Pipeline stages:
1. M1: Water Meter Detection (YOLO)
2. M2: Orientation + Smart Rotate (Angle Regression)
3. M3: ROI Detection (YOLOv8n)
4. M3.5: Black Digit Extraction
5. M4: CRNN OCR Reading + Beam Search Decoder

Dataset: data/data_4digit2/ (images) + data/images_4digit2.csv (labels)
"""
import sys
import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules
from src.m2_orientation.model import M2_OrientationModel
from src.m3_5_digit_extraction import M3_5_DigitExtractor
from src.m4_crnn_reading.model import CRNN
from src.m4_crnn_reading.beam_search_decoder import create_decoder

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ============================================
# CONFIGURATION
# ============================================

class Config:
    # Model paths
    M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
    M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
    M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"
    M4_MODEL = r"F:\Workspace\Project\model\M4_OCR.pth"

    # Data paths
    DATA_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
    LABELS_FILE = Path(r"F:\Workspace\Project\data\images_4digit2.csv")

    # Output paths
    OUTPUT_DIR = Path(r"F:\Workspace\Project\results\full_pipeline_data_4digit2")
    LOG_DIR = OUTPUT_DIR / "logs"

    # Stage outputs
    M1_CROPS_DIR = OUTPUT_DIR / "m1_crops"
    M2_ALIGNED_DIR = OUTPUT_DIR / "m2_aligned"
    M3_ROI_DIR = OUTPUT_DIR / "m3_roi_crops"
    M3_5_DIGITS_DIR = OUTPUT_DIR / "m3_5_black_digits"

    # Thresholds
    M1_CONFIDENCE = 0.25
    M2_CONFIDENCE = 0.5
    M3_CONFIDENCE = 0.25

    # M4 config
    IMG_SIZE = (64, 224)
    CHAR_MAP = "0123456789"

    # Decoder config
    DECODER_METHOD = 'beam'
    BEAM_WIDTH = 10

    # M3.5 config
    M3_5_MIN_CROP_RATIO = 0.75
    M3_5_FALLBACK_RATIO = 0.8

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test on first N images (set to None for full dataset)
    SAMPLE_SIZE = None  # None = all images

# ============================================
# PIPELINE STAGES
# ============================================

class M1_WaterMeterDetector:
    """M1: Detect water meter in image using YOLO"""

    def __init__(self, model_path: str, confidence: float = 0.25):
        print("[M1] Loading water meter detection model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"      Model: {model_path}")
        print(f"      Confidence threshold: {confidence}")

    def detect(self, image: np.ndarray) -> dict:
        """Detect water meter in image"""
        results = self.model(image, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return {'detected': False, 'bbox': None, 'confidence': 0.0}

        # Get best detection
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()

        if confidences[best_idx] < self.confidence:
            return {
                'detected': False,
                'bbox': None,
                'confidence': float(confidences[best_idx])
            }

        # Get bounding box
        box = boxes[best_idx]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        return {
            'detected': True,
            'bbox': (x1, y1, x2, y2),
            'confidence': float(confidences[best_idx])
        }

    def crop(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        """Crop water meter region"""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]


class M2_SmartRotator:
    """M2: Orientation prediction + Smart Rotate"""

    def __init__(self, model_path: str, device: torch.device):
        print("[M2] Loading orientation + smart rotate model...")
        self.device = device
        self.model = M2_OrientationModel().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        print(f"      Model: {model_path}")
        print(f"      Device: {device}")

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
        """Smart rotate image"""
        # Normalize angle to [-45, 45]
        if angle > 180:
            angle -= 360
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        # Rotate
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        return rotated, angle


class M3_ROIDetector:
    """M3: Detect ROI region using YOLO"""

    def __init__(self, model_path: str, confidence: float = 0.25):
        print("[M3] Loading ROI detection model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"      Model: {model_path}")
        print(f"      Confidence threshold: {confidence}")

    def detect(self, image: np.ndarray) -> dict:
        """Detect ROI in image"""
        results = self.model(image, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return {'detected': False, 'bbox': None, 'confidence': 0.0}

        # Get best detection
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()

        if confidences[best_idx] < self.confidence:
            return {
                'detected': False,
                'bbox': None,
                'confidence': float(confidences[best_idx])
            }

        # Get bounding box
        box = boxes[best_idx]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        return {
            'detected': True,
            'bbox': (x1, y1, x2, y2),
            'confidence': float(confidences[best_idx])
        }

    def crop(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        """Crop ROI region"""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]


class M4_CRNNOCR:
    """M4: CRNN OCR Reading"""

    def __init__(self, model_path: str, device: torch.device,
                 decoder_method: str = 'beam', beam_width: int = 10):
        print("[M4] Loading CRNN OCR model...")
        self.device = device
        self.char_map = "0123456789"
        self.blank_idx = 10

        # Load model
        self.model = CRNN(num_chars=len(self.char_map) + 1)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.model = self.model.to(device)

        # Initialize beam search decoder
        self.decoder = create_decoder(
            method=decoder_method,
            chars=self.char_map,
            blank_idx=self.blank_idx,
            beam_width=beam_width
        )

        print(f"      Model: {model_path}")
        print(f"      Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"      Decoder: {decoder_method} (beam_width={beam_width})")

    def preprocess(self, img: np.ndarray, target_size=(64, 224)) -> torch.Tensor:
        """Preprocess image for CRNN"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (target_size[1], target_size[0]))
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor

    def recognize(self, image: np.ndarray) -> dict:
        """Recognize text from image"""
        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)

        # Get logits
        logits = output[:, 0, :]

        # Decode using beam search
        text = self.decoder.decode(logits)

        # Calculate confidence
        probs = torch.softmax(logits, dim=1)
        max_probs = probs.max(dim=1)[0].cpu().numpy()
        confidence = float(np.mean(max_probs))

        return {
            'text': text,
            'confidence': confidence
        }


# ============================================
# LOGGER
# ============================================

class PipelineLogger:
    """Logger for pipeline execution"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.start_time = datetime.now()
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create log file
        self.log_file = log_dir / f"pipeline_run_{self.session_id}.log"

        # Statistics
        self.stats = {
            'total_images': 0,
            'm1_success': 0,
            'm2_success': 0,
            'm3_success': 0,
            'm3_5_success': 0,
            'm4_success': 0,
            'correct_predictions': 0,
            'start_time': self.start_time.isoformat(),
        }

    def log(self, message, print_to_console=True):
        """Log message to file and optionally print to console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"

        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

        # Print to console
        if print_to_console:
            print(message)

    def save_summary(self):
        """Save execution summary as JSON"""
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['duration_seconds'] = (
            datetime.now() - self.start_time
        ).total_seconds()

        summary_file = self.log_dir / f"summary_{self.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        return summary_file


# ============================================
# MAIN PIPELINE
# ============================================

def main():
    print("=" * 80)
    print("FULL PIPELINE TEST ON DATA_4DIGIT2")
    print("=" * 80)
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Labels file: {Config.LABELS_FILE}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Device: {Config.DEVICE}")
    print("=" * 80)

    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.M1_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    Config.M2_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    Config.M3_ROI_DIR.mkdir(parents=True, exist_ok=True)
    Config.M3_5_DIGITS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = PipelineLogger(Config.LOG_DIR)
    logger.log("=" * 80)
    logger.log("FULL PIPELINE TEST ON DATA_4DIGIT2")
    logger.log("=" * 80)

    # Load labels
    print(f"\n[1/8] Loading labels...")
    logger.log(f"Loading labels from: {Config.LABELS_FILE}")

    try:
        labels_df = pd.read_csv(Config.LABELS_FILE)
        print(f"      Loaded {len(labels_df)} labels")
        logger.log(f"Loaded {len(labels_df)} labels")
    except Exception as e:
        print(f"      ERROR: Failed to load labels: {e}")
        logger.log(f"ERROR: Failed to load labels: {e}")
        sys.exit(1)

    # Check data directory
    print(f"\n[2/8] Checking data directory...")
    logger.log(f"Checking data directory: {Config.DATA_DIR}")

    image_files = list(Config.DATA_DIR.glob('*.jpg')) + list(Config.DATA_DIR.glob('*.png'))
    print(f"      Found {len(image_files)} images")
    logger.log(f"Found {len(image_files)} images")

    # Sample if needed
    if Config.SAMPLE_SIZE is not None:
        image_files = sorted(image_files)[:Config.SAMPLE_SIZE]
        print(f"      Testing on first {len(image_files)} images")
        logger.log(f"Testing on first {len(image_files)} images")

    # Initialize pipeline stages
    print(f"\n[3/8] Initializing pipeline stages...")

    m1_detector = M1_WaterMeterDetector(Config.M1_MODEL, Config.M1_CONFIDENCE)
    m2_rotator = M2_SmartRotator(Config.M2_MODEL, Config.DEVICE)
    m3_detector = M3_ROIDetector(Config.M3_MODEL, Config.M3_CONFIDENCE)
    m3_5_extractor = M3_5_DigitExtractor(
        min_crop_ratio=Config.M3_5_MIN_CROP_RATIO,
        fallback_ratio=Config.M3_5_FALLBACK_RATIO
    )
    m4_ocr = M4_CRNNOCR(
        Config.M4_MODEL,
        Config.DEVICE,
        decoder_method=Config.DECODER_METHOD,
        beam_width=Config.BEAM_WIDTH
    )

    logger.log("Pipeline stages initialized")

    # Run pipeline
    print(f"\n[4/8] Running pipeline on {len(image_files)} images...")
    logger.log(f"Starting pipeline processing on {len(image_files)} images")
    print("-" * 80)

    results = []
    error_files = []

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            img_name = img_path.name

            # Get ground truth label
            label_rows = labels_df[labels_df['photo_name'] == img_name]
            if len(label_rows) == 0:
                ground_truth = ""
            else:
                ground_truth = str(label_rows.iloc[0]['value']).strip()

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                raise Exception("Could not read image")

            original_size = f"{img.shape[1]}x{img.shape[0]}"

            # M1: Detect water meter
            m1_result = m1_detector.detect(img)
            if not m1_result['detected']:
                results.append({
                    'filename': img_name,
                    'ground_truth': ground_truth,
                    'predicted_text': '',
                    'confidence': 0.0,
                    'is_correct': False,
                    'error_stage': 'M1',
                    'original_size': original_size
                })
                continue

            m1_crop = m1_detector.crop(img, m1_result['bbox'])
            logger.stats['m1_success'] += 1

            # M2: Rotate image
            angle = m2_rotator.predict_angle(m1_crop)
            m2_aligned, actual_angle = m2_rotator.smart_rotate(m1_crop, angle)
            logger.stats['m2_success'] += 1

            # Save M2 aligned image
            cv2.imwrite(str(Config.M2_ALIGNED_DIR / img_name), m2_aligned)

            # M3: Detect ROI
            m3_result = m3_detector.detect(m2_aligned)
            if not m3_result['detected']:
                results.append({
                    'filename': img_name,
                    'ground_truth': ground_truth,
                    'predicted_text': '',
                    'confidence': 0.0,
                    'is_correct': False,
                    'error_stage': 'M3',
                    'original_size': original_size
                })
                continue

            m3_crop = m3_detector.crop(m2_aligned, m3_result['bbox'])
            logger.stats['m3_success'] += 1

            # Save M3 ROI crop
            cv2.imwrite(str(Config.M3_ROI_DIR / img_name), m3_crop)

            # M3.5: Extract black digits
            m3_5_crop, metadata = m3_5_extractor.extract(m3_crop)
            logger.stats['m3_5_success'] += 1

            # Save M3.5 black digits
            cv2.imwrite(str(Config.M3_5_DIGITS_DIR / img_name), m3_5_crop)

            # M4: OCR recognition
            m4_result = m4_ocr.recognize(m3_5_crop)
            predicted_text = m4_result['text']
            confidence = m4_result['confidence']
            logger.stats['m4_success'] += 1

            # Check if correct
            is_correct = (predicted_text == ground_truth) if ground_truth else None
            if is_correct:
                logger.stats['correct_predictions'] += 1

            results.append({
                'filename': img_name,
                'ground_truth': ground_truth,
                'predicted_text': predicted_text,
                'confidence': confidence,
                'is_correct': is_correct,
                'error_stage': None,
                'original_size': original_size
            })

            # Update total
            logger.stats['total_images'] += 1

            # Log incorrect predictions
            if is_correct is False:
                logger.log(f"INCORRECT: {img_name} | GT: '{ground_truth}' | Pred: '{predicted_text}' | Conf: {confidence:.4f}")

        except Exception as e:
            error_files.append((img_name, str(e)))
            logger.stats['total_images'] += 1
            logger.log(f"ERROR processing {img_name}: {e}")

    # ============================================
    # SAVE RESULTS
    # ============================================

    print(f"\n[5/8] Saving results...")

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Save detailed results
    output_csv = Config.OUTPUT_DIR / "pipeline_results.csv"
    df.to_csv(output_csv, index=False)
    logger.log(f"Results saved to: {output_csv}")

    # Save only incorrect predictions
    if len(df[df['is_correct'] == False]) > 0:
        incorrect_csv = Config.OUTPUT_DIR / "incorrect_predictions.csv"
        df[df['is_correct'] == False].to_csv(incorrect_csv, index=False)
        logger.log(f"Incorrect predictions saved to: {incorrect_csv}")

    # Save summary
    summary_file = logger.save_summary()

    # ============================================
    # PRINT SUMMARY
    # ============================================

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)

    total = logger.stats['total_images']
    m1_success = logger.stats['m1_success']
    m2_success = logger.stats['m2_success']
    m3_success = logger.stats['m3_success']
    m3_5_success = logger.stats['m3_5_success']
    m4_success = logger.stats['m4_success']
    correct = logger.stats['correct_predictions']

    print(f"\nTotal images:     {total}")
    print(f"\nStage Success Rates:")
    print(f"  M1 (Detection):  {m1_success}/{total} ({m1_success/total*100:.1f}%)")
    print(f"  M2 (Rotation):   {m2_success}/{total} ({m2_success/total*100:.1f}%)")
    print(f"  M3 (ROI):        {m3_success}/{total} ({m3_success/total*100:.1f}%)")
    print(f"  M3.5 (Digits):   {m3_5_success}/{total} ({m3_5_success/total*100:.1f}%)")
    print(f"  M4 (OCR):        {m4_success}/{total} ({m4_success/total*100:.1f}%)")

    if m4_success > 0:
        accuracy = correct / m4_success * 100
        print(f"\nFinal Accuracy (on {m4_success} successful OCR predictions):")
        print(f"  Correct:         {correct}")
        print(f"  Incorrect:       {m4_success - correct}")
        print(f"  Accuracy:        {accuracy:.2f}%")

    # Show incorrect predictions
    if len(df[df['is_correct'] == False]) > 0:
        print(f"\n" + "=" * 80)
        print("INCORRECT PREDICTIONS (First 20)")
        print("=" * 80)

        incorrect_df = df[df['is_correct'] == False].head(20)

        print(f"\n{'Filename':<40} {'Ground Truth':<15} {'Predicted':<15} {'Conf':<10}")
        print("-" * 80)

        for _, row in incorrect_df.iterrows():
            print(f"{row['filename']:<40} {row['ground_truth']:<15} {row['predicted_text']:<15} {row['confidence']:<10.4f}")

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Results CSV:      {output_csv}")
    if len(df[df['is_correct'] == False]) > 0:
        print(f"Incorrect CSV:    {incorrect_csv}")
    print(f"Summary JSON:     {summary_file}")
    print(f"Log file:         {logger.log_file}")

    print("\n" + "=" * 80)
    print("[OK] PIPELINE TEST COMPLETED!")
    print("=" * 80)

    print(f"\n💡 Next steps:")
    print("  1. Review incorrect predictions CSV")
    print("  2. Check log file for detailed errors")
    print("  3. Analyze patterns in incorrect cases")
    print("  4. Consider fine-tuning if accuracy is low")
    print("=" * 80)


if __name__ == "__main__":
    main()

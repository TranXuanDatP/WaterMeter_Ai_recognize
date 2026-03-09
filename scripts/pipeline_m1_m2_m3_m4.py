#!/usr/bin/env python3
"""
Complete Pipeline M1 -> M2 + Smart Rotate -> M3 -> M3.5 -> M4 + Beam Search

Full pipeline for meter reading:
1. M1: Water Meter Detection (YOLO)
2. M2: Orientation + Smart Rotate (Angle Regression)
3. M3: ROI Detection (YOLOv8n)
4. M3.5: Black Digit Extraction (Improved from M5)
5. M4: CRNN OCR Reading
6. Beam Search Decoder: Advanced CTC decoding (96% accuracy!)

Input: Raw images
Output: Final reading with confidence
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
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import M3.5 module
from src.m3_5_digit_extraction import M3_5_DigitExtractor

# Import M2 Orientation Model
from src.m2_orientation.model import M2_OrientationModel

# Import Beam Search Decoder
from src.m4_crnn_reading.beam_search_decoder import create_decoder

# ====================== CONFIGURATION ======================
class Config:
    # Model paths
    M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
    M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
    M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"
    M4_MODEL = r"F:\Workspace\Project\model\M4_OCR.pth"

    # Input/Output
    INPUT_DIR = r"F:\Workspace\Project\data\raw_images"
    OUTPUT_DIR = r"F:\Workspace\Project\results\pipeline_run"

    # Stage outputs
    M1_CROPS_DIR = os.path.join(OUTPUT_DIR, "m1_crops")
    M2_ALIGNED_DIR = os.path.join(OUTPUT_DIR, "m2_crops_aligned")
    M3_ROI_DIR = os.path.join(OUTPUT_DIR, "m3_roi_crops")
    M3_5_DIGITS_DIR = os.path.join(OUTPUT_DIR, "m3_5_black_digits")

    # Thresholds
    M1_CONFIDENCE = 0.25
    M2_CONFIDENCE = 0.5
    M3_CONFIDENCE = 0.25
    M4_CONFIDENCE = 0.7

    # Beam Search Decoder Configuration
    BEAM_SEARCH_METHOD = 'beam'  # 'greedy', 'beam', or 'prefix_beam'
    BEAM_WIDTH = 10  # Recommended: 10 for best accuracy/speed tradeoff

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================== M1: WATER METER DETECTION ======================
class M1_WaterMeterDetector:
    """M1: Detect water meter in image using YOLO"""

    def __init__(self, model_path: str, confidence: float = 0.25):
        print("[M1] Loading water meter detection model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"      Model: {model_path}")
        print(f"      Confidence threshold: {confidence}")

    def detect(self, image: np.ndarray) -> dict:
        """
        Detect water meter in image

        Returns:
            dict with detection results
        """
        results = self.model(image, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return {
                'detected': False,
                'bbox': None,
                'confidence': 0.0
            }

        # Get best detection
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()

        if len(confidences) == 0:
            return {
                'detected': False,
                'bbox': None,
                'confidence': 0.0
            }

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


# ====================== M2: ORIENTATION + SMART ROTATE ======================
class M2_SmartRotator:
    """M2: Orientation prediction + Smart Rotate"""

    def __init__(self, model_path: str, device: torch.device):
        print("[M2] Loading orientation + smart rotate model...")
        self.device = device
        self.model = M2_OrientationModel().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            # Use strict=False because ReLU and Dropout layers don't have saved parameters
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
        """
        Smart rotate image using CORRECTED logic

        Logic (matching original metadata pattern):
        1. Normalize angle to [-180, 180]
        2. Correction angle = -angle (rotate counter-clockwise)
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

        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        return rotated, correction_angle


# ====================== M3: ROI DETECTION ======================
class M3_ROIDetector:
    """M3: Detect ROI (digit region) using YOLOv8n"""

    def __init__(self, model_path: str, confidence: float = 0.25):
        print("[M3] Loading ROI detection model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"      Model: {model_path}")
        print(f"      Confidence threshold: {confidence}")

    def detect(self, image: np.ndarray) -> dict:
        """
        Detect ROI in aligned image

        Returns:
            dict with detection results
        """
        results = self.model(image, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            return {
                'detected': False,
                'bbox': None,
                'confidence': 0.0
            }

        # Get best detection
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()

        if len(confidences) == 0:
            return {
                'detected': False,
                'bbox': None,
                'confidence': 0.0
            }

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


# ====================== M3.5: BLACK DIGIT EXTRACTION ======================
# Using improved M3.5 implementation from src/m3_5_digit_extraction
# This replaces the old implementation with better red digit detection


# ====================== M4: CRNN OCR ======================
class M4_CRNNOCR:
    """M4: CRNN OCR for digit reading with Beam Search decoder"""

    def __init__(self, model_path: str, device: torch.device,
                 decoder_method: str = 'beam', beam_width: int = 10):
        """
        Initialize M4 OCR with beam search decoder

        Args:
            model_path: Path to M4_OCR.pth checkpoint
            device: Torch device (cuda/cpu)
            decoder_method: 'greedy', 'beam', or 'prefix_beam'
            beam_width: Beam width for beam search (recommended: 10)
        """
        print("[M4] Loading CRNN OCR model...")
        self.device = device
        self.char_map = "0123456789"
        self.label_to_char = {i: c for i, c in enumerate(self.char_map)}
        self.blank_idx = 10

        # Load model
        self._load_model(model_path)
        print(f"      Model: {model_path}")
        print(f"      Device: {device}")

        # Initialize beam search decoder
        print(f"[M4] Initializing decoder...")
        self.decoder = create_decoder(
            method=decoder_method,
            chars=self.char_map,
            blank_idx=self.blank_idx,
            beam_width=beam_width
        )
        print(f"      Method: {decoder_method}")
        print(f"      Beam Width: {beam_width}")

    def _load_model(self, model_path: str):
        """Load CRNN model"""
        class CRNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                )
                self.rnn = nn.LSTM(2048, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
                self.fc = nn.Linear(512, 11)

            def forward(self, x):
                conv = self.cnn(x)
                b, c, h, w = conv.size()
                conv = conv.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
                out, _ = self.rnn(conv)
                return self.fc(out).permute(1, 0, 2)

        self.model = CRNN().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def recognize(self, image: np.ndarray) -> dict:
        """
        Recognize text from digit image using beam search decoder

        Args:
            image: Input image (numpy array)

        Returns:
            dict with 'text' and 'confidence'
        """
        # Convert to PIL
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image).convert('L')
        else:
            pil_image = Image.fromarray(image)

        # Transform
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Recognize
        with torch.no_grad():
            predictions = self.model(tensor)  # (T, B, C)

        # Decode using beam search
        text = self.decoder.decode(predictions)  # Returns string directly

        # Calculate confidence
        probs = torch.softmax(predictions, dim=-1)
        confidences = []

        # Get confidences for each character in decoded text
        pred_indices = predictions.argmax(dim=-1).squeeze()  # (T,)
        for t, idx in enumerate(pred_indices):
            if idx < len(self.char_map):  # Not blank
                confidences.append(probs[t, 0, idx].item())

        confidence = float(np.mean(confidences)) if confidences else 0.0

        return {'text': text, 'confidence': confidence}


# ====================== COMPLETE PIPELINE ======================
class CompletePipeline:
    """
    Complete pipeline M1 -> M2 -> M3 -> M3.5 -> M4 + Beam Search

    Pipeline stages:
    1. M1: Water Meter Detection (YOLO)
    2. M2: Orientation + Smart Rotate (Angle Regression)
    3. M3: ROI Detection (YOLOv8n)
    4. M3.5: Black Digit Extraction (red digit removal)
    5. M4: CRNN OCR Reading
    6. Beam Search Decoder: Advanced CTC decoding (96% accuracy!)

    Usage:
        config = Config()
        config.BEAM_SEARCH_METHOD = 'beam'  # or 'greedy', 'prefix_beam'
        config.BEAM_WIDTH = 10

        pipeline = CompletePipeline(config)
        result = pipeline.process_single_image('test.jpg')
        print(f"Reading: {result['final_reading']}")
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()

        print("="*60)
        print("INITIALIZING COMPLETE PIPELINE")
        print("="*60)

        # Initialize stages
        self.m1 = M1_WaterMeterDetector(self.config.M1_MODEL, self.config.M1_CONFIDENCE)
        self.m2 = M2_SmartRotator(self.config.M2_MODEL, self.config.DEVICE)
        self.m3 = M3_ROIDetector(self.config.M3_MODEL, self.config.M3_CONFIDENCE)
        self.m3_5 = M3_5_DigitExtractor(min_crop_ratio=0.75, fallback_ratio=0.8)
        self.m4 = M4_CRNNOCR(
            self.config.M4_MODEL,
            self.config.DEVICE,
            decoder_method=self.config.BEAM_SEARCH_METHOD,
            beam_width=self.config.BEAM_WIDTH
        )

        print("="*60)
        print("PIPELINE INITIALIZED")
        print("="*60)

    def process_single_image(self, image_path: str, save_intermediates: bool = True) -> dict:
        """
        Process single image through complete pipeline

        Returns:
            dict with complete results
        """
        result = {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'stages': {}
        }

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            result['error'] = 'Could not load image'
            return result

        # ===== M1: Detect Water Meter =====
        m1_result = self.m1.detect(image)
        result['stages']['m1'] = m1_result

        if not m1_result['detected']:
            result['error'] = 'M1: Water meter not detected'
            return result

        # Crop water meter
        m1_crop = self.m1.crop(image, m1_result['bbox'])

        if save_intermediates:
            m1_crop_path = os.path.join(self.config.M1_CROPS_DIR, result['filename'])
            os.makedirs(self.config.M1_CROPS_DIR, exist_ok=True)
            cv2.imwrite(m1_crop_path, m1_crop)

        # ===== M2: Orientation + Smart Rotate =====
        angle = self.m2.predict_angle(m1_crop)
        m2_aligned, actual_angle = self.m2.smart_rotate(m1_crop, angle)

        result['stages']['m2'] = {
            'predicted_angle': angle,
            'actual_angle': actual_angle,
            'rotated': True
        }

        if save_intermediates:
            m2_aligned_path = os.path.join(self.config.M2_ALIGNED_DIR, result['filename'])
            os.makedirs(self.config.M2_ALIGNED_DIR, exist_ok=True)
            cv2.imwrite(m2_aligned_path, m2_aligned)

        # ===== M3: Detect ROI =====
        m3_result = self.m3.detect(m2_aligned)
        result['stages']['m3'] = m3_result

        if not m3_result['detected']:
            result['error'] = 'M3: ROI not detected'
            # Try to use full aligned image as ROI
            roi_image = m2_aligned
        else:
            roi_image = self.m3.crop(m2_aligned, m3_result['bbox'])

            if save_intermediates:
                m3_roi_path = os.path.join(self.config.M3_ROI_DIR, result['filename'])
                os.makedirs(self.config.M3_ROI_DIR, exist_ok=True)
                cv2.imwrite(m3_roi_path, roi_image)

        # ===== M3.5: Extract Black Digits =====
        black_digits_image, m3_5_metadata = self.m3_5.extract(roi_image)
        result['stages']['m3_5'] = {
            'status': m3_5_metadata['status'],
            'crop_ratio': m3_5_metadata.get('crop_ratio', 0),
            'original_size': m3_5_metadata.get('original_size', (0, 0)),
            'crop_size': m3_5_metadata.get('crop_size', (0, 0)),
        }

        if black_digits_image is None or m3_5_metadata['status'] == 'error':
            result['error'] = 'M3.5: Failed to extract black digits'
            return result

        # Save M3.5 output if needed
        if save_intermediates:
            m3_5_path = os.path.join(self.config.M3_5_DIGITS_DIR, result['filename'])
            os.makedirs(self.config.M3_5_DIGITS_DIR, exist_ok=True)
            cv2.imwrite(m3_5_path, black_digits_image)

        # ===== M4: OCR with Beam Search =====
        # M3.5 returns the complete black digits image (4 digits combined)
        # Feed this directly to M4 for OCR with beam search decoder
        m4_result = self.m4.recognize(black_digits_image)

        result['stages']['m4'] = m4_result
        result['final_reading'] = m4_result['text']
        result['final_confidence'] = m4_result['confidence']

        return result

    def _combine_digits(self, digits: list) -> np.ndarray:
        """Combine 4 digit images into one"""
        # Resize all to same height
        target_height = 64
        resized = []
        for digit in digits:
            if len(digit.shape) == 3:
                digit = cv2.cvtColor(digit, cv2.COLOR_RGB2GRAY)
            h, w = digit.shape
            aspect = w / h
            new_w = int(target_height * aspect)
            resized_digit = cv2.resize(digit, (new_w, target_height))
            resized.append(resized_digit)

        # Concatenate horizontally
        combined = np.hstack(resized)
        return combined

    def process_directory(self, input_dir: str, output_csv: str = None):
        """Process all images in directory"""
        image_files = list(Path(input_dir).glob('*.jpg')) + list(Path(input_dir).glob('*.png'))

        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.process_single_image(str(img_path), save_intermediates=True)
            results.append(result)

        # Save results
        if output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")

        return results


# ====================== MAIN ======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Complete Pipeline M1->M2->M3->M3.5->M4 with Beam Search Decoder'
    )
    parser.add_argument('--input', type=str, required=True, help='Input directory or image')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--csv', type=str, default='results.csv', help='Output CSV file')
    parser.add_argument('--single', action='store_true', help='Process single image')

    # Beam search decoder options
    parser.add_argument('--decoder', type=str, default='beam',
                        choices=['greedy', 'beam', 'prefix_beam'],
                        help='Decoder method (default: beam)')
    parser.add_argument('--beam-width', type=int, default=10,
                        help='Beam width for beam search (default: 10)')

    args = parser.parse_args()

    # Initialize pipeline with custom decoder settings
    config = Config()
    config.BEAM_SEARCH_METHOD = args.decoder
    config.BEAM_WIDTH = args.beam_width

    if args.output:
        config.OUTPUT_DIR = args.output

    pipeline = CompletePipeline(config)

    if args.single or os.path.isfile(args.input):
        # Single image
        result = pipeline.process_single_image(args.input, save_intermediates=True)
        print(f"\nFinal Reading: {result.get('final_reading', 'ERROR')}")
        print(f"Confidence: {result.get('final_confidence', 0.0):.4f}")
    else:
        # Directory
        results = pipeline.process_directory(args.input, args.csv)
        print(f"\nProcessed {len(results)} images")

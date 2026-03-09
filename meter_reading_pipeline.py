#!/usr/bin/env python3
"""
Meter Reading Pipeline: M1 -> M2 + Smart Rotate -> M3 -> M4

M1: Image Preprocessing
    - Denoise, enhance contrast, adjust brightness

M2: Object Detection + Smart Rotation
    - Detect meter region
    - Calculate optimal rotation angle
    - Smart rotate for better OCR

M3: OCR Recognition
    - CRNN + biLSTM + CTC
    - Text recognition

M4: Post-processing
    - Validation
    - Format correction
    - Confidence scoring
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# ====================== M1: IMAGE PREPROCESSING ======================
class M1_ImagePreprocessor:
    """
    M1: Image Preprocessing Module
    - Denoise
    - Contrast enhancement
    - Brightness adjustment
    - Size normalization
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image

        Args:
            image: Input image (BGR or RGB)

        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Denoise
        image = self._denoise(image)

        # Enhance contrast
        image = self._enhance_contrast(image)

        # Adjust brightness
        image = self._adjust_brightness(image)

        # Resize
        image = self._resize(image)

        return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        # Use bilateral filter for edge-preserving denoising
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return denoised

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

        return enhanced

    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Adjust image brightness"""
        # Convert to HSV if color image
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            # Increase brightness by 20%
            v = cv2.add(v, 50)

            # Clip to valid range
            v = np.clip(v, 0, 255).astype('uint8')

            adjusted = cv2.merge([h, s, v])
            adjusted = cv2.cvtColor(adjusted, cv2.COLOR_HSV2RGB)
        else:
            # For grayscale, just add constant
            adjusted = cv2.add(image, 50)
            adjusted = np.clip(adjusted, 0, 255).astype('uint8')

        return adjusted

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

    def visualize_steps(self, original: np.ndarray, preprocessed: np.ndarray, save_path: Optional[str] = None):
        """Visualize preprocessing results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(preprocessed)
        axes[1].set_title('Preprocessed Image')
        axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()


# ====================== M2: OBJECT DETECTION + SMART ROTATION ======================
class M2_SmartRotator:
    """
    M2: Object Detection + Smart Rotation Module
    - Detect meter/reading region
    - Calculate optimal rotation angle
    - Smart rotate to upright position
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect meter region and rotate to upright position

        Args:
            image: Input image

        Returns:
            Tuple of (rotated_image, rotation_angle)
        """
        # Detect meter region
        meter_region = self._detect_meter_region(image)

        # Calculate rotation angle
        angle = self._calculate_rotation_angle(meter_region)

        # Rotate image
        rotated = self._rotate_image(image, angle)

        if self.debug:
            self._visualize_detection(image, meter_region, rotated, angle)

        return rotated, angle

    def _detect_meter_region(self, image: np.ndarray) -> np.ndarray:
        """
        Detect meter region in image
        Using edge detection and contour finding
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image  # Return original if no contours found

        # Find largest rectangular contour (likely the meter)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract meter region with some padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        meter_region = image[y1:y2, x1:x2]

        return meter_region

    def _calculate_rotation_angle(self, image: np.ndarray) -> float:
        """
        Calculate optimal rotation angle using text line detection

        Returns:
            Rotation angle in degrees (positive = clockwise)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect text lines using morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find contours of text lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0  # No rotation needed if no text detected

        # Calculate angle using PCA on all contour points
        all_points = np.vstack([cnt for cnt in contours])

        # PCA
        mean, eigenvectors = cv2.PCACompute(all_points.astype(np.float32), mean=None)

        # Calculate angle from principal component
        angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))

        # Adjust angle to be in [-45, 45] range
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        return angle

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle

        Args:
            image: Input image
            angle: Rotation angle in degrees

        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Rotate image
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255) if len(image.shape) == 3 else 255)

        return rotated

    def _visualize_detection(self, original: np.ndarray, region: np.ndarray,
                           rotated: np.ndarray, angle: float, save_path: Optional[str] = None):
        """Visualize detection and rotation results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(region)
        axes[1].set_title('Detected Meter Region')
        axes[1].axis('off')

        axes[2].imshow(rotated)
        axes[2].set_title(f'Rotated ({angle:.2f}°)')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detection visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()


# ====================== M3: OCR RECOGNITION ======================
class M3_OCRRecognizer:
    """
    M3: OCR Recognition Module
    - Load trained CRNN model
    - Recognize text from image
    - Return raw text with confidence
    """

    def __init__(self, model_path: str, device: torch.device,
                 img_height: int = 64, img_width: int = 256):
        self.model_path = model_path
        self.device = device
        self.img_height = img_height
        self.img_width = img_width

        # Character mappings
        self.char_map = "0123456789"
        self.label_to_char = {i: c for i, c in enumerate(self.char_map)}
        self.char_to_label = {c: i for i, c in enumerate(self.char_map)}
        self.blank_idx = 10

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _load_model(self) -> nn.Module:
        """Load trained CRNN model"""
        # Define model architecture (same as training)
        class CRNN(nn.Module):
            def __init__(self, num_classes=11, num_channels=1):
                super(CRNN, self).__init__()

                self.cnn = nn.Sequential(
                    nn.Conv2d(num_channels, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                )

                self.rnn_input_size = 512 * 2
                self.hidden_size = 256

                self.rnn = nn.LSTM(
                    self.rnn_input_size,
                    self.hidden_size,
                    num_layers=2,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0.3
                )

                self.fc = nn.Linear(self.hidden_size * 2, 11)

            def forward(self, x):
                conv = self.cnn(x)
                b, c, h, w = conv.size()
                conv = conv.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
                rnn_out, _ = self.rnn(conv)
                out = self.fc(rnn_out).permute(1, 0, 2)
                return out

        # Load model
        model = CRNN(num_classes=11, num_channels=1).to(self.device)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from: {self.model_path}")
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
        else:
            print(f"⚠️  Model file not found: {self.model_path}")
            print(f"   Using initialized model (not trained)")

        return model

    def recognize(self, image: np.ndarray) -> Dict:
        """
        Recognize text from image

        Args:
            image: Input image (RGB or grayscale)

        Returns:
            Dictionary with text and confidence
        """
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image).convert('L')
        else:
            pil_image = Image.fromarray(image)

        # Transform
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Recognize
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Decode
        text, confidence = self._decode_predictions(predictions)

        return {
            'text': text,
            'confidence': confidence
        }

    def _decode_predictions(self, predictions: torch.Tensor) -> Tuple[str, float]:
        """
        Decode CTC predictions to text

        Returns:
            Tuple of (text, confidence)
        """
        # Get probabilities
        probs = predictions.softmax(dim=-1)  # (T, 1, C)

        # Get max probability indices
        pred_indices = predictions.argmax(dim=-1)  # (T, 1)
        pred_indices = pred_indices.permute(1, 0).cpu().numpy()[0]  # (T,)

        # Decode with CTC rules
        chars = []
        prev_char = None
        confidences = []

        for t, idx in enumerate(pred_indices):
            if idx != self.blank_idx and idx != prev_char:
                char = self.label_to_char[idx]
                chars.append(char)
                # Get confidence for this character
                conf = probs[t, 0, idx].item()
                confidences.append(conf)
            prev_char = idx

        text = ''.join(chars)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return text, avg_confidence


# ====================== M4: POST-PROCESSING ======================
class M4_PostProcessor:
    """
    M4: Post-processing Module
    - Validate reading
    - Format correction
    - Confidence scoring
    - Error detection
    """

    def __init__(self, expected_length: int = 4):
        self.expected_length = expected_length

    def process(self, ocr_result: Dict, metadata: Optional[Dict] = None) -> Dict:
        """
        Post-process OCR result

        Args:
            ocr_result: OCR result from M3
            metadata: Optional metadata from previous modules

        Returns:
            Post-processed result
        """
        text = ocr_result['text']
        raw_confidence = ocr_result['confidence']

        # Validate
        validation = self._validate_reading(text)

        # Format correction
        corrected_text = self._correct_format(text)

        # Calculate final confidence
        final_confidence = self._calculate_final_confidence(
            raw_confidence, validation, corrected_text != text
        )

        result = {
            'text': corrected_text,
            'raw_text': text,
            'confidence': final_confidence,
            'raw_confidence': raw_confidence,
            'is_valid': validation['valid'],
            'validation_errors': validation['errors'],
            'metadata': metadata or {}
        }

        return result

    def _validate_reading(self, text: str) -> Dict:
        """
        Validate reading text

        Returns:
            Dictionary with validation status and errors
        """
        errors = []

        # Check length
        if len(text) != self.expected_length:
            errors.append(f"Expected {self.expected_length} digits, got {len(text)}")

        # Check if all digits
        if not text.isdigit():
            errors.append("Contains non-digit characters")

        # Check for common OCR errors
        common_errors = {
            'O': '0',  # Letter O vs Number 0
            'I': '1',  # Letter I vs Number 1
            'l': '1',  # Letter l vs Number 1
            'S': '5',  # Letter S vs Number 5
        }

        for wrong, correct in common_errors.items():
            if wrong in text:
                errors.append(f"Possible OCR error: '{wrong}' should be '{correct}'")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def _correct_format(self, text: str) -> str:
        """
        Correct common OCR format errors

        Returns:
            Corrected text
        """
        corrected = text

        # Common OCR corrections
        corrections = {
            'O': '0',
            'o': '0',
            'I': '1',
            'l': '1',
            'S': '5',
            's': '5',
            'Z': '2',
            'z': '2',
            'B': '8',
        }

        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)

        # Remove non-digits
        corrected = ''.join(c for c in corrected if c.isdigit())

        # Pad or truncate to expected length
        if len(corrected) < self.expected_length:
            corrected = corrected.zfill(self.expected_length)
        elif len(corrected) > self.expected_length:
            corrected = corrected[:self.expected_length]

        return corrected

    def _calculate_final_confidence(self, raw_confidence: float,
                                   validation: Dict,
                                   was_corrected: bool) -> float:
        """
        Calculate final confidence score

        Args:
            raw_confidence: Raw OCR confidence
            validation: Validation result
            was_corrected: Whether text was corrected

        Returns:
            Final confidence score (0-1)
        """
        confidence = raw_confidence

        # Reduce confidence if validation failed
        if not validation['valid']:
            confidence *= 0.8

        # Reduce confidence if corrections were made
        if was_corrected:
            confidence *= 0.9

        # Reduce confidence for each error
        confidence *= (0.95 ** len(validation['errors']))

        return max(0.0, min(1.0, confidence))


# ====================== COMPLETE PIPELINE ======================
class MeterReadingPipeline:
    """
    Complete Meter Reading Pipeline
    M1 -> M2 + Smart Rotate -> M3 -> M4
    """

    def __init__(self, model_path: str, device: torch.device,
                 debug: bool = False, save_intermediates: bool = False):
        """
        Initialize pipeline

        Args:
            model_path: Path to trained OCR model
            device: Torch device
            debug: Enable debug visualization
            save_intermediates: Save intermediate results
        """
        self.debug = debug
        self.save_intermediates = save_intermediates

        # Initialize modules
        print("🔧 Initializing Meter Reading Pipeline...")
        print("="*60)

        print("M1: Image Preprocessing")
        self.m1 = M1_ImagePreprocessor()

        print("M2: Object Detection + Smart Rotation")
        self.m2 = M2_SmartRotator(debug=debug)

        print("M3: OCR Recognition")
        self.m3 = M3_OCRRecognizer(model_path, device)

        print("M4: Post-processing")
        self.m4 = M4_PostProcessor()

        print("="*60)
        print("✅ Pipeline initialized successfully!")

    def process(self, image_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Process image through complete pipeline

        Args:
            image_path: Path to input image
            output_dir: Optional directory to save intermediate results

        Returns:
            Final result with all intermediate data
        """
        print("\n" + "="*60)
        print("🚀 PROCESSING IMAGE")
        print("="*60)
        print(f"Input: {image_path}")

        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = {
            'input_image': image_path,
            'stages': {}
        }

        # ===== M1: Image Preprocessing =====
        print("\n[M1] Image Preprocessing...")
        preprocessed = self.m1.preprocess(image)
        result['stages']['m1_preprocessed'] = preprocessed

        if self.save_intermediates and output_dir:
            save_path = os.path.join(output_dir, '1_preprocessed.png')
            cv2.imwrite(save_path, cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))
            self.m1.visualize_steps(image, preprocessed,
                                    os.path.join(output_dir, '1_preprocessing_comparison.png'))

        # ===== M2: Object Detection + Smart Rotation =====
        print("[M2] Object Detection + Smart Rotation...")
        rotated, rotation_angle = self.m2.process(preprocessed)
        result['stages']['m2_rotated'] = rotated
        result['stages']['rotation_angle'] = rotation_angle

        if self.save_intermediates and output_dir:
            save_path = os.path.join(output_dir, '2_rotated.png')
            cv2.imwrite(save_path, cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
            self.m2._visualize_detection(preprocessed, preprocessed, rotated, rotation_angle,
                                        os.path.join(output_dir, '2_rotation_comparison.png'))

        # ===== M3: OCR Recognition =====
        print("[M3] OCR Recognition...")
        ocr_result = self.m3.recognize(rotated)
        result['stages']['m3_ocr'] = ocr_result

        print(f"    Raw text: {ocr_result['text']}")
        print(f"    Confidence: {ocr_result['confidence']:.4f}")

        # ===== M4: Post-processing =====
        print("[M4] Post-processing...")
        metadata = {
            'rotation_angle': rotation_angle,
            'image_size': image.shape[:2]
        }
        final_result = self.m4.process(ocr_result, metadata)

        result['final'] = final_result
        result['stages']['m4_postprocessed'] = final_result

        print(f"    Final text: {final_result['text']}")
        print(f"    Final confidence: {final_result['confidence']:.4f}")
        print(f"    Valid: {final_result['is_valid']}")

        if not final_result['is_valid']:
            print(f"    Errors: {final_result['validation_errors']}")

        # Save result
        if output_dir:
            result_path = os.path.join(output_dir, 'result.json')
            with open(result_path, 'w') as f:
                # Convert numpy arrays for JSON serialization
                save_result = self._prepare_for_save(result)
                json.dump(save_result, f, indent=2)
            print(f"\n💾 Result saved to: {result_path}")

        print("="*60)
        print("✅ PROCESSING COMPLETE!")
        print("="*60)

        return result

    def _prepare_for_save(self, result: Dict) -> Dict:
        """Prepare result for JSON serialization"""
        save_result = {
            'input_image': result['input_image'],
            'final': {
                'text': result['final']['text'],
                'raw_text': result['final']['raw_text'],
                'confidence': result['final']['confidence'],
                'raw_confidence': result['final']['raw_confidence'],
                'is_valid': result['final']['is_valid'],
                'validation_errors': result['final']['validation_errors'],
                'metadata': result['final']['metadata']
            },
            'stages': {
                'rotation_angle': result['stages']['rotation_angle'],
                'm3_ocr': result['stages']['m3_ocr']
            }
        }
        return save_result


# ====================== MAIN ======================
if __name__ == "__main__":
    """
    Example usage of the pipeline
    """
    import argparse

    parser = argparse.ArgumentParser(description='Meter Reading Pipeline')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained OCR model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug visualization')
    parser.add_argument('--save-intermediates', action='store_true',
                       help='Save intermediate processing results')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize pipeline
    pipeline = MeterReadingPipeline(
        model_path=args.model,
        device=device,
        debug=args.debug,
        save_intermediates=args.save_intermediates
    )

    # Process image
    result = pipeline.process(
        image_path=args.image,
        output_dir=args.output
    )

    # Print final result
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Text: {result['final']['text']}")
    print(f"Raw Text: {result['final']['raw_text']}")
    print(f"Confidence: {result['final']['confidence']:.4f}")
    print(f"Valid: {result['final']['is_valid']}")
    print(f"Rotation Angle: {result['stages']['rotation_angle']:.2f}°")

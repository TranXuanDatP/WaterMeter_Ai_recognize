#!/usr/bin/env python3
"""
Flexible Meter Reading Pipeline - Load 4 Separate Models

M1 -> M2 + Smart Rotate -> M3 -> M4

Support for loading custom models at each stage.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# ====================== M1: CUSTOM PREPROCESSING MODEL ======================
class M1_CustomPreprocessor:
    """
    M1: Custom Preprocessing Model
    Supports: PyTorch models, ONNX models, or custom functions
    """

    def __init__(self, model_path: Optional[str] = None, model_type: str = "pytorch",
                 custom_function: Optional[callable] = None):
        """
        Args:
            model_path: Path to model weights
            model_type: 'pytorch', 'onnx', 'custom', 'opencv'
            custom_function: Custom preprocessing function
        """
        self.model_path = model_path
        self.model_type = model_type
        self.custom_function = custom_function
        self.model = None

        if model_path and model_type != "custom":
            self._load_model()

    def _load_model(self):
        """Load model based on type"""
        if self.model_type == "pytorch":
            self.model = torch.load(self.model_path, map_location='cpu')
            if isinstance(self.model, dict) and 'model_state_dict' in self.model:
                # Handle checkpoint format
                # You need to define your model architecture here
                pass

        elif self.model_type == "onnx":
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)

        elif self.model_type == "opencv":
            # OpenCV DNN model
            self.model = cv2.dnn.readNet(self.model_path)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing"""
        if self.custom_function:
            return self.custom_function(image)

        if self.model is None:
            # Default preprocessing if no model
            return self._default_preprocess(image)

        # Apply custom model
        return self._apply_model(image)

    def _default_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Default: Basic enhancement"""
        # Denoise
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

        # Enhance contrast
        if len(image.shape) == 3:
            lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

        return enhanced

    def _apply_model(self, image: np.ndarray) -> np.ndarray:
        """Apply loaded model"""
        if self.model_type == "pytorch":
            with torch.no_grad():
                # Convert to tensor
                tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0)

                # Apply model
                output = self.model(tensor)

                # Convert back to numpy
                output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output = (output * 255).astype(np.uint8)

                return output

        elif self.model_type == "onnx":
            # Prepare input
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: image.astype(np.float32)})
            return output[0]

        return image


# ====================== M2: CUSTOM DETECTION + ROTATION MODEL ======================
class M2_CustomDetector:
    """
    M2: Custom Detection + Smart Rotation Model
    Supports: YOLO, SSD, Custom detection models
    """

    def __init__(self, model_path: Optional[str] = None, model_type: str = "custom",
                 detection_function: Optional[callable] = None,
                 rotation_function: Optional[callable] = None):
        """
        Args:
            model_path: Path to detection/rotation model
            model_type: 'yolo', 'ssd', 'dnn', 'custom'
            detection_function: Custom detection function
            rotation_function: Custom rotation function
        """
        self.model_path = model_path
        self.model_type = model_type
        self.detection_function = detection_function
        self.rotation_function = rotation_function
        self.model = None

        if model_path and model_type != "custom":
            self._load_model()

    def _load_model(self):
        """Load detection model"""
        if self.model_type == "yolo":
            # Load YOLO model
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

        elif self.model_type == "dnn":
            # OpenCV DNN
            self.model = cv2.dnn.readNet(self.model_path)

        elif self.model_type == "pytorch":
            # PyTorch detection model
            self.model = torch.load(self.model_path, map_location='cpu')

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        Detect and rotate

        Returns:
            Tuple of (rotated_image, rotation_angle, metadata)
        """
        metadata = {}

        # Detection
        if self.detection_function:
            detected_region, detection_meta = self.detection_function(image)
            metadata['detection'] = detection_meta
        else:
            detected_region = self._default_detection(image)
            metadata['detection'] = {}

        # Rotation
        if self.rotation_function:
            rotated, angle, rotation_meta = self.rotation_function(detected_region)
            metadata['rotation'] = rotation_meta
        else:
            angle = self._calculate_rotation_angle(detected_region)
            rotated = self._rotate_image(detected_region, angle)
            metadata['rotation'] = {'angle': angle, 'method': 'pca'}

        return rotated, angle, metadata

    def _default_detection(self, image: np.ndarray) -> np.ndarray:
        """Default detection: Find largest rectangular region"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        return image[y1:y2, x1:x2]

    def _calculate_rotation_angle(self, image: np.ndarray) -> float:
        """Calculate rotation angle using PCA"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        all_points = np.vstack([cnt for cnt in contours])
        mean, eigenvectors = cv2.PCACompute(all_points.astype(np.float32), mean=None)

        angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))

        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        return angle

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        border_color = (255, 255, 255) if len(image.shape) == 3 else 255
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_color)
        return rotated


# ====================== M3: CUSTOM OCR MODEL ======================
class M3_CustomOCR:
    """
    M3: Custom OCR Model
    Supports: CRNN, PaddleOCR, EasyOCR, Tesseract, Custom
    """

    def __init__(self, model_path: Optional[str] = None, model_type: str = "pytorch",
                 ocr_function: Optional[callable] = None,
                 char_map: str = "0123456789"):
        """
        Args:
            model_path: Path to OCR model
            model_type: 'pytorch', 'paddleocr', 'easyocr', 'tesseract', 'custom'
            ocr_function: Custom OCR function
            char_map: Character set for decoding
        """
        self.model_path = model_path
        self.model_type = model_type
        self.ocr_function = ocr_function
        self.char_map = char_map
        self.label_to_char = {i: c for i, c in enumerate(char_map)}
        self.blank_idx = len(char_map)
        self.model = None

        if model_path and model_type != "custom":
            self._load_model()

    def _load_model(self):
        """Load OCR model"""
        if self.model_type == "pytorch":
            # Load PyTorch model
            checkpoint = torch.load(self.model_path, map_location='cpu')
            # You need to define your CRNN model architecture here
            # self.model = YourCRNNModel()
            # self.model.load_state_dict(checkpoint['model_state_dict'])

        elif self.model_type == "paddleocr":
            from paddleocr import PaddleOCR
            self.model = PaddleOCR(use_angle_char=True, lang='en')

        elif self.model_type == "easyocr":
            import easyocr
            self.model = easyocr.Reader(['en'])

        elif self.model_type == "tesseract":
            import pytesseract
            self.model = pytesseract

    def recognize(self, image: np.ndarray) -> Dict:
        """
        Recognize text from image

        Returns:
            Dictionary with text and confidence
        """
        if self.ocr_function:
            return self.ocr_function(image)

        if self.model is None:
            # Default: return placeholder
            return {
                'text': '',
                'confidence': 0.0,
                'raw_output': None
            }

        # Apply model-specific recognition
        if self.model_type == "pytorch":
            return self._pytorch_ocr(image)
        elif self.model_type == "paddleocr":
            return self._paddleocr_ocr(image)
        elif self.model_type == "easyocr":
            return self._easyocr_ocr(image)
        elif self.model_type == "tesseract":
            return self._tesseract_ocr(image)

        return {'text': '', 'confidence': 0.0}

    def _pytorch_ocr(self, image: np.ndarray) -> Dict:
        """PyTorch OCR recognition"""
        # Convert to PIL and transform
        pil_image = Image.fromarray(image).convert('L')
        transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        image_tensor = transform(pil_image).unsqueeze(0)

        # Recognize (you need to have model loaded)
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Decode predictions
        text, confidence = self._decode_predictions(predictions)

        return {
            'text': text,
            'confidence': confidence,
            'raw_output': predictions.cpu().numpy()
        }

    def _decode_predictions(self, predictions: torch.Tensor) -> Tuple[str, float]:
        """Decode CTC predictions"""
        probs = predictions.softmax(dim=-1)
        pred_indices = predictions.argmax(dim=-1).squeeze().cpu().numpy()

        chars = []
        confidences = []
        prev = None

        for t, idx in enumerate(pred_indices):
            if idx != self.blank_idx and idx != prev:
                chars.append(self.label_to_char[idx])
                confidences.append(probs[t, 0, idx].item())
            prev = idx

        text = ''.join(chars)
        confidence = np.mean(confidences) if confidences else 0.0

        return text, confidence

    def _paddleocr_ocr(self, image: np.ndarray) -> Dict:
        """PaddleOCR recognition"""
        result = self.model.ocr(image, cls=True)

        if result and result[0]:
            text = ' '.join([line[1][0] for line in result[0]])
            confidences = [line[1][1] for line in result[0]]
            confidence = np.mean(confidences) if confidences else 0.0

            return {
                'text': text,
                'confidence': confidence,
                'raw_output': result
            }

        return {'text': '', 'confidence': 0.0}

    def _easyocr_ocr(self, image: np.ndarray) -> Dict:
        """EasyOCR recognition"""
        results = self.model.readtext(image)

        if results:
            text = ' '.join([result[1] for result in results])
            confidences = [result[2] for result in results]
            confidence = np.mean(confidences) if confidences else 0.0

            return {
                'text': text,
                'confidence': confidence,
                'raw_output': results
            }

        return {'text': '', 'confidence': 0.0}

    def _tesseract_ocr(self, image: np.ndarray) -> Dict:
        """Tesseract OCR recognition"""
        # Configure Tesseract
        custom_config = r'--oem 3 --psm 6'

        # Get text and confidence
        text = self.model.image_to_string(image, config=custom_config)
        text = text.strip()

        # Get detailed data for confidence
        data = self.model.image_to_data(image, config=custom_config, output_type=dict.DICT)
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        confidence = np.mean(confidences) / 100.0 if confidences else 0.0

        return {
            'text': text,
            'confidence': confidence,
            'raw_output': data
        }


# ====================== M4: CUSTOM POST-PROCESSING MODEL ======================
class M4_CustomPostProcessor:
    """
    M4: Custom Post-processing Model
    Supports: Validation models, Correction models, Custom logic
    """

    def __init__(self, model_path: Optional[str] = None, model_type: str = "rule_based",
                 validation_function: Optional[callable] = None,
                 correction_function: Optional[callable] = None):
        """
        Args:
            model_path: Path to post-processing model
            model_type: 'rule_based', 'ml', 'transformer', 'custom'
            validation_function: Custom validation function
            correction_function: Custom correction function
        """
        self.model_path = model_path
        self.model_type = model_type
        self.validation_function = validation_function
        self.correction_function = correction_function
        self.model = None

        if model_path and model_type != "custom":
            self._load_model()

    def _load_model(self):
        """Load post-processing model"""
        if self.model_type == "ml":
            # ML-based validation/correction
            self.model = torch.load(self.model_path, map_location='cpu')

        elif self.model_type == "transformer":
            # Transformer-based correction
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def process(self, ocr_result: Dict, metadata: Optional[Dict] = None) -> Dict:
        """
        Post-process OCR result

        Returns:
            Final processed result
        """
        # Validation
        if self.validation_function:
            validation_result = self.validation_function(ocr_result)
        else:
            validation_result = self._default_validation(ocr_result)

        # Correction
        if self.correction_function:
            corrected_text = self.correction_function(ocr_result['text'])
        else:
            corrected_text = self._default_correction(ocr_result['text'])

        # Calculate final confidence
        final_confidence = self._calculate_confidence(
            ocr_result['confidence'],
            validation_result,
            corrected_text != ocr_result['text']
        )

        return {
            'text': corrected_text,
            'raw_text': ocr_result['text'],
            'confidence': final_confidence,
            'raw_confidence': ocr_result['confidence'],
            'is_valid': validation_result['valid'],
            'errors': validation_result['errors'],
            'metadata': metadata or {}
        }

    def _default_validation(self, ocr_result: Dict) -> Dict:
        """Default validation logic"""
        text = ocr_result['text']
        errors = []

        # Expected length
        if len(text) != 4:
            errors.append(f"Expected 4 digits, got {len(text)}")

        # All digits
        if not text.isdigit():
            errors.append("Contains non-digit characters")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def _default_correction(self, text: str) -> str:
        """Default correction logic"""
        corrected = text

        # Common OCR corrections
        corrections = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1',
            'S': '5', 's': '5',
            'Z': '2', 'z': '2',
            'B': '8'
        }

        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)

        # Keep only digits
        corrected = ''.join(c for c in corrected if c.isdigit())

        # Pad to 4 digits
        if len(corrected) < 4:
            corrected = corrected.zfill(4)
        elif len(corrected) > 4:
            corrected = corrected[:4]

        return corrected

    def _calculate_confidence(self, raw_confidence: float,
                            validation: Dict, was_corrected: bool) -> float:
        """Calculate final confidence"""
        confidence = raw_confidence

        if not validation['valid']:
            confidence *= 0.8

        if was_corrected:
            confidence *= 0.9

        confidence *= (0.95 ** len(validation['errors']))

        return max(0.0, min(1.0, confidence))


# ====================== FLEXIBLE PIPELINE ======================
class FlexibleMeterPipeline:
    """
    Flexible Meter Reading Pipeline
    M1 -> M2 + Smart Rotate -> M3 -> M4

    Supports loading custom models at each stage.
    """

    def __init__(self,
                 m1_model: Optional[str] = None, m1_type: str = "custom",
                 m2_model: Optional[str] = None, m2_type: str = "custom",
                 m3_model: Optional[str] = None, m3_type: str = "pytorch",
                 m4_model: Optional[str] = None, m4_type: str = "rule_based",

                 # Custom functions (optional)
                 m1_function: Optional[callable] = None,
                 m2_detection_function: Optional[callable] = None,
                 m2_rotation_function: Optional[callable] = None,
                 m3_ocr_function: Optional[callable] = None,
                 m4_validation_function: Optional[callable] = None,
                 m4_correction_function: Optional[callable] = None,

                 device: torch.device = torch.device('cpu'),
                 debug: bool = False):
        """
        Initialize flexible pipeline with custom models

        Args:
            m1_model: Path to M1 model (preprocessing)
            m1_type: Type of M1 model
            m1_function: Custom M1 function

            m2_model: Path to M2 model (detection + rotation)
            m2_type: Type of M2 model
            m2_detection_function: Custom detection function
            m2_rotation_function: Custom rotation function

            m3_model: Path to M3 model (OCR)
            m3_type: Type of M3 model
            m3_ocr_function: Custom OCR function

            m4_model: Path to M4 model (post-processing)
            m4_type: Type of M4 model
            m4_validation_function: Custom validation function
            m4_correction_function: Custom correction function

            device: Torch device
            debug: Enable debug mode
        """
        self.device = device
        self.debug = debug

        print("🔧 Initializing Flexible Meter Reading Pipeline...")
        print("="*60)

        # Initialize M1
        print("[M1] Loading Preprocessing Model...")
        self.m1 = M1_CustomPreprocessor(
            model_path=m1_model,
            model_type=m1_type,
            custom_function=m1_function
        )
        print(f"      Type: {m1_type}")
        if m1_model:
            print(f"      Model: {m1_model}")
        if m1_function:
            print(f"      Using custom function")

        # Initialize M2
        print("\n[M2] Loading Detection + Rotation Model...")
        self.m2 = M2_CustomDetector(
            model_path=m2_model,
            model_type=m2_type,
            detection_function=m2_detection_function,
            rotation_function=m2_rotation_function
        )
        print(f"      Type: {m2_type}")
        if m2_model:
            print(f"      Model: {m2_model}")

        # Initialize M3
        print("\n[M3] Loading OCR Model...")
        self.m3 = M3_CustomOCR(
            model_path=m3_model,
            model_type=m3_type,
            ocr_function=m3_ocr_function
        )
        print(f"      Type: {m3_type}")
        if m3_model:
            print(f"      Model: {m3_model}")

        # Initialize M4
        print("\n[M4] Loading Post-processing Model...")
        self.m4 = M4_CustomPostProcessor(
            model_path=m4_model,
            model_type=m4_type,
            validation_function=m4_validation_function,
            correction_function=m4_correction_function
        )
        print(f"      Type: {m4_type}")
        if m4_model:
            print(f"      Model: {m4_model}")

        print("="*60)
        print("✅ Pipeline initialized successfully!")

    def process(self, image_path: str, save_dir: Optional[str] = None) -> Dict:
        """
        Process image through pipeline

        Args:
            image_path: Path to input image
            save_dir: Optional directory to save results

        Returns:
            Complete processing result
        """
        print("\n" + "="*60)
        print("🚀 PROCESSING IMAGE")
        print("="*60)
        print(f"Input: {image_path}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = {
            'input_image': image_path,
            'stages': {}
        }

        # ===== M1: Preprocessing =====
        print("\n[M1] Preprocessing...")
        preprocessed = self.m1.preprocess(image)
        result['stages']['m1'] = {
            'image': preprocessed,
            'model': self.m1.model_type
        }

        # ===== M2: Detection + Rotation =====
        print("[M2] Detection + Rotation...")
        rotated, angle, m2_meta = self.m2.process(preprocessed)
        result['stages']['m2'] = {
            'image': rotated,
            'angle': angle,
            'metadata': m2_meta,
            'model': self.m2.model_type
        }
        print(f"    Rotation angle: {angle:.2f}°")

        # ===== M3: OCR =====
        print("[M3] OCR Recognition...")
        ocr_result = self.m3.recognize(rotated)
        result['stages']['m3'] = {
            **ocr_result,
            'model': self.m3.model_type
        }
        print(f"    Raw text: {ocr_result['text']}")
        print(f"    Confidence: {ocr_result['confidence']:.4f}")

        # ===== M4: Post-processing =====
        print("[M4] Post-processing...")
        final_result = self.m4.process(ocr_result, metadata={'rotation_angle': angle})
        result['final'] = final_result
        result['stages']['m4'] = {
            **final_result,
            'model': self.m4.model_type
        }
        print(f"    Final text: {final_result['text']}")
        print(f"    Final confidence: {final_result['confidence']:.4f}")
        print(f"    Valid: {final_result['is_valid']}")

        # Save results
        if save_dir:
            self._save_results(result, save_dir)

        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE!")
        print("="*60)

        return result

    def _save_results(self, result: Dict, save_dir: str):
        """Save processing results"""
        # Save JSON
        json_path = os.path.join(save_dir, 'result.json')
        save_result = self._prepare_for_save(result)
        with open(json_path, 'w') as f:
            json.dump(save_result, f, indent=2)
        print(f"\n💾 Result saved: {json_path}")

        # Save visualization
        if self.debug:
            viz_path = os.path.join(save_dir, 'pipeline_visualization.png')
            self._visualize_result(result, viz_path)
            print(f"💾 Visualization saved: {viz_path}")

    def _prepare_for_save(self, result: Dict) -> Dict:
        """Prepare result for JSON serialization"""
        return {
            'input_image': result['input_image'],
            'final': {
                'text': result['final']['text'],
                'raw_text': result['final']['raw_text'],
                'confidence': result['final']['confidence'],
                'is_valid': result['final']['is_valid'],
                'errors': result['final']['errors'],
                'metadata': result['final']['metadata']
            },
            'stages': {
                'm1_model': result['stages']['m1']['model'],
                'm2_model': result['stages']['m2']['model'],
                'm2_angle': result['stages']['m2']['angle'],
                'm3_model': result['stages']['m3']['model'],
                'm3_text': result['stages']['m3']['text'],
                'm3_confidence': result['stages']['m3']['confidence'],
                'm4_model': result['stages']['m4']['model']
            }
        }

    def _visualize_result(self, result: Dict, save_path: str):
        """Create visualization of pipeline stages"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # M1
        axes[0, 0].imshow(result['stages']['m1']['image'])
        axes[0, 0].set_title(f"M1: Preprocessed ({result['stages']['m1']['model']})")
        axes[0, 0].axis('off')

        # M2
        axes[0, 1].imshow(result['stages']['m2']['image'])
        axes[0, 1].set_title(f"M2: Rotated {result['stages']['m2']['angle']:.1f}°")
        axes[0, 1].axis('off')

        # M3
        axes[0, 2].axis('off')
        axes[0, 2].text(0.5, 0.5, f"M3 Raw:\n{result['stages']['m3']['text']}\nConf: {result['stages']['m3']['confidence']:.3f}",
                       ha='center', va='center', fontsize=14)

        # M4
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, f"M4 Final:\n{result['final']['text']}\nConf: {result['final']['confidence']:.3f}",
                       ha='center', va='center', fontsize=16, weight='bold', color='green')

        # Status
        axes[1, 1].axis('off')
        status = "✅ VALID" if result['final']['is_valid'] else "❌ INVALID"
        axes[1, 1].text(0.5, 0.6, status, ha='center', va='center', fontsize=18, weight='bold')
        if result['final']['errors']:
            axes[1, 1].text(0.5, 0.3, '\n'.join(result['final']['errors'][:3]),
                           ha='center', va='center', fontsize=9, color='red')

        # Model info
        axes[1, 2].axis('off')
        info_text = f"Models:\n"
        info_text += f"M1: {result['stages']['m1']['model']}\n"
        info_text += f"M2: {result['stages']['m2']['model']}\n"
        info_text += f"M3: {result['stages']['m3']['model']}\n"
        info_text += f"M4: {result['stages']['m4']['model']}"
        axes[1, 2].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10,
                       family='monospace', verticalalignment='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# ====================== EXAMPLE USAGE ======================
if __name__ == "__main__":
    """
    Example: Initialize pipeline with your 4 models
    """

    # ===== OPTION 1: Load all 4 models =====
    pipeline = FlexibleMeterPipeline(
        # M1: Preprocessing model
        m1_model="path/to/m1_model.pth",
        m1_type="pytorch",

        # M2: Detection + Rotation model
        m2_model="path/to/m2_model.pth",
        m2_type="yolo",

        # M3: OCR model
        m3_model="path/to/m3_model.pth",
        m3_type="pytorch",

        # M4: Post-processing model
        m4_model="path/to/m4_model.pth",
        m4_type="ml",

        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        debug=True
    )

    # ===== OPTION 2: Use custom functions =====
    """
    pipeline = FlexibleMeterPipeline(
        # Custom functions instead of models
        m1_function=my_custom_preprocessing,
        m2_detection_function=my_custom_detection,
        m2_rotation_function=my_custom_rotation,
        m3_ocr_function=my_custom_ocr,
        m4_validation_function=my_custom_validation,
        m4_correction_function=my_custom_correction,

        device=torch.device('cuda'),
        debug=True
    )
    """

    # ===== OPTION 3: Mixed approach =====
    """
    pipeline = FlexibleMeterPipeline(
        # Use model for M3 (OCR)
        m3_model="path/to/crnn_model.pth",
        m3_type="pytorch",

        # Use custom functions for others
        m1_function=custom_preprocess,
        m2_detection_function=custom_detect,
        m2_rotation_function=custom_rotate,

        # Use rule-based for M4
        m4_type="rule_based",

        device=torch.device('cuda'),
        debug=True
    )
    """

    # Process image
    result = pipeline.process(
        image_path="test_image.jpg",
        save_dir="output"
    )

    print(f"\nFinal Reading: {result['final']['text']}")
    print(f"Confidence: {result['final']['confidence']:.4f}")

#!/usr/bin/env python3
"""
FastAPI Server for Meter Reading Pipeline
M1 -> M2 -> M3 -> M3.5 -> M4

API Endpoint: POST /predict
Input: {"image": "base64_encoded_string"}
Output: {"prediction": "1234", "confidence": 0.95, "pipeline_data": {...}}
"""
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import torchvision.models as models

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from m4_crnn_reading.beam_search_decoder import create_decoder

# ====================== MODELS ======================


class M2_OrientationModel_Fixed(nn.Module):
    """M2: Angle Detection Model"""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.angle_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 1024),
            nn.GroupNorm(32, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
            nn.Tanh()
        )

    def forward(self, x):
        feats = self.backbone(x)
        vec = self.angle_head(feats)
        return vec


class M2_SmartRotator_Fixed:
    """M2: Smart Rotation"""
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = M2_OrientationModel_Fixed(pretrained=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_angle(self, image: np.ndarray) -> float:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(tensor)
        cos_val = vec[0, 0].cpu().item()
        sin_val = vec[0, 1].cpu().item()
        angle_deg = np.degrees(np.arctan2(sin_val, cos_val))
        return angle_deg

    def smart_rotate(self, image: np.ndarray, angle: float) -> tuple:
        correction_angle = -angle
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


class M3_5_DigitExtractor:
    """M3.5: Black Digit Extraction"""
    def __init__(self, min_crop_ratio: float = 0.75, fallback_ratio: float = 0.8):
        self.min_crop_ratio = min_crop_ratio
        self.fallback_ratio = fallback_ratio

    def detect_red_digit_region(self, img: np.ndarray) -> int:
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red color ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return int(w * self.fallback_ratio)

        red_regions = []
        for cnt in contours:
            x, y, w_red, h_red = cv2.boundingRect(cnt)
            if w_red > 5 and h_red > 10:
                red_regions.append((x, y, w_red, h_red))

        if len(red_regions) == 0:
            return int(w * self.fallback_ratio)

        # Find leftmost red digit
        red_regions.sort(key=lambda r: r[0])
        leftmost_red_x = red_regions[0][0]
        return leftmost_red_x

    def extract(self, img: np.ndarray) -> np.ndarray:
        red_x = self.detect_red_digit_region(img)
        h, w = img.shape[:2]

        # Safety check
        max_crop = int(w * self.min_crop_ratio)
        if red_x > max_crop:
            red_x = max_crop

        # Crop black digits (left side)
        black_digits = img[:, :red_x].copy()
        return black_digits


class CRNN(nn.Module):
    """M4: CRNN OCR Model"""
    def __init__(self, num_chars=11, num_channels=1, img_height=64, hidden_size=256):
        super(CRNN, self).__init__()

        self.num_chars = num_chars
        self.hidden_size = hidden_size

        # Custom CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        h_out = img_height // 16
        self.rnn_input_size = 512 * h_out

        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x):
        conv_out = self.cnn(x)
        b, c, h, w = conv_out.size()
        features = conv_out.permute(0, 3, 1, 2)
        features = features.contiguous().view(b, w, c * h)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        logits = logits.permute(1, 0, 2)
        return logits


class M4_OCR:
    """M4: OCR with Beam Search"""
    def __init__(self, model_path: str, beam_width: int = 10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CRNN(num_chars=11, img_height=64, hidden_size=256).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        self.decoder = create_decoder(method="beam", chars="0123456789", blank_idx=10, beam_width=beam_width)
        self.img_height = 64
        self.img_width = 224

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize
        resized = cv2.resize(gray, (self.img_width, self.img_height))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5

        # Add channel and batch dimensions
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image: np.ndarray) -> str:
        tensor = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)

        text = self.decoder.decode(logits)
        return text


# ====================== PIPELINE ======================


class MeterReadingPipeline:
    """M1 -> M2 -> M3 -> M3.5 -> M4"""

    def __init__(self,
                 m1_model_path: str,
                 m2_model_path: str,
                 m3_model_path: str,
                 m4_model_path: str,
                 m1_confidence: float = 0.15,
                 m3_confidence: float = 0.10,
                 beam_width: int = 10):

        print("[Pipeline] Loading models...")

        # M1: YOLO Detection
        print("[M1] Loading YOLO model...")
        self.m1_model = YOLO(m1_model_path)
        self.m1_confidence = m1_confidence

        # M2: Angle Detection + Rotation
        print("[M2] Loading angle model...")
        self.m2_rotator = M2_SmartRotator_Fixed(m2_model_path)

        # M3: ROI Detection
        print("[M3] Loading ROI YOLO model...")
        self.m3_model = YOLO(m3_model_path)
        self.m3_confidence = m3_confidence

        # M3.5: Digit Extraction
        print("[M3.5] Initializing digit extractor...")
        self.m3_5_extractor = M3_5_DigitExtractor()

        # M4: OCR
        print("[M4] Loading OCR model...")
        self.m4_ocr = M4_OCR(m4_model_path, beam_width=beam_width)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Pipeline] All models loaded on {self.device}")

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process image through full pipeline

        Returns:
            dict: {
                'prediction': '1234',
                'm1_bbox': [x1, y1, x2, y2],
                'm2_angle': 45.0,
                'm3_bbox': [x1, y1, x2, y2],
                'success': True
            }
        """
        result = {
            'prediction': None,
            'm1_bbox': None,
            'm2_angle': None,
            'm3_bbox': None,
            'success': False,
            'error': None
        }

        try:
            # M1: Detect meter
            m1_results = self.m1_model(image, verbose=False, conf=self.m1_confidence)
            if len(m1_results) == 0 or len(m1_results[0].boxes) == 0:
                result['error'] = 'M1: No meter detected'
                return result

            boxes = m1_results[0].boxes
            best_idx = boxes.conf.argmax()
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            result['m1_bbox'] = [int(x1), int(y1), int(x2), int(y2)]
            meter_crop = image[y1:y2, x1:x2]

            # M2: Rotate
            detected_angle = self.m2_rotator.predict_angle(meter_crop)
            meter_aligned, correction_angle = self.m2_rotator.smart_rotate(meter_crop, detected_angle)
            result['m2_angle'] = float(detected_angle)

            # M3: Detect ROI
            m3_results = self.m3_model(meter_aligned, verbose=False, conf=self.m3_confidence)
            if len(m3_results) == 0 or len(m3_results[0].boxes) == 0:
                result['error'] = 'M3: No ROI detected'
                return result

            boxes = m3_results[0].boxes
            best_idx = boxes.conf.argmax()
            cx1, cy1, cx2, cy2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            result['m3_bbox'] = [int(cx1), int(cy1), int(cx2), int(cy2)]
            roi_crop = meter_aligned[cy1:cy2, cx1:cx2]

            # M3.5: Extract black digits
            black_digits = self.m3_5_extractor.extract(roi_crop)

            # M4: OCR
            predicted_text = self.m4_ocr.predict(black_digits)
            result['prediction'] = predicted_text
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            import traceback
            traceback.print_exc()

        return result


# ====================== FASTAPI APP ======================


# Pydantic models
class PredictRequest(BaseModel):
    image: str  # Base64 encoded image


class PredictResponse(BaseModel):
    prediction: Optional[str]
    success: bool
    pipeline_data: Optional[Dict[str, Any]]
    error: Optional[str]
    timestamp: str


# Initialize FastAPI
app = FastAPI(
    title="Meter Reading API",
    description="AI Pipeline for reading water meter values (M1->M2->M3->M3.5->M4)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[MeterReadingPipeline] = None


# Model paths (adjust these to your actual paths)
MODEL_PATHS = {
    'm1': r"F:\Workspace\Project\model\M1_DetectWatermeter.pt",
    'm2': r"F:\Workspace\Project\model\m2_angle_model_epoch15_FIXED_COS_SIN.pth",
    'm3': r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt",
    'm4': r"F:\Workspace\Project\model\M4_OCR.pth",
}


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global pipeline
    print("="*70)
    print("STARTING FASTAPI SERVER")
    print("="*70)
    pipeline = MeterReadingPipeline(
        m1_model_path=MODEL_PATHS['m1'],
        m2_model_path=MODEL_PATHS['m2'],
        m3_model_path=MODEL_PATHS['m3'],
        m4_model_path=MODEL_PATHS['m4'],
        m1_confidence=0.15,
        m3_confidence=0.10,
        beam_width=10
    )
    print("="*70)
    print("SERVER READY - Listening on http://localhost:8000")
    print("="*70)


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        'name': 'Meter Reading API',
        'version': '1.0.0',
        'status': 'running',
        'pipeline': 'M1 -> M2 -> M3 -> M3.5 -> M4',
        'docs': '/docs'
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'pipeline_loaded': pipeline is not None,
        'device': str(pipeline.device) if pipeline else 'N/A'
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Main prediction endpoint

    Expects JSON: {"image": "base64_encoded_string"}
    Returns: {"prediction": "1234", "success": true, "pipeline_data": {...}}
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        # Decode base64 image
        image_data = request.image
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        # Decode
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Run pipeline
        result = pipeline.process(image)

        return PredictResponse(
            prediction=result['prediction'],
            success=result['success'],
            pipeline_data={
                'm1_bbox': result['m1_bbox'],
                'm2_angle': result['m2_angle'],
                'm3_bbox': result['m3_bbox'],
            } if result['success'] else None,
            error=result['error'],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

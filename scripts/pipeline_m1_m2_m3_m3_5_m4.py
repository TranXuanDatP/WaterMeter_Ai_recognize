"""
Full Pipeline: M1 -> M2 (Fixed) -> M3 -> M3.5 -> M4

Changes from pipeline_with_lower_threshold.py:
1. M1 Confidence: 0.25 -> 0.15
2. M3 Confidence: 0.25 -> 0.10
3. ADD M3.5: Extract black digits from ROI
4. ADD M4: CRNN OCR reading
"""
import sys
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import torchvision.models as models

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import beam search decoder
from m4_crnn_reading.beam_search_decoder import create_decoder

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# M2 Model (Fixed)
class M2_OrientationModel_Fixed(nn.Module):
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
    def __init__(self, model_path: str):
        print("[M2] Loading FIXED model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = M2_OrientationModel_Fixed(pretrained=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"      Model: {model_path}")
        print(f"      Epoch: {checkpoint.get('epoch')}")
        print(f"      Val Loss: {checkpoint.get('val_loss'):.6f}")
        print(f"      Device: {self.device}")

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


# M3.5: Black Digit Extraction
class M3_5_DigitExtractor:
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


# M4: CRNN OCR Model (matching src/m4_crnn_reading/model.py)
class CRNN(nn.Module):
    def __init__(self, num_chars=11, num_channels=1, img_height=64, hidden_size=256):
        super(CRNN, self).__init__()

        self.num_chars = num_chars
        self.hidden_size = hidden_size

        # Custom CNN Feature Extractor (matches Colab training)
        self.cnn = nn.Sequential(
            # Block 1: 1 -> 64
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2

            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4

            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/8, W/8

            # Block 4: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/16, W/8

            # Block 5: 512 -> 512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # No pooling here
        )

        # Calculate RNN input size
        h_out = img_height // 16
        self.rnn_input_size = 512 * h_out  # 512 * 4 = 2048 for img_height=64

        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        # Output projection (bidirectional → 2*hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_chars)

    def forward(self, x):
        # CNN feature extraction
        conv_out = self.cnn(x)  # (B, 512, H/16, W/8)

        # Reshape for RNN: (B, C, H, W) -> (B, W, C*H)
        b, c, h, w = conv_out.size()

        # Permute and reshape: flatten features along height
        features = conv_out.permute(0, 3, 1, 2)  # (B, W, C, H)
        features = features.contiguous().view(b, w, c * h)  # (B, W, 512*4=2048)

        # RNN processing
        rnn_out, _ = self.rnn(features)  # (B, W, 512)

        # Output projection
        logits = self.fc(rnn_out)  # (B, W, num_chars)

        # For compatibility with CTC decoder, transpose: (B, T, C) -> (T, B, C)
        logits = logits.permute(1, 0, 2)  # (T, B, C)

        return logits


class M4_OCR:
    def __init__(self, model_path: str, beam_width: int = 10):
        print("[M4] Loading OCR model with Beam Search...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CRNN(num_chars=11, img_height=64, hidden_size=256).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"      Model: {model_path}")
        print(f"      Device: {self.device}")
        print(f"      Beam Search: Enabled (width={beam_width})")

        # Use beam search decoder instead of greedy
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

    def decode(self, output) -> str:
        # Get predictions
        _, preds = torch.max(output, dim=2)
        preds = preds.cpu().numpy()[0]

        # Remove duplicates and blanks
        chars = []
        last_char = None
        for p in preds:
            if p != 10:  # Not blank (assuming 10 is blank/CTC)
                if p != last_char:
                    chars.append(self.chars[p] if p < len(self.chars) else '')
                last_char = p

        return ''.join(chars)

    def predict(self, image: np.ndarray) -> str:
        tensor = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)

        text = self.decoder.decode(logits)
        return text


class Config:
    M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
    M2_MODEL = r"F:\Workspace\Project\model\m2_angle_model_epoch15_FIXED_COS_SIN.pth"
    M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"
    M4_MODEL = r"F:\Workspace\Project\model\M4_OCR.pth"

    DATA_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
    LABELS_FILE = Path(r"F:\Workspace\Project\data\images_4digit2.csv")

    OUTPUT_DIR = Path(r"F:\Workspace\Project\results\pipeline_full_m1_m2_m3_m3_5_m4_beam")
    LOG_DIR = OUTPUT_DIR / "logs"

    M1_CROPS_DIR = OUTPUT_DIR / "m1_crops"
    M2_ALIGNED_DIR = OUTPUT_DIR / "m2_aligned"
    M3_ROI_DIR = OUTPUT_DIR / "m3_roi_crops"
    M3_5_BLACK_DIGITS_DIR = OUTPUT_DIR / "m3_5_black_digits"

    # Lower thresholds
    M1_CONFIDENCE = 0.15  # Was 0.25
    M3_CONFIDENCE = 0.10  # Was 0.25

    # Beam search decoder
    BEAM_WIDTH = 10  # Recommended: 10 for best accuracy/speed tradeoff


def run_pipeline(config: Config, num_samples: int = None):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    config.M1_CROPS_DIR.mkdir(exist_ok=True)
    config.M2_ALIGNED_DIR.mkdir(exist_ok=True)
    config.M3_ROI_DIR.mkdir(exist_ok=True)
    config.M3_5_BLACK_DIGITS_DIR.mkdir(exist_ok=True)

    print("="*70)
    print("FULL PIPELINE: M1 -> M2 -> M3 -> M3.5 -> M4")
    print("="*70)
    print(f"M1 Confidence: {config.M1_CONFIDENCE} (was 0.25)")
    print(f"M3 Confidence: {config.M3_CONFIDENCE} (was 0.25)")
    print(f"Output: {config.OUTPUT_DIR}")
    print("="*70)
    print()

    print("[1/5] Loading models...")
    m1_model = YOLO(config.M1_MODEL)
    m2_rotator = M2_SmartRotator_Fixed(config.M2_MODEL)
    m3_model = YOLO(config.M3_MODEL)
    m3_5_extractor = M3_5_DigitExtractor()
    m4_ocr = M4_OCR(config.M4_MODEL, beam_width=config.BEAM_WIDTH)
    print(">> All models loaded\n")

    print("[2/5] Loading data...")
    labels_df = pd.read_csv(config.LABELS_FILE)
    image_files = list(config.DATA_DIR.glob("*.jpg"))
    if num_samples:
        image_files = image_files[:num_samples]
    print(f">> Found {len(image_files)} images\n")

    print("[3/5] Running pipeline...")
    results = []

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            img_name = img_path.name
            label_row = labels_df[labels_df['photo_name'] == img_name]
            if len(label_row) == 0:
                continue

            true_value = label_row.iloc[0]['value']
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # M1
            m1_results = m1_model(img, verbose=False, conf=config.M1_CONFIDENCE)
            if len(m1_results) == 0 or len(m1_results[0].boxes) == 0:
                continue

            boxes = m1_results[0].boxes
            best_idx = boxes.conf.argmax()
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            meter_crop = img[y1:y2, x1:x2]

            # M2
            detected_angle = m2_rotator.predict_angle(meter_crop)
            meter_aligned, correction_angle = m2_rotator.smart_rotate(meter_crop, detected_angle)

            # M3
            m3_results = m3_model(meter_aligned, verbose=False, conf=config.M3_CONFIDENCE)
            if len(m3_results) == 0 or len(m3_results[0].boxes) == 0:
                continue

            boxes = m3_results[0].boxes
            best_idx = boxes.conf.argmax()
            cx1, cy1, cx2, cy2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            roi_crop = meter_aligned[cy1:cy2, cx1:cx2]

            # M3.5
            black_digits = m3_5_extractor.extract(roi_crop)

            # M4
            predicted_text = m4_ocr.predict(black_digits)

            # Save intermediate results
            base_name = img_path.stem
            cv2.imwrite(str(config.M1_CROPS_DIR / f"{base_name}.jpg"), meter_crop)
            cv2.imwrite(str(config.M2_ALIGNED_DIR / f"{base_name}.jpg"), meter_aligned)
            cv2.imwrite(str(config.M3_ROI_DIR / f"{base_name}.jpg"), roi_crop)
            cv2.imwrite(str(config.M3_5_BLACK_DIGITS_DIR / f"{base_name}.jpg"), black_digits)

            results.append({
                'filename': img_path.name,
                'true_value': true_value,
                'predicted_value': predicted_text,
                'm1_bbox': [x1, y1, x2, y2],
                'm2_detected_angle': detected_angle,
                'm2_correction_angle': correction_angle,
                'm3_bbox': [cx1, cy1, cx2, cy2],
                'correct': str(true_value) == predicted_text
            })

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n>> Processed {len(results)} images\n")

    print("[4/5] Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.OUTPUT_DIR / "pipeline_results.csv", index=False)

    stats = {
        'total_images': len(image_files),
        'successful': len(results),
        'success_rate': len(results) / len(image_files) if image_files else 0,
        'm1_confidence': config.M1_CONFIDENCE,
        'm3_confidence': config.M3_CONFIDENCE,
        'timestamp': datetime.now().isoformat()
    }

    with open(config.OUTPUT_DIR / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f">> Results saved to {config.OUTPUT_DIR}\n")

    print("[5/5] Computing OCR accuracy...")
    if results:
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / len(results) * 100

        angles = [r['m2_detected_angle'] for r in results]

        print()
        print("="*70)
        print("STATISTICS")
        print("="*70)
        print(f"Total images: {stats['total_images']}")
        print(f"Successful pipeline: {stats['successful']}")
        print(f"Pipeline success rate: {stats['success_rate']:.2%}")
        print(f"\nOCR Accuracy:")
        print(f"  Correct: {correct}/{len(results)}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"\nM2 Angles:")
        print(f"  Mean: {np.mean(angles):.2f}°")
        print(f"  Std:  {np.std(angles):.2f}°")
        print(f"  Range: {np.min(angles):.2f}° → {np.max(angles):.2f}°")
        print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full pipeline with M3.5, M4 and Beam Search")
    parser.add_argument("--samples", type=int, default=None,
                       help="Number of samples to process (default: all)")
    parser.add_argument("--m1-conf", type=float, default=0.15)
    parser.add_argument("--m3-conf", type=float, default=0.10)
    parser.add_argument("--beam-width", type=int, default=10,
                       help="Beam search width (default: 10)")

    args = parser.parse_args()

    config = Config()
    config.M1_CONFIDENCE = args.m1_conf
    config.M3_CONFIDENCE = args.m3_conf
    config.BEAM_WIDTH = args.beam_width

    run_pipeline(config, num_samples=args.samples)

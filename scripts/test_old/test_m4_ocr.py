"""
M4 OCR Inference - Test CRNN model with M5 black digits

This script tests the M4_OCR.pth model to read digits from black digit images.
"""
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn as nn

# Fix encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ============================================
# CONFIGURATION
# ============================================

INPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m5_black_digits")
MODEL_PATH = Path(r"F:\Workspace\Project\model\M4_OCR.pth")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m4_ocr_results")
IMG_SIZE = (64, 224)  # (height, width) for CRNN input

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# MODEL ARCHITECTURE (Matching Colab Training)
# ============================================

class CRNN(nn.Module):
    """
    CRNN Model for meter reading - Custom CNN Architecture
    Matching M4_ocr_training_complete_colab.ipynb
    """
    def __init__(self, num_classes=11, num_channels=1, img_height=64, hidden_size=256):
        super(CRNN, self).__init__()

        # CNN Feature Extractor (Custom, not ResNet!)
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
        # After CNN: height = img_height // 16, width = img_width // 8
        # Features: 512 channels * (img_height // 16)
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

        # Output projection
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            logits: (B, T, C) - batch_first=True
        """
        # CNN feature extraction
        conv_out = self.cnn(x)  # (B, 512, H/16, W/8)

        # Reshape for RNN: (B, C, H, W) -> (B, W, C*H)
        b, c, h, w = conv_out.size()
        assert h == 4, f"Expected height=4, got {h}"

        # Permute and reshape: flatten features along height
        features = conv_out.permute(0, 3, 1, 2)  # (B, W, C, H)
        features = features.contiguous().view(b, w, c * h)  # (B, W, 512*4=2048)

        # RNN processing
        rnn_out, _ = self.rnn(features)  # (B, W, 512)

        # Output projection
        logits = self.fc(rnn_out)  # (B, W, num_classes)

        # For compatibility with decoder, transpose: (B, T, C) -> (T, B, C)
        logits = logits.permute(1, 0, 2)  # (T, B, C)

        return logits

print("=" * 80)
print("M4 OCR INFERENCE - CRNN Digit Recognition")
print("=" * 80)
print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

# ============================================
# CTC DECODER
# ============================================

class CTCDecoder:
    """
    CTC Decoder for converting logits to text
    """
    def __init__(self, chars="0123456789", blank_idx=10):
        self.chars = chars
        self.blank_idx = blank_idx

    def decode(self, logits):
        """
        Decode CTC logits to text using greedy decoding
        Args:
            logits: (T, N, C) - T time steps, N batch, C classes
        Returns:
            text: Decoded string
        """
        # Get argmax along class dimension
        pred_indices = logits.argmax(dim=2)  # (T, N)

        # Collapse CTC output
        decoded = []
        prev_idx = None

        for t in range(pred_indices.size(0)):
            idx = pred_indices[t, 0].item()  # First batch item

            # Skip blank tokens
            if idx == self.blank_idx:
                continue

            # Skip consecutive duplicates
            if idx == prev_idx:
                continue

            # Add character
            decoded.append(self.chars[idx])
            prev_idx = idx

        return ''.join(decoded)


CHAR_MAP = "0123456789"  # 10 digits
decoder = CTCDecoder(chars=CHAR_MAP, blank_idx=10)

def decode_predictions(pred_logits):
    """
    Decode predictions using CTC decoder

    Args:
        pred_logits: Can be torch.Tensor (T, C) or numpy array (T, C)

    Returns:
        Decoded text string
    """
    # Convert to torch tensor if numpy array
    if isinstance(pred_logits, np.ndarray):
        # Convert to tensor and add batch dimension: (T, C) -> (T, 1, C)
        pred_tensor = torch.from_numpy(pred_logits).unsqueeze(1)
    else:
        # Already a tensor, add batch dimension if needed
        if pred_logits.dim() == 2:
            pred_tensor = pred_logits.unsqueeze(1)  # (T, C) -> (T, 1, C)
        else:
            pred_tensor = pred_logits

    return decoder.decode(pred_tensor)

# ============================================
# PREPROCESSING
# ============================================

def preprocess_image(img, target_size=IMG_SIZE):
    """
    Preprocess image for CRNN

    Args:
        img: Input image (BGR)
        target_size: (height, width)

    Returns:
        tensor: Preprocessed tensor (1, 1, H, W)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(gray, (target_size[1], target_size[0]))

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

    return tensor

# ============================================
# LOAD MODEL
# ============================================

print(f"\n[1/4] Loading model...")

if not MODEL_PATH.exists():
    print(f"      ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

model = CRNN(num_classes=len(CHAR_MAP) + 1)  # +1 for CTC blank
model = model.to(device)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"      ✓ Model loaded")
    if 'epoch' in checkpoint:
        print(f"      Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"      Loss: {checkpoint['loss']:.4f}")
else:
    model.load_state_dict(checkpoint, strict=False)
    print(f"      ✓ Model loaded (legacy format)")

model.eval()

# ============================================
# LOAD IMAGES
# ============================================

print(f"\n[2/4] Loading images...")

image_files = sorted(list(INPUT_DIR.glob('*.jpg'))) + \
               sorted(list(INPUT_DIR.glob('*.png')))

if len(image_files) == 0:
    print(f"      ERROR: No images found in {INPUT_DIR}")
    sys.exit(1)

print(f"      Found {len(image_files)} images")

# ============================================
# RUN INFERENCE
# ============================================

print(f"\n[3/4] Running inference...")

results = []
success_count = 0
error_count = 0

for img_path in tqdm(image_files, desc="Processing"):
    try:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            error_count += 1
            results.append({
                'filename': img_path.name,
                'status': 'error',
                'error': 'Could not read image'
            })
            continue

        h, w = img.shape[:2]
        original_size = f"{w}x{h}"

        # Preprocess
        input_tensor = preprocess_image(img).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)  # (T, B, C) - T time steps, B batch, C classes

        # Get predictions for first batch item: (T, B, C) -> (T, C)
        pred = output[:, 0, :].cpu()  # (T, C)

        # Decode to text
        predicted_text = decode_predictions(pred)

        # Calculate confidence (mean max probability)
        probs = torch.softmax(pred, dim=1)  # (T, C)
        max_probs = probs.max(dim=1)[0].cpu().numpy()
        confidence = float(np.mean(max_probs))

        results.append({
            'filename': img_path.name,
            'status': 'success',
            'original_size': original_size,
            'predicted_text': predicted_text,
            'confidence': confidence,
            'length': len(predicted_text)
        })

        success_count += 1

    except Exception as e:
        error_count += 1
        results.append({
            'filename': img_path.name,
            'status': 'error',
            'error': str(e)
        })

# ============================================
# SAVE RESULTS
# ============================================

print(f"\n[4/4] Results:")

# Save to CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "ocr_results.csv", index=False)

print("-" * 80)

# Show results
for i, r in enumerate(results, 1):
    if r['status'] == 'success':
        text = r['predicted_text']
        conf = r['confidence']
        size = r['original_size']
        length = r['length']
        print(f"  {i:2d}. {r['filename'][:50]:50s}")
        print(f"      Size: {size} | Text: '{text}' (len={length}) | Conf: {conf:.3f}")
    else:
        print(f"  {i:2d}. {r['filename'][:50]:50s}")
        print(f"      ERROR: {r.get('error', 'Unknown')}")

# ============================================
# STATISTICS
# ============================================

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)

print(f"\nTotal images:   {len(image_files)}")
print(f"Success:        {success_count} ({success_count/len(image_files)*100:.1f}%)")
print(f"Errors:         {error_count} ({error_count/len(image_files)*100:.1f}%)")

if success_count > 0:
    success_df = df[df['status'] == 'success']

    print("\n" + "=" * 80)
    print("TEXT STATISTICS")
    print("=" * 80)

    # Text length distribution
    lengths = success_df['length'].values
    print(f"\nText length:")
    print(f"  Min:  {lengths.min()}")
    print(f"  Max:  {lengths.max()}")
    print(f"  Mean: {lengths.mean():.1f}")

    # Count by length
    print(f"\nLength distribution:")
    for length in sorted(lengths):
        count = (lengths == length).sum()
        print(f"  Length {length}: {count} images")

    # Confidence statistics
    confidences = success_df['confidence'].values
    print(f"\nConfidence:")
    print(f"  Min:  {confidences.min():.3f}")
    print(f"  Max:  {confidences.max():.3f}")
    print(f"  Mean: {confidences.mean():.3f}")
    print(f"  Std:  {confidences.std():.3f}")

    # Show unique predictions
    print(f"\nUnique predictions:")
    unique_texts = success_df['predicted_text'].unique()
    print(f"  Total: {len(unique_texts)} unique texts")
    if len(unique_texts) <= 20:
        for i, text in enumerate(sorted(unique_texts), 1):
            count = (success_df['predicted_text'] == text).sum()
            print(f"  {i:2d}. '{text}' ({count}x)")
    else:
        print(f"  Top 20 most common:")
        text_counts = success_df['predicted_text'].value_counts().head(20)
        for i, (text, count) in enumerate(text_counts.items(), 1):
            print(f"  {i:2d}. '{text}' ({count}x)")

print("\n" + "=" * 80)
print("OUTPUT LOCATIONS")
print("=" * 80)
print(f"Results CSV: {OUTPUT_DIR / 'ocr_results.csv'}")
print("=" * 80)

print(f"\n💡 Next steps:")
print(f"   1. Review OCR results in CSV")
print(f"   2. Check for any incorrect predictions")
print(f"   3. Use results for meter reading validation")
print(f"   4. Consider retraining if accuracy is low")
print("=" * 80)

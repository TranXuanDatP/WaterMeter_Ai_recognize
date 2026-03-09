"""
Quick Test Pipeline on data_4digit2 (Sample Only)

This script runs a quick test on the first 100 images from data_4digit2
to verify the pipeline works correctly before running on the full dataset.
"""
import sys
import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.m4_crnn_reading.model import CRNN
from src.m4_crnn_reading.beam_search_decoder import create_decoder

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ============================================
# CONFIGURATION
# ============================================

# Data paths
DATA_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
LABELS_FILE = Path(r"F:\Workspace\Project\data\images_4digit2.csv")
MODEL_PATH = Path(r"F:\Workspace\Project\model\M4_OCR.pth")

# Output paths
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline_data_4digit2_quick")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model config
IMG_SIZE = (64, 224)
CHAR_MAP = "0123456789"

# Decoder config
DECODER_METHOD = 'beam'
BEAM_WIDTH = 10

# Test on first N images only
SAMPLE_SIZE = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# SETUP
# ============================================

print("=" * 80)
print(f"QUICK PIPELINE TEST ON DATA_4DIGIT2 (First {SAMPLE_SIZE} images)")
print("=" * 80)
print(f"Total images in dataset: 6527")
print(f"Testing on: {SAMPLE_SIZE} images")
print(f"Decoder: {DECODER_METHOD} (beam_width={BEAM_WIDTH})")
print("=" * 80)

# ============================================
# LOAD DATA
# ============================================

print(f"\n[1/4] Loading labels...")
labels_df = pd.read_csv(LABELS_FILE)
print(f"      Loaded {len(labels_df)} labels")

print(f"\n[2/4] Loading image list...")
all_image_files = list(DATA_DIR.glob('*.jpg')) + list(DATA_DIR.glob('*.png'))
print(f"      Found {len(all_image_files)} images")

# Sample first N images
image_files = sorted(all_image_files)[:SAMPLE_SIZE]
print(f"      Testing on first {len(image_files)} images")

# ============================================
# LOAD MODEL & DECODER
# ============================================

print(f"\n[3/4] Loading model and decoder...")

model = CRNN(num_chars=len(CHAR_MAP) + 1)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
model = model.to(device)

decoder = create_decoder(
    method=DECODER_METHOD,
    chars=CHAR_MAP,
    blank_idx=10,
    beam_width=BEAM_WIDTH
)

print(f"      Model loaded (epoch: {checkpoint.get('epoch', 'N/A')})")
print(f"      Decoder created")

# ============================================
# PREPROCESSING
# ============================================

def preprocess_image(img, target_size=IMG_SIZE):
    """Preprocess image for CRNN"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (target_size[1], target_size[0]))
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor

# ============================================
# RUN PIPELINE
# ============================================

print(f"\n[4/4] Running pipeline...")
print("-" * 80)

results = []
correct_count = 0
incorrect_count = 0
error_count = 0

from tqdm import tqdm

for img_path in tqdm(image_files, desc="Processing"):
    try:
        # Get ground truth label
        img_name = img_path.name
        label_rows = labels_df[labels_df['photo_name'] == img_name]

        if len(label_rows) == 0:
            continue

        ground_truth = str(label_rows.iloc[0]['value']).strip()

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            error_count += 1
            continue

        h, w = img.shape[:2]

        # Preprocess
        input_tensor = preprocess_image(img).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)

        # Get logits
        logits = output[:, 0, :]

        # Decode with beam search
        predicted_text = decoder.decode(logits)

        # Calculate confidence
        probs = torch.softmax(logits, dim=1)
        max_probs = probs.max(dim=1)[0].cpu().numpy()
        confidence = float(np.mean(max_probs))

        # Check if correct
        is_correct = (predicted_text == ground_truth)

        result = {
            'filename': img_name,
            'ground_truth': ground_truth,
            'predicted_text': predicted_text,
            'confidence': confidence,
            'is_correct': is_correct,
            'original_size': f"{w}x{h}"
        }

        results.append(result)

        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

    except Exception as e:
        error_count += 1
        continue

# ============================================
# PRINT SUMMARY
# ============================================

print("\n" + "=" * 80)
print("QUICK TEST SUMMARY")
print("=" * 80)

total = len(results)
accuracy = correct_count / total * 100 if total > 0 else 0

print(f"\nTotal processed: {total}")
print(f"Correct:         {correct_count} ({accuracy:.2f}%)")
print(f"Incorrect:       {incorrect_count} ({100-accuracy:.2f}%)")
print(f"Errors:          {error_count}")

if len(results) > 0:
    df = pd.DataFrame(results)

    print(f"\nConfidence Statistics:")
    print(f"  Mean: {df['confidence'].mean():.4f}")
    print(f"  Min:  {df['confidence'].min():.4f}")
    print(f"  Max:  {df['confidence'].max():.4f}")

    # Show incorrect predictions
    if incorrect_count > 0:
        print(f"\n" + "=" * 80)
        print("INCORRECT PREDICTIONS (First 10)")
        print("=" * 80)

        incorrect_df = df[df['is_correct'] == False].head(10)

        print(f"\n{'Filename':<50} {'Ground Truth':<15} {'Predicted':<15}")
        print("-" * 80)

        for _, row in incorrect_df.iterrows():
            print(f"{row['filename']:<50} {row['ground_truth']:<15} {row['predicted_text']:<15}")

# Save results
output_csv = OUTPUT_DIR / "quick_test_results.csv"
if len(results) > 0:
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

print("\n" + "=" * 80)
print("[OK] QUICK TEST COMPLETED!")
print("=" * 80)

print(f"\n💡 Next steps:")
print(f"  1. If accuracy is good (>90%), run full pipeline on all 6527 images")
print(f"  2. Use: python scripts/test_pipeline_data_4digit2.py")
print(f"  3. Monitor logs in: {OUTPUT_DIR / 'logs'}")
print("=" * 80)

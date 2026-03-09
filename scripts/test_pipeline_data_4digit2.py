"""
Test Pipeline on data_4digit2 with Full Logging

This script runs the complete pipeline (M1-M4) on the data_4digit2 dataset
with comprehensive logging and beam search decoder.

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

# Add src to path
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
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline_data_4digit2")
LOG_DIR = OUTPUT_DIR / "logs"
M4_OUTPUT_DIR = OUTPUT_DIR / "m4_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
M4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model config
IMG_SIZE = (64, 224)
CHAR_MAP = "0123456789"

# Decoder config
DECODER_METHOD = 'beam'
BEAM_WIDTH = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# SETUP LOGGING
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
            'successful': 0,
            'errors': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
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
# LOAD DATA
# ============================================

print("=" * 80)
print("PIPELINE TEST ON DATA_4DIGIT2")
print("=" * 80)
print(f"Data directory: {DATA_DIR}")
print(f"Labels file: {LABELS_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model: {MODEL_PATH}")
print(f"Decoder: {DECODER_METHOD} (beam_width={BEAM_WIDTH})")
print("=" * 80)

logger = PipelineLogger(LOG_DIR)
logger.log("=" * 80)
logger.log("PIPELINE TEST ON DATA_4DIGIT2")
logger.log("=" * 80)

# Load labels
print(f"\n[1/6] Loading labels...")
logger.log(f"Loading labels from: {LABELS_FILE}")

try:
    labels_df = pd.read_csv(LABELS_FILE)
    print(f"      Loaded {len(labels_df)} labels")
    logger.log(f"Loaded {len(labels_df)} labels")
    logger.log(f"Columns: {list(labels_df.columns)}")
except Exception as e:
    print(f"      ERROR: Failed to load labels: {e}")
    logger.log(f"ERROR: Failed to load labels: {e}")
    sys.exit(1)

# Check data directory
print(f"\n[2/6] Checking data directory...")
logger.log(f"Checking data directory: {DATA_DIR}")

image_files = list(DATA_DIR.glob('*.jpg')) + list(DATA_DIR.glob('*.png'))
print(f"      Found {len(image_files)} images")
logger.log(f"Found {len(image_files)} images")

if len(image_files) == 0:
    print(f"      ERROR: No images found in {DATA_DIR}")
    logger.log(f"ERROR: No images found in {DATA_DIR}")
    sys.exit(1)

# ============================================
# LOAD MODEL
# ============================================

print(f"\n[3/6] Loading M4 OCR model...")
logger.log(f"Loading model: {MODEL_PATH}")

try:
    model = CRNN(num_chars=len(CHAR_MAP) + 1)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model = model.to(device)

    print(f"      Epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.log(f"Model loaded successfully (epoch: {checkpoint.get('epoch', 'N/A')})")
except Exception as e:
    print(f"      ERROR: Failed to load model: {e}")
    logger.log(f"ERROR: Failed to load model: {e}")
    sys.exit(1)

# ============================================
# CREATE DECODER
# ============================================

print(f"\n[4/6] Creating beam search decoder...")
logger.log(f"Creating decoder: method={DECODER_METHOD}, beam_width={BEAM_WIDTH}")

try:
    decoder = create_decoder(
        method=DECODER_METHOD,
        chars=CHAR_MAP,
        blank_idx=10,
        beam_width=BEAM_WIDTH
    )
    print(f"      Decoder created successfully")
    logger.log(f"Decoder created successfully")
except Exception as e:
    print(f"      ERROR: Failed to create decoder: {e}")
    logger.log(f"ERROR: Failed to create decoder: {e}")
    sys.exit(1)

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

print(f"\n[5/6] Running pipeline on {len(image_files)} images...")
logger.log(f"Starting pipeline processing on {len(image_files)} images")
print("-" * 80)

results = []
error_files = []

for img_path in tqdm(image_files, desc="Processing"):
    try:
        # Get ground truth label
        img_name = img_path.name
        label_rows = labels_df[labels_df['photo_name'] == img_name]

        if len(label_rows) == 0:
            logger.log(f"WARNING: No label found for {img_name}")
            ground_truth = ""
        else:
            ground_truth = str(label_rows.iloc[0]['value']).strip()

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise Exception("Could not read image")

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
        is_correct = (predicted_text == ground_truth) if ground_truth else None

        result = {
            'filename': img_name,
            'ground_truth': ground_truth,
            'predicted_text': predicted_text,
            'confidence': confidence,
            'is_correct': is_correct,
            'original_size': f"{w}x{h}",
            'status': 'success'
        }

        results.append(result)

        # Update statistics
        logger.stats['total_images'] += 1
        logger.stats['successful'] += 1

        if is_correct is True:
            logger.stats['correct_predictions'] += 1
        elif is_correct is False:
            logger.stats['incorrect_predictions'] += 1

        # Log incorrect predictions
        if is_correct is False:
            logger.log(f"INCORRECT: {img_name} | GT: '{ground_truth}' | Pred: '{predicted_text}' | Conf: {confidence:.4f}")

    except Exception as e:
        error_files.append((img_name, str(e)))
        logger.stats['total_images'] += 1
        logger.stats['errors'] += 1
        logger.log(f"ERROR processing {img_name}: {e}")

# ============================================
# SAVE RESULTS
# ============================================

print(f"\n[6/6] Saving results...")

# Create results DataFrame
df = pd.DataFrame(results)

# Save detailed results
output_csv = OUTPUT_DIR / "pipeline_results.csv"
df.to_csv(output_csv, index=False)
logger.log(f"Results saved to: {output_csv}")

# Save only incorrect predictions for review
if len(df[df['is_correct'] == False]) > 0:
    incorrect_csv = OUTPUT_DIR / "incorrect_predictions.csv"
    df[df['is_correct'] == False].to_csv(incorrect_csv, index=False)
    logger.log(f"Incorrect predictions saved to: {incorrect_csv}")

# ============================================
# PRINT SUMMARY
# ============================================

print("\n" + "=" * 80)
print("PIPELINE EXECUTION SUMMARY")
print("=" * 80)

# Overall statistics
total = logger.stats['total_images']
success = logger.stats['successful']
errors = logger.stats['errors']
correct = logger.stats['correct_predictions']
incorrect = logger.stats['incorrect_predictions']

print(f"\nTotal images:     {total}")
print(f"Successful:       {success} ({success/total*100:.1f}%)")
print(f"Errors:            {errors} ({errors/total*100:.1f}%)")

# Accuracy (excluding errors)
if success > 0:
    accuracy = correct / success * 100
    print(f"\nAccuracy (on {success} successful predictions):")
    print(f"  Correct:         {correct}")
    print(f"  Incorrect:       {incorrect}")
    print(f"  Accuracy:        {accuracy:.2f}%")

# Confidence statistics
if len(df) > 0 and 'confidence' in df.columns:
    print(f"\nConfidence Statistics:")
    print(f"  Mean:            {df['confidence'].mean():.4f}")
    print(f"  Std:             {df['confidence'].std():.4f}")
    print(f"  Min:             {df['confidence'].min():.4f}")
    print(f"  Max:             {df['confidence'].max():.4f}")

# Text length statistics
if len(df) > 0 and 'predicted_text' in df.columns:
    df['text_length'] = df['predicted_text'].str.len()
    print(f"\nText Length Statistics:")
    print(f"  Mean:            {df['text_length'].mean():.1f}")
    print(f"  Min:             {df['text_length'].min():.0f}")
    print(f"  Max:             {df['text_length'].max():.0f}")

# Length distribution
length_counts = df['text_length'].value_counts().sort_index()
print(f"\nLength Distribution:")
for length, count in length_counts.items():
    print(f"  Length {int(length):2d}:    {count:4d} images")

# Show incorrect predictions
if incorrect > 0:
    print(f"\n" + "=" * 80)
    print("INCORRECT PREDICTIONS (First 20)")
    print("=" * 80)

    incorrect_df = df[df['is_correct'] == False].head(20)

    print(f"\n{'Filename':<50} {'Ground Truth':<15} {'Predicted':<15} {'Conf':<10}")
    print("-" * 80)

    for _, row in incorrect_df.iterrows():
        print(f"{row['filename']:<50} {row['ground_truth']:<15} {row['predicted_text']:<15} {row['confidence']:<10.4f}")

# Save summary
summary_file = logger.save_summary()

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)
print(f"Results CSV:      {output_csv}")
if incorrect > 0:
    print(f"Incorrect CSV:    {incorrect_csv}")
print(f"Summary JSON:     {summary_file}")
print(f"Log file:         {logger.log_file}")

print("\n" + "=" * 80)
print("[OK] PIPELINE TEST COMPLETED!")
print("=" * 80)

# Log summary
logger.log("=" * 80)
logger.log("PIPELINE EXECUTION SUMMARY")
logger.log("=" * 80)
logger.log(f"Total images:     {total}")
logger.log(f"Successful:       {success} ({success/total*100:.1f}%)")
logger.log(f"Errors:            {errors} ({errors/total*100:.1f}%)")
logger.log(f"Accuracy:         {accuracy:.2f}%")
logger.log("=" * 80)

print("\n💡 Next steps:")
print("  1. Review incorrect predictions CSV")
print("  2. Check log file for detailed errors")
print("  3. Analyze patterns in incorrect cases")
print("  4. Consider fine-tuning if accuracy is low")
print("=" * 80)

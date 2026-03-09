"""
Compare M4 OCR Decoders: Greedy vs Beam Search

Run this script to compare performance between greedy and beam search decoders
on the same dataset.
"""
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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

INPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m5_black_digits")
MODEL_PATH = Path(r"F:\Workspace\Project\model\M4_OCR.pth")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\decoder_comparison")
LABELS_FILE = r"F:\Workspace\Project\data\data_4digit.csv"

IMG_SIZE = (64, 224)
CHAR_MAP = "0123456789"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("M4 OCR DECODER COMPARISON: Greedy vs Beam Search")
print("=" * 80)
print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

# ============================================
# LOAD MODEL
# ============================================

print(f"\n[1/5] Loading model...")

model = CRNN(num_chars=len(CHAR_MAP) + 1)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
model = model.to(device)

print(f"      Epoch: {checkpoint.get('epoch', 'N/A')}")

# ============================================
# CREATE DECODERS
# ============================================

print(f"\n[2/5] Creating decoders...")

decoders = {
    'greedy': create_decoder('greedy', chars=CHAR_MAP, blank_idx=10),
    'beam_5': create_decoder('beam', chars=CHAR_MAP, blank_idx=10, beam_width=5),
    'beam_10': create_decoder('beam', chars=CHAR_MAP, blank_idx=10, beam_width=10),
    'beam_15': create_decoder('beam', chars=CHAR_MAP, blank_idx=10, beam_width=15),
    'prefix_beam_10': create_decoder('prefix_beam', chars=CHAR_MAP, blank_idx=10, beam_width=10),
}

print(f"      Created {len(decoders)} decoders:")
for name in decoders.keys():
    print(f"      - {name}")

# ============================================
# LOAD GROUND TRUTH LABELS
# ============================================

print(f"\n[3/5] Loading ground truth labels...")

try:
    labels_df = pd.read_csv(LABELS_FILE)
    print(f"      Loaded {len(labels_df)} labels")
    has_labels = True
except:
    print(f"      Warning: Could not load labels from {LABELS_FILE}")
    print(f"      Will run without ground truth comparison")
    has_labels = False

# ============================================
# LOAD IMAGES
# ============================================

print(f"\n[4/5] Loading images...")

image_files = sorted(list(INPUT_DIR.glob('*.jpg'))) + \
               sorted(list(INPUT_DIR.glob('*.png')))

if len(image_files) == 0:
    print(f"      ERROR: No images found in {INPUT_DIR}")
    sys.exit(1)

print(f"      Found {len(image_files)} images")

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
# RUN COMPARISON
# ============================================

print(f"\n[5/5] Running decoder comparison...")
print("-" * 80)

results = []

for img_path in tqdm(image_files, desc="Processing"):
    try:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Preprocess
        input_tensor = preprocess_image(img).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)  # (T, B, C)

        # Get logits
        logits = output[:, 0, :]  # (T, C)

        # Test all decoders
        result = {'filename': img_path.name}

        for decoder_name, decoder in decoders.items():
            try:
                pred_text = decoder.decode(logits)
                result[decoder_name] = pred_text
            except Exception as e:
                result[decoder_name] = f"ERROR: {e}"

        # Add ground truth if available
        if has_labels:
            # Try to find label for this image
            # Extract image ID from filename
            img_id = img_path.stem.split('_')[0]  # Get first part before underscore
            label_rows = labels_df[labels_df['filename'].str.contains(img_id)]

            if len(label_rows) > 0:
                result['ground_truth'] = label_rows.iloc[0]['value']
            else:
                result['ground_truth'] = ''

        results.append(result)

    except Exception as e:
        print(f"\nError processing {img_path.name}: {e}")
        continue

# ============================================
# ANALYZE RESULTS
# ============================================

print(f"\n" + "=" * 80)
print("DECODER COMPARISON RESULTS")
print("=" * 80)

# Create DataFrame
df = pd.DataFrame(results)

# Calculate accuracy for each decoder if ground truth available
if has_labels and 'ground_truth' in df.columns:
    df_with_gt = df[df['ground_truth'].notna() & (df['ground_truth'] != '')]

    if len(df_with_gt) > 0:
        print(f"\nAccuracy on {len(df_with_gt)} images with ground truth:")
        print(f"{'Decoder':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
        print("-" * 80)

        accuracy_results = {}

        for decoder_name in decoders.keys():
            if decoder_name not in df.columns:
                continue

            correct = sum(df_with_gt[decoder_name] == df_with_gt['ground_truth'])
            total = len(df_with_gt)
            accuracy = correct / total * 100 if total > 0 else 0

            accuracy_results[decoder_name] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }

            print(f"{decoder_name:<20} {correct:<10} {total:<10} {accuracy:<10.1f}%")

        # Find best decoder
        best_decoder = max(accuracy_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest decoder: {best_decoder[0]} ({best_decoder[1]['accuracy']:.1f}%)")

        # Show improvement over greedy
        if 'greedy' in accuracy_results and 'beam_10' in accuracy_results:
            greedy_acc = accuracy_results['greedy']['accuracy']
            beam_acc = accuracy_results['beam_10']['accuracy']
            improvement = beam_acc - greedy_acc
            print(f"Improvement (beam_10 vs greedy): +{improvement:.1f}%")
else:
    print("\nNo ground truth available for accuracy calculation")
    print("Showing first 10 predictions instead:")

    print(f"\n{'Filename':<60} {'Greedy':<15} {'Beam_10':<15}")
    print("-" * 80)

    for _, row in df.head(10).iterrows():
        filename = row['filename'][:60]
        greedy = row.get('greedy', 'N/A')
        beam_10 = row.get('beam_10', 'N/A')
        print(f"{filename:<60} {greedy:<15} {beam_10:<15}")

# Save results
output_csv = OUTPUT_DIR / 'decoder_comparison.csv'
df.to_csv(output_csv, index=False)

print(f"\n" + "=" * 80)
print(f"Results saved to: {output_csv}")
print("=" * 80)

# Show cases where beam search fixes greedy errors
if has_labels and 'ground_truth' in df.columns:
    print(f"\n" + "=" * 80)
    print("CASES WHERE BEAM SEARCH (width=10) FIXES GREEDY ERRORS")
    print("=" * 80)

    df_with_gt = df[df['ground_truth'].notna() & (df['ground_truth'] != '')]

    greedy_wrong = df_with_gt['greedy'] != df_with_gt['ground_truth']
    beam_right = df_with_gt['beam_10'] == df_with_gt['ground_truth']

    fixed_cases = df_with_gt[greedy_wrong & beam_right]

    if len(fixed_cases) > 0:
        print(f"\nFound {len(fixed_cases)} cases fixed by beam search:\n")

        for _, row in fixed_cases.head(20).iterrows():
            print(f"  {row['filename'][:60]}")
            print(f"    Ground Truth: {row['ground_truth']}")
            print(f"    Greedy:       {row['greedy']} → Beam_10: {row['beam_10']}")
            print()
    else:
        print("None found")

print("=" * 80)
print("[OK] DECODER COMPARISON COMPLETED!")
print("=" * 80)

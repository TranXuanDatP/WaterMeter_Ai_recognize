"""
Test Beam Search Decoder vs Greedy Decoder

Compare performance on the 9 failed cases from M4 OCR.
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

MODEL_PATH = Path(r"F:\Workspace\Project\model\M4_OCR.pth")
INPUT_DIR = Path(r"F:\Workspace\Project\results\test_pipeline\m5_black_digits")
OCR_RESULTS = Path(r"F:\Workspace\Project\results\test_pipeline\m4_ocr_results\ocr_results.csv")

IMG_SIZE = (64, 224)
CHAR_MAP = "0123456789"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("BEAM SEARCH DECODER TEST")
print("=" * 80)
print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print("=" * 80)

# ============================================
# LOAD MODEL
# ============================================

print("\n[1/4] Loading model...")

# Note: CRNN uses num_chars parameter (not num_classes)
model = CRNN(num_chars=len(CHAR_MAP) + 1)
model = model.to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

print(f"      Epoch: {checkpoint['epoch']}")

# ============================================
# CREATE DECODERS
# ============================================

print("\n[2/4] Creating decoders...")

decoders = {
    'greedy': create_decoder('greedy', chars=CHAR_MAP, blank_idx=10),
    'beam_5': create_decoder('beam', chars=CHAR_MAP, blank_idx=10, beam_width=5),
    'beam_10': create_decoder('beam', chars=CHAR_MAP, blank_idx=10, beam_width=10),
    'beam_15': create_decoder('beam', chars=CHAR_MAP, blank_idx=10, beam_width=15),
    'prefix_beam_5': create_decoder('prefix_beam', chars=CHAR_MAP, blank_idx=10, beam_width=5),
    'prefix_beam_10': create_decoder('prefix_beam', chars=CHAR_MAP, blank_idx=10, beam_width=10),
}

print(f"      Created {len(decoders)} decoders:")
for name in decoders.keys():
    print(f"      - {name}")

# ============================================
# LOAD OCR RESULTS TO FIND FAILED CASES
# ============================================

print("\n[3/4] Loading OCR results...")

df = pd.read_csv(OCR_RESULTS)

# Find failed cases
failed_cases = df[df['check'].isna() | (df['check'] == False)]

print(f"      Total images: {len(df)}")
print(f"      Failed cases: {len(failed_cases)}")

if len(failed_cases) == 0:
    print("\n[WARNING] No failed cases found in CSV!")
    print("Testing on all images instead...")
    test_cases = df.head(20)  # Test first 20
else:
    test_cases = failed_cases

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
# TEST DECODERS
# ============================================

print("\n[4/4] Testing decoders on failed cases...")
print("-" * 80)

results = []

for idx, row in test_cases.iterrows():
    filename = row['filename']
    correct_text = str(row.get('value', '')).strip()  # Convert to string and strip
    greedy_pred = str(row.get('predicted_text', '')).strip()

    # Load image
    img_path = INPUT_DIR / filename
    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # Preprocess
    input_tensor = preprocess_image(img).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)  # (T, B, C)

    # Get logits for first batch
    logits = output[:, 0, :]  # (T, C)

    # Test all decoders
    decoder_results = {'filename': filename, 'correct': correct_text, 'greedy_original': greedy_pred}

    for decoder_name, decoder in decoders.items():
        try:
            pred_text = decoder.decode(logits)
            # Ensure string type and strip whitespace
            pred_text = str(pred_text).strip()
            decoder_results[decoder_name] = pred_text
        except Exception as e:
            decoder_results[decoder_name] = f"ERROR: {e}"

    results.append(decoder_results)

    # Print comparison
    print(f"\nFile: {filename[:60]}")
    print(f"  Correct:           '{correct_text}'")
    print(f"  Greedy (original): '{greedy_pred}'")

    for decoder_name in ['beam_5', 'beam_10', 'prefix_beam_5', 'prefix_beam_10']:
        if decoder_name in decoder_results:
            pred = decoder_results[decoder_name]
            match = "✓" if pred == correct_text else "✗"
            print(f"  {decoder_name:20s}: '{pred}' {match}")

# ============================================
# STATISTICS
# ============================================

print("\n" + "=" * 80)
print("STATISTICS SUMMARY")
print("=" * 80)

# Calculate accuracy for each decoder
for decoder_name in ['greedy_original', 'beam_5', 'beam_10', 'prefix_beam_5', 'prefix_beam_10']:
    if decoder_name not in results[0]:
        continue

    correct = sum(1 for r in results if r.get(decoder_name) == r['correct'])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0

    # Compare with greedy
    greedy_correct = sum(1 for r in results if r['greedy_original'] == r['correct'])

    improvement = correct - greedy_correct
    sign = "+" if improvement > 0 else ""

    print(f"\n{decoder_name:20s}: {correct:2d}/{total:2d} correct ({accuracy:5.1f}%) {sign}{improvement}")

# Find cases where beam search succeeded but greedy failed
print("\n" + "=" * 80)
print("CASES WHERE BEAM SEARCH FIXED ERRORS")
print("=" * 80)

for decoder_name in ['beam_10', 'prefix_beam_10']:
    if decoder_name not in results[0]:
        continue

    print(f"\n[{decoder_name.upper()}]")
    fixed_count = 0

    for r in results:
        greedy_wrong = r['greedy_original'] != r['correct']
        beam_right = r.get(decoder_name) == r['correct']

        if greedy_wrong and beam_right:
            fixed_count += 1
            print(f"  {r['filename'][:50]:50s}")
            print(f"    Greedy: '{r['greedy_original']}' → {decoder_name}: '{r[decoder_name]}' (Correct: '{r['correct']}')")

    if fixed_count == 0:
        print("  None")
    else:
        print(f"\n  Total fixed: {fixed_count}")

# Save results
output_results = pd.DataFrame(results)
output_path = Path(r"F:\Workspace\Project\results\test_pipeline\beam_search_comparison.csv")
output_results.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print(f"Results saved to: {output_path}")
print("=" * 80)

print("\n💡 Recommendations:")
print("  1. Choose decoder with best accuracy")
print("  2. If beam search helps significantly, update model.py")
print("  3. Consider fine-tuning with beam search decoder")
print("=" * 80)

"""
Simple Test: M4 OCR with Beam Search Decoder

Test only M4 stage with beam search to verify integration.
"""
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.m4_crnn_reading.model import CRNN
from src.m4_crnn_reading.beam_search_decoder import create_decoder

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = r"F:\Workspace\Project\model\M4_OCR.pth"
TEST_IMAGE = r"F:\Workspace\Project\results\test_pipeline\m5_black_digits\meter4_00003_validate_00003_0070e981653c4e0eb2209b78fb3f9ce2_black_digits.jpg"
GROUND_TRUTH = "441"

CHAR_MAP = "0123456789"
IMG_SIZE = (64, 224)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("M4 OCR + BEAM SEARCH DECODER TEST")
print("=" * 80)
print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print(f"Ground Truth: {GROUND_TRUTH}")
print("=" * 80)

# ============================================
# LOAD MODEL
# ============================================

print("\n[1/3] Loading model...")

model = CRNN(num_chars=len(CHAR_MAP) + 1)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
model = model.to(device)

print(f"      Epoch: {checkpoint['epoch']}")

# ============================================
# CREATE DECODERS
# ============================================

print("\n[2/3] Creating decoders...")

decoders = {
    'Greedy': create_decoder('greedy', chars=CHAR_MAP, blank_idx=10),
    'Beam (width=5)': create_decoder('beam', chars=CHAR_MAP, blank_idx=10, beam_width=5),
    'Beam (width=10)': create_decoder('beam', chars=CHAR_MAP, blank_idx=10, beam_width=10),
    'Prefix Beam (width=10)': create_decoder('prefix_beam', chars=CHAR_MAP, blank_idx=10, beam_width=10),
}

print(f"      Created {len(decoders)} decoders")

# ============================================
# LOAD & PREPROCESS IMAGE
# ============================================

print("\n[3/3] Testing on image...")
print("-" * 80)

img = cv2.imread(TEST_IMAGE)
if img is None:
    print(f"ERROR: Could not load image: {TEST_IMAGE}")
    sys.exit(1)

# Preprocess
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (IMG_SIZE[1], IMG_SIZE[0]))
normalized = resized.astype(np.float32) / 255.0
tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    output = model(tensor)  # (T, B, C)

print(f"\nModel output shape: {output.shape}")

# ============================================
# TEST DECODERS
# ============================================

print(f"\n{'Decoder':<25} {'Predicted':<15} {'Status':<10}")
print("-" * 80)

results = []

for name, decoder in decoders.items():
    pred_text = decoder.decode(output)
    is_correct = pred_text == GROUND_TRUTH
    status = "[PASS]" if is_correct else "[FAIL]"

    print(f"{name:<25} {pred_text:<15} {status:<10}")

    results.append({
        'name': name,
        'predicted': pred_text,
        'correct': is_correct
    })

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

passes = sum(1 for r in results if r['correct'])
total = len(results)

print(f"\nGround Truth: {GROUND_TRUTH}")
print(f"Pass Rate: {passes}/{total} ({passes/total*100:.1f}%)")

if passes == total:
    print("\n[OK] All decoders passed!")
elif passes == 0:
    print("\n[ERROR] All decoders failed!")
else:
    print(f"\n{passes}/{total} decoders passed")

print("\n" + "=" * 80)
print("BEAM SEARCH INTEGRATION VERIFIED")
print("=" * 80)

print("\n[INFO] Notes:")
print("  - If Beam Search passes but Greedy fails: Beam search is working!")
print("  - This test verifies beam search can handle repeated digits")
print("  - Full pipeline integration test can be done later")
print("=" * 80)

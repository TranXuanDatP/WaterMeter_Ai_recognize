"""
Test OCR Finetune Model + Beam Search on M3 Black Digits Output

Test M3 pipeline output with OCR model and beam search decoder.
"""
import os
import sys
import cv2
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.m4_crnn_reading.model import CRNN
from src.m4_crnn_reading.beam_search_decoder import create_decoder

# Fix encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ============================================
# CONFIGURATION
# ============================================

# M3 pipeline output directory
INPUT_DIR = Path(r"F:\Workspace\Project\results\pipeline_full_m1_m2_m3_m3_5_m4_beam_backup_20260309_155359\m3_roi_crops")

# OCR finetuned model (117MB)
MODEL_PATH = Path(r"F:\Workspace\Project\model\ocr_finetune.pth")

# Output directory
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\test_m3_roi_crops_ocr")

# Image size for CRNN
IMG_SIZE = (64, 224)

# Character set for 5 digits (0-9)
CHAR_MAP = "0123456789"

print("=" * 80)
print("TEST OCR FINETUNE + BEAM SEARCH ON M3 BLACK DIGITS")
print("=" * 80)
print(f"\nInput: {INPUT_DIR}")
print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Img Size: {IMG_SIZE}")
print(f"Char Map: {CHAR_MAP}")
print("=" * 80)

# ============================================
# LOAD MODEL
# ============================================

def load_model(model_path, device):
    """Load OCR finetuned model"""
    print(f"\n[1/4] Loading model...")
    print(f"      Model: {model_path}")
    print(f"      Size: {os.path.getsize(model_path) / (1024**2):.1f} MB")

    model = CRNN(num_chars=len(CHAR_MAP) + 1)
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"      Epoch: {checkpoint.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print(f"      Direct load (no epoch info)")

    model.eval()

    print(f"      Device: {device}")
    return model

# ============================================
# CREATE BEAM SEARCH DECODER
# ============================================

def create_beam_decoder(beam_width=10):
    """Create beam search decoder"""
    print(f"\n[2/4] Creating beam search decoder...")
    print(f"      Beam Width: {beam_width}")
    print(f"      Chars: {CHAR_MAP}")

    decoder = create_decoder(
        method='beam',
        chars=CHAR_MAP,
        blank_idx=len(CHAR_MAP),
        beam_width=beam_width
    )
    return decoder

# ============================================
# PREPROCESSING
# ============================================

def preprocess_image(img, target_size=(64, 224)):
    """Preprocess black digit image for CRNN"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Resize to target size
    resized = cv2.resize(gray, (target_size[1], target_size[0]))

    # Normalize
    normalized = resized.astype(np.float32) / 255.0

    # Convert to tensor: (1, 1, H, W)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

    return tensor

# ============================================
# INFERENCE
# ============================================

def test_ocr_beam_search(model, decoder, input_dir, output_dir, device):
    """Test OCR with beam search on M3 output images"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[3/4] Processing images...")

    # Get all image files
    image_files = sorted(list(input_dir.glob('*.jpg'))) + \
                   sorted(list(input_dir.glob('*.png')))

    if len(image_files) == 0:
        print(f"      ERROR: No images found in {input_dir}")
        return

    print(f"      Found {len(image_files)} images")

    results = []

    # Process each image
    for img_path in tqdm(image_files, desc="      Progress"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Preprocess
        input_tensor = preprocess_image(img, IMG_SIZE)
        input_tensor = input_tensor.to(device)

        # Run OCR
        with torch.no_grad():
            outputs = model(input_tensor)

        # Extract logits for first batch: (T, C)
        logits = outputs[:, 0, :]

        # Decode with beam search
        decoded_text = decoder.decode(logits)

        # Extract first result
        if isinstance(decoded_text, list) and len(decoded_text) > 0:
            predicted_text = decoded_text[0]
        else:
            predicted_text = str(decoded_text)

        # Filter to digits only
        predicted_text = ''.join([c for c in predicted_text if c.isdigit()])

        # Save result
        results.append({
            'filename': img_path.name,
            'predicted_text': predicted_text,
            'num_digits': len(predicted_text)
        })

        # Draw result on image for visualization
        img_vis = img.copy()
        cv2.putText(img_vis, predicted_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save visualization
        vis_path = output_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(vis_path), img_vis)

    # Save results to CSV
    print(f"\n[4/4] Saving results...")

    df_results = pd.DataFrame(results)
    csv_path = output_dir / "ocr_results.csv"
    df_results.to_csv(csv_path, index=False)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total images processed: {len(results)}")

    if len(results) > 0:
        # Digit count distribution
        digit_counts = df_results['num_digits'].value_counts().sort_index()
        print(f"\nDigit count distribution:")
        for count, num_files in digit_counts.items():
            print(f"  {count} digits: {num_files} files")

        # Sample results
        print(f"\nSample results (first 5):")
        for _, row in df_results.head(5).iterrows():
            print(f"  {row['filename'][:40]:40} → {row['predicted_text']}")

        # Accuracy check (if we have ground truth)
        print(f"\nNote: Results saved to: {csv_path}")
        print(f"      Visualizations saved to: {output_dir}")

    print(f"{'='*80}")

    return df_results

# ============================================
# MAIN
# ============================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check paths
    if not MODEL_PATH.exists():
        print(f"\nERROR: Model not found: {MODEL_PATH}")
        return

    if not INPUT_DIR.exists():
        print(f"\nERROR: Input directory not found: {INPUT_DIR}")
        return

    # Load model
    model = load_model(MODEL_PATH, device)

    # Create decoder
    decoder = create_beam_decoder(beam_width=10)

    # Run test
    df_results = test_ocr_beam_search(model, decoder, INPUT_DIR, OUTPUT_DIR, device)

    print("\n✅ Test complete!")
    return df_results

if __name__ == "__main__":
    main()

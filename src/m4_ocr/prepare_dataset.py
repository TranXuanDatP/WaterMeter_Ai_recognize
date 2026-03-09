"""
Prepare M4 OCR Dataset from data_4digit.csv
Convert labels to OCR training format
"""
import os
import sys
import codecs
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Config
# Sử dụng file data_4digit_fixed.csv với zero-padding
LABELS_FILE = r"F:\Workspace\Project\data\data_4digit_fixed.csv"
ROI_IMAGES_DIR = r"F:\Workspace\Project\data\m5_black_digits"
OUTPUT_DIR = r"F:\Workspace\Project\data\m4_ocr_dataset_black_digits"
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "labels.csv")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("M4: PREPARE OCR DATASET (BLACK DIGITS ONLY)")
print("=" * 70)

# Read labels
print(f"\n[LOAD] Reading labels from: {LABELS_FILE}")
# QUAN TRỌNG: Đọc cột 'value' dưới dạng string để giữ zero-padding
df_labels = pd.read_csv(LABELS_FILE, dtype={'value': str})
print(f"[LOAD] Loaded {len(df_labels)} labels")
print(f"[LOAD] Value column dtype: {df_labels['value'].dtype}")

# Verify labels (đã được fix sẵn trong data_4digit_fixed.csv)
print(f"\n[VERIFY] Verifying labels from data_4digit_fixed.csv...")
print(f"[VERIFY] Sample values: {df_labels['value'].head().tolist()}")

# Labels đã được format sẵn trong data_4digit_fixed.csv, chỉ cần copy sang cột 'text'
df_labels['text'] = df_labels['value'].astype(str)

print(f"[VERIFY] Sample text: {df_labels['text'].head().tolist()}")

# Validate labels
text_lengths = df_labels['text'].apply(len)
invalid_labels = df_labels[text_lengths != 4]

if len(invalid_labels) > 0:
    print(f"\n⚠️  WARNING: Found {len(invalid_labels)} labels that are not 4 digits!")
    print(f"[VALIDATE] First 10 invalid labels:")
    for i, row in invalid_labels.head(10).iterrows():
        print(f"  Value: {row['value']} -> Text: '{row['text']}' (length: {len(row['text'])})")
else:
    print(f"[VERIFY] ✅ All {len(df_labels)} labels are correctly formatted as 4-digit strings!")

# Show distribution
print(f"\n[STATS] Label length distribution:")
for length in sorted(text_lengths.unique()):
    count = (text_lengths == length).sum()
    percentage = count / len(df_labels) * 100
    status = "✅" if length == 4 else "❌"
    print(f"  {status} {length} digits: {count:5d} ({percentage:5.1f}%)")

# Map filename: meter4_* -> crop_meter4_*
def map_filename(original_name):
    """Map original filename to ROI crop filename"""
    # Remove extension
    name_without_ext = os.path.splitext(original_name)[0]

    # Add 'crop_' prefix
    return f"crop_{name_without_ext}.jpg"

df_labels['roi_filename'] = df_labels['photo_name'].apply(map_filename)

# Check which files exist in ROI directory
print(f"\n[CHECK] Checking ROI files...")
roi_files = set(os.listdir(ROI_IMAGES_DIR))
df_labels['exists'] = df_labels['roi_filename'].apply(
    lambda f: f in roi_files
)

# Filter only existing files
df_matched = df_labels[df_labels['exists'] == True].copy()

print(f"[MATCH] Matched {len(df_matched)} ROI files with labels")
print(f"[MISSING] {len(df_labels) - len(df_matched)} files not found in ROI")

# Create labels CSV for OCR training
df_output = df_matched[['roi_filename', 'text']].copy()
df_output = df_output.rename(columns={'roi_filename': 'filename'})

# Save labels
df_output.to_csv(OUTPUT_LABELS, index=False)

print(f"\n[SAVE] Saved labels to: {OUTPUT_LABELS}")

# Statistics
print(f"\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

print(f"\nTotal samples: {len(df_output)}")

# Text length distribution
text_lengths = df_output['text'].apply(len)
print(f"\nText length distribution:")
for length in sorted(text_lengths.unique()):
    count = (text_lengths == length).sum()
    percentage = count / len(df_output) * 100
    print(f"  {length} digits: {count:5d} ({percentage:5.1f}%)")

# Sample labels
print(f"\n[SAMPLE LABELS] (First 10)")
print("-" * 70)
for i, row in df_output.head(10).iterrows():
    print(f"  {row['filename'][:60]:60s} | {row['text']}")

# Value distribution
print(f"\n[VALUE DISTRIBUTION]")
value_ranges = {
    '0000-0999': ((df_labels['value'] >= '0000') & (df_labels['value'] < '1000')).sum(),
    '1000-1999': ((df_labels['value'] >= '1000') & (df_labels['value'] < '2000')).sum(),
    '2000-2999': ((df_labels['value'] >= '2000') & (df_labels['value'] < '3000')).sum(),
    '3000-3999': ((df_labels['value'] >= '3000') & (df_labels['value'] < '4000')).sum(),
    '4000-4999': ((df_labels['value'] >= '4000') & (df_labels['value'] < '5000')).sum(),
    '5000-5999': ((df_labels['value'] >= '5000') & (df_labels['value'] < '6000')).sum(),
    '6000-6999': ((df_labels['value'] >= '6000') & (df_labels['value'] < '7000')).sum(),
    '7000-7999': ((df_labels['value'] >= '7000') & (df_labels['value'] < '8000')).sum(),
    '8000-8999': ((df_labels['value'] >= '8000') & (df_labels['value'] < '9000')).sum(),
    '9000-9999': ((df_labels['value'] >= '9000') & (df_labels['value'] <= '9999')).sum(),
}

for range_name, count in value_ranges.items():
    percentage = count / len(df_labels) * 100
    bar = '█' * int(percentage / 5)
    print(f"  {range_name:12s}: {count:5d} ({percentage:5.1f}%) {bar}")

# Copy images to output directory
print(f"\n[COPY] Copying images to output directory...")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

copied_count = 0
for _, row in tqdm(df_output.iterrows(), total=len(df_output), desc="Copying images"):
    src_path = os.path.join(ROI_IMAGES_DIR, row['filename'])
    dst_path = os.path.join(OUTPUT_IMG_DIR, row['filename'])

    if os.path.exists(src_path):
        import shutil
        shutil.copy2(src_path, dst_path)
        copied_count += 1

print(f"\n[COPY] Copied {copied_count} images to {OUTPUT_IMG_DIR}")

print("\n" + "=" * 70)
print("✅ DATASET PREPARATION COMPLETED!")
print("=" * 70)
print(f"\n📁 Dataset location: {OUTPUT_DIR}")
print(f"   Images: {OUTPUT_IMG_DIR}/")
print(f"   Labels: {OUTPUT_LABELS}")
print(f"\n📊 Dataset Info:")
print(f"   Total samples: {len(df_output):,}")
print(f"   Image source: M5 (Black digits only, no red decimal digits)")
print(f"   Label format: 4-digit strings (e.g., '0187' for 187)")
print(f"   Image size: Variable (cropped to ~80% width)")

print(f"\n💡 Next steps:")
print(f"   1. ZIP the folder: {OUTPUT_DIR}")
print(f"   2. Upload to Google Drive")
print(f"   3. Run M4_OCR_Training_BlackDigits.ipynb in Colab")

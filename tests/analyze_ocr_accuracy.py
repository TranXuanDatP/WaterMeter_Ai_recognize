"""
Analyze OCR Accuracy against Ground Truth (Result_38.csv)
"""
import pandas as pd
from pathlib import Path

# Paths
OCR_RESULTS = Path(r"F:\Workspace\Project\results\test_m3_5_ocr_beam_search\ocr_results.csv")
GROUND_TRUTH = Path(r"F:\Workspace\Project\data\Result_40.csv")

print("=" * 80)
print("OCR ACCURACY ANALYSIS")
print("=" * 80)

# Load OCR results
df_ocr = pd.read_csv(OCR_RESULTS)
print(f"\nOCR Results: {len(df_ocr)} records")

# Load ground truth
df_gt = pd.read_csv(GROUND_TRUTH, header=None, names=['reading', 'url'])
print(f"Ground Truth: {len(df_gt)} records")

# Extract index from filename
# Format: meter4_00000_00385501ab4d419fa7b0bdf0d9f8451f.jpg
# Map by index position (OCR results are sorted by filename)
df_ocr['file_index'] = df_ocr['filename'].str.extract(r'meter4_(\d{5})_')[0].astype(int)

# Ground truth is already sorted by index (0 to 6484)
# We need to map OCR index to GT index
# OCR has 6282 files, GT has 6485 files
# Map by row index since both are sorted
df_ocr = df_ocr.reset_index(drop=True)
df_gt = df_gt.reset_index(drop=True)

# Merge by row index
df_merged = pd.concat([
    df_ocr[['filename', 'predicted_text', 'file_index']].reset_index(drop=True),
    df_gt[['reading']].reset_index(drop=True)
], axis=1)
df_merged = df_merged[df_merged['file_index'] < len(df_gt)]

print(f"Merged records: {len(df_merged)}")

# Ensure predicted_text is string
df_merged['predicted_text'] = df_merged['predicted_text'].astype(str)
df_merged['predicted_value'] = pd.to_numeric(df_merged['predicted_text'], errors='coerce').fillna(0).astype(int)

# Calculate accuracy
df_merged['is_correct'] = (df_merged['predicted_value'] == df_merged['reading'])
accuracy = df_merged['is_correct'].mean()

# Digit-level accuracy
df_merged['pred_digits'] = df_merged['predicted_text'].str.len()
df_merged['gt_digits'] = df_merged['reading'].astype(str).str.len()
df_merged['digit_match'] = (df_merged['pred_digits'] == df_merged['gt_digits'])
digit_accuracy = df_merged['digit_match'].mean()

print(f"\n{'='*80}")
print("ACCURACY RESULTS")
print(f"{'='*80}")
print(f"Exact Match Accuracy: {accuracy*100:.2f}% ({df_merged['is_correct'].sum()}/{len(df_merged)})")
print(f"Digit Length Match:   {digit_accuracy*100:.2f}% ({df_merged['digit_match'].sum()}/{len(df_merged)})")

# Error analysis
df_errors = df_merged[~df_merged['is_correct']].copy()
print(f"\nError Analysis: {len(df_errors)} errors ({len(df_errors)/len(df_merged)*100:.1f}%)")

if len(df_errors) > 0:
    # Error by digit length
    print(f"\nErrors by Ground Truth Digit Length:")
    for length in sorted(df_errors['gt_digits'].unique()):
        count = (df_errors['gt_digits'] == length).sum()
        total = (df_merged['gt_digits'] == length).sum()
        pct = count / total * 100 if total > 0 else 0
        print(f"  {length} digits: {count} errors / {total} total ({pct:.1f}%)")

    # Sample errors
    print(f"\nSample Errors (first 10):")
    for i, row in df_errors.head(10).iterrows():
        print(f"  GT: {row['reading']:4d} ({row['gt_digits']}d) | Pred: {row['predicted_value']:4d} ({row['pred_digits']}d) | {row['filename'][:40]}")

# Correct samples
print(f"\nSample Correct (first 10):")
df_correct = df_merged[df_merged['is_correct']].head(10)
for i, row in df_correct.iterrows():
    print(f"  GT: {row['reading']:4d} | Pred: {row['predicted_value']:4d} | {row['filename'][:40]}")

print(f"\n{'='*80}")

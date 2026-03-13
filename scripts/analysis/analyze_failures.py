"""
Analyze pipeline failures and create visualizations of incorrect predictions
"""
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import shutil

# Paths
RESULTS_DIR = Path(r"F:\Workspace\Project\results\pipeline_full_m1_m2_m3_m3_5_m4_beam")
DATA_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
OUTPUT_DIR = Path(r"F:\Workspace\Project\results\failure_analysis")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)
ERROR_IMAGES_DIR = OUTPUT_DIR / "error_images"
ERROR_IMAGES_DIR.mkdir(exist_ok=True)

# Load results
results_df = pd.read_csv(RESULTS_DIR / "pipeline_results.csv")

# Separate correct and incorrect
correct_df = results_df[results_df['correct'] == True]
incorrect_df = results_df[results_df['correct'] == False]

print("="*70)
print("FAILURE ANALYSIS")
print("="*70)
print(f"\nTotal predictions: {len(results_df)}")
print(f"Correct: {len(correct_df)} ({len(correct_df)/len(results_df)*100:.2f}%)")
print(f"Incorrect: {len(incorrect_df)} ({len(incorrect_df)/len(results_df)*100:.2f}%)")

# Analyze error types
error_types = {
    'wrong_digits': [],
    'missing_digits': [],
    'extra_digits': [],
    'completely_wrong': []
}

for idx, row in incorrect_df.iterrows():
    true_val = str(row['true_value'])
    pred_val = str(row['predicted_value'])

    if len(pred_val) < len(true_val):
        error_types['missing_digits'].append(row)
    elif len(pred_val) > len(true_val):
        error_types['extra_digits'].append(row)
    elif len(pred_val) == len(true_val):
        # Check how many digits are wrong
        wrong_count = sum(1 for t, p in zip(true_val, pred_val) if t != p)
        if wrong_count == len(true_val):
            error_types['completely_wrong'].append(row)
        else:
            error_types['wrong_digits'].append(row)

print(f"\nError Types:")
print(f"  Missing digits: {len(error_types['missing_digits'])}")
print(f"  Extra digits: {len(error_types['extra_digits'])}")
print(f"  Wrong digits: {len(error_types['wrong_digits'])}")
print(f"  Completely wrong: {len(error_types['completely_wrong'])}")

# Copy and visualize first 20 incorrect images
print(f"\nCopying incorrect images for analysis...")
sample_size = min(20, len(incorrect_df))
sample_errors = incorrect_df.head(sample_size)

for idx, row in sample_errors.iterrows():
    filename = row['filename']
    true_val = row['true_value']
    pred_val = row['predicted_value']

    # Copy original image
    src_path = DATA_DIR / filename
    if src_path.exists():
        dst_path = ERROR_IMAGES_DIR / f"{idx:04d}_{true_val}_pred_{pred_val}_{filename}"
        shutil.copy(src_path, dst_path)

        # Create visualization with prediction
        img = cv2.imread(str(src_path))
        if img is not None:
            # Add text overlay
            h, w = img.shape[:2]
            overlay = img.copy()

            # Add semi-transparent box at top
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

            # Add text
            cv2.putText(img, f"True: {true_val}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Pred: {pred_val}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Save visualization
            vis_path = ERROR_IMAGES_DIR / f"{idx:04d}_{true_val}_pred_{pred_val}_vis.jpg"
            cv2.imwrite(str(vis_path), img)

# Calculate digit-wise accuracy
print(f"\nDigit-wise Analysis:")
all_true = []
all_pred = []

for idx, row in incorrect_df.iterrows():
    true_val = str(row['true_value'])
    pred_val = str(row['predicted_value'])

    # Pad shorter string with spaces
    max_len = max(len(true_val), len(pred_val))
    true_val = true_val.ljust(max_len, ' ')
    pred_val = pred_val.ljust(max_len, ' ')

    for t, p in zip(true_val, pred_val):
        if t != ' ' and p != ' ':  # Only compare actual digits
            all_true.append(t)
            all_pred.append(p)

digit_accuracy = sum(1 for t, p in zip(all_true, all_pred) if t == p) / len(all_true) * 100
print(f"  Individual digit accuracy: {digit_accuracy:.2f}%")

# Most common error patterns
from collections import Counter
errors = []

for idx, row in incorrect_df.iterrows():
    true_val = str(row['true_value'])
    pred_val = str(row['predicted_value'])

    max_len = max(len(true_val), len(pred_val))
    true_val = true_val.ljust(max_len, ' ')
    pred_val = pred_val.ljust(max_len, ' ')

    for i, (t, p) in enumerate(zip(true_val, pred_val)):
        if t != ' ' and p != ' ' and t != p:
            errors.append(f"{t}->{p}")

error_counter = Counter(errors)
print(f"\nMost common errors:")
for error, count in error_counter.most_common(10):
    print(f"  {error}: {count} times")

print(f"\n{'='*70}")
print(f"Analysis complete!")
print(f"Error images saved to: {ERROR_IMAGES_DIR}")
print(f"{'='*70}")

# Save summary
summary = {
    'total_predictions': len(results_df),
    'correct': len(correct_df),
    'incorrect': len(incorrect_df),
    'accuracy': len(correct_df) / len(results_df) * 100,
    'error_types': {
        'missing_digits': len(error_types['missing_digits']),
        'extra_digits': len(error_types['extra_digits']),
        'wrong_digits': len(error_types['wrong_digits']),
        'completely_wrong': len(error_types['completely_wrong'])
    },
    'digit_accuracy': digit_accuracy,
    'most_common_errors': dict(error_counter.most_common(10))
}

import json
with open(OUTPUT_DIR / "failure_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {OUTPUT_DIR / 'failure_summary.json'}")

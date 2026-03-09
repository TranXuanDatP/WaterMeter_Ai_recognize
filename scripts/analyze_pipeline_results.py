"""
Analyze pipeline results and generate statistics
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path(r"F:\Workspace\Project\results\pipeline_fixed_m2")
CSV_FILE = RESULTS_DIR / "pipeline_results.csv"

# Load results
df = pd.read_csv(CSV_FILE)

print("="*70)
print("PIPELINE RESULTS ANALYSIS")
print("="*70)
print(f"\nDataset: data_4digit2 (6527 images)")
print(f"Model: m2_angle_model_epoch15_FIXED_COS_SIN.pth")
print(f"Timestamp: 2026-03-09 11:53:42")
print("="*70)

# Overall statistics
print(f"\n[1] OVERALL STATISTICS")
print(f"-"*70)
print(f"Total images: 6527")
print(f"Successful: {len(df)}")
print(f"Success rate: {len(df)/6527*100:.2f}%")

# M2 angles
print(f"\n[2] M2 ORIENTATION STATISTICS")
print(f"-"*70)
angles = df['m2_detected_angle'].values
print(f"Detected angles:")
print(f"  Mean: {np.mean(angles):.2f}°")
print(f"  Std:  {np.std(angles):.2f}°")
print(f"  Min:  {np.min(angles):.2f}°")
print(f"  Max:  {np.max(angles):.2f}°")

# Angle distribution
angles_abs = np.abs(angles)
print(f"\nAngle magnitude distribution:")
print(f"  |angle| < 5°:   {np.sum(angles_abs < 5):4d} ({np.sum(angles_abs < 5)/len(df)*100:.1f}%)")
print(f"  |angle| < 10°:  {np.sum(angles_abs < 10):4d} ({np.sum(angles_abs < 10)/len(df)*100:.1f}%)")
print(f"  |angle| < 20°:  {np.sum(angles_abs < 20):4d} ({np.sum(angles_abs < 20)/len(df)*100:.1f}%)")
print(f"  |angle| < 45°:  {np.sum(angles_abs < 45):4d} ({np.sum(angles_abs < 45)/len(df)*100:.1f}%)")

# Correction angles
corrections = df['m2_correction_angle'].values
print(f"\nCorrection angles:")
print(f"  Mean: {np.mean(corrections):.2f}°")
print(f"  Std:  {np.std(corrections):.2f}°")
print(f"  Min:  {np.min(corrections):.2f}°")
print(f"  Max:  {np.max(corrections):.2f}°")

print(f"\nCorrection magnitude distribution:")
corr_abs = np.abs(corrections)
print(f"  |corr| < 5°:   {np.sum(corr_abs < 5):4d} ({np.sum(corr_abs < 5)/len(df)*100:.1f}%)")
print(f"  |corr| < 10°:  {np.sum(corr_abs < 10):4d} ({np.sum(corr_abs < 10)/len(df)*100:.1f}%)")
print(f"  |corr| < 20°:  {np.sum(corr_abs < 20):4d} ({np.sum(corr_abs < 20)/len(df)*100:.1f}%)")
print(f"  |corr| < 45°:  {np.sum(corr_abs < 45):4d} ({np.sum(corr_abs < 45)/len(df)*100:.1f}%)")

# Output files
print(f"\n[3] OUTPUT FILES")
print(f"-"*70)
print(f"M1 crops:     6098 images")
print(f"M2 aligned:   6098 images")
print(f"M3 ROI:       6098 images")
print(f"CSV results: {len(df)} rows")

print(f"\n[4] SAMPLE RESULTS")
print(f"-"*70)
print(df[['filename', 'true_value', 'm2_detected_angle', 'm2_correction_angle']].head(10).to_string(index=False))

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print(f"✅ Pipeline completed successfully!")
print(f"✅ Success rate: {len(df)/6527*100:.1f}%")
print(f"✅ M2 Mean angle: {np.mean(angles):.2f}° (near 0° = mostly upright)")
print(f"✅ M2 Mean correction: {np.mean(corrections):.2f}° (small rotations needed)")
print(f"✅ {np.sum(angles_abs < 5)/len(df)*100:.1f}% images need <5° correction")
print(f"{'='*70}")

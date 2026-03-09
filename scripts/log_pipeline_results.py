"""
Pipeline Results Logger

Logs detailed results from each stage of the meter reading pipeline.
Useful for debugging and analyzing pipeline performance.
"""
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ============================================
# CONFIGURATION
# ============================================

# Results directories
RESULTS_DIR = Path(r"F:\Workspace\Project\results\test_pipeline")

# Stage outputs
M1_CROPS_DIR = RESULTS_DIR / "m1_crops"
M2_ALIGNED_DIR = RESULTS_DIR / "m2_aligned"
M3_ROI_DIR = RESULTS_DIR / "m3_roi_crops"
M3_5_DIGITS_DIR = RESULTS_DIR / "m3_5_black_digits"
M4_RESULTS_DIR = RESULTS_DIR / "m4_ocr_results"
M4_BEAM_DIR = RESULTS_DIR / "m4_ocr_beam_search"

# Output
LOG_DIR = RESULTS_DIR / "pipeline_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PIPELINE RESULTS LOGGER")
print("=" * 80)
print(f"Results dir: {RESULTS_DIR}")
print(f"Log dir: {LOG_DIR}")
print("=" * 80)

# ============================================
# LOAD RESULTS FROM EACH STAGE
# ============================================

def load_stage_results(stage_dir, stage_name):
    """Load results from a pipeline stage"""
    results = {}

    if not stage_dir.exists():
        print(f"  [WARNING] {stage_name} directory not found: {stage_dir}")
        return results

    # Count files
    files = list(stage_dir.glob('*.jpg')) + list(stage_dir.glob('*.png'))
    results['file_count'] = len(files)
    results['files'] = [f.name for f in files]

    print(f"  {stage_name}: {len(files)} files")

    return results

def load_csv_results(csv_path, stage_name):
    """Load results from CSV file"""
    results = {}

    if not csv_path.exists():
        print(f"  [WARNING] {stage_name} CSV not found: {csv_path}")
        return results

    try:
        df = pd.read_csv(csv_path)
        results['row_count'] = len(df)
        results['success_count'] = len(df[df['status'] == 'success'])
        results['error_count'] = len(df[df['status'] == 'error'])

        if 'status' in df.columns:
            success_rate = results['success_count'] / len(df) * 100
            print(f"  {stage_name}: {len(df)} rows ({results['success_count']} success, {results['error_count']} errors) - {success_rate:.1f}% success")
        else:
            print(f"  {stage_name}: {len(df)} rows")

        return results, df
    except Exception as e:
        print(f"  [ERROR] Failed to load {stage_name}: {e}")
        return {}, None

print(f"\n[1/4] Scanning pipeline stages...")
print("-" * 80)

# Stage 1: M1 - Water Meter Detection
m1_results = load_stage_results(M1_CROPS_DIR, "M1 (Detection)")

# Stage 2: M2 - Alignment
m2_results = load_stage_results(M2_ALIGNED_DIR, "M2 (Alignment)")

# Stage 3: M3 - ROI Detection
m3_results = load_stage_results(M3_ROI_DIR, "M3 (ROI Detection)")

# Stage 3.5: Black Digit Extraction
m3_5_results = load_stage_results(M3_5_DIGITS_DIR, "M3.5 (Black Digits)")

# Stage 4: M4 - OCR (Greedy)
m4_csv = M4_RESULTS_DIR / "ocr_results.csv"
m4_results, m4_df = load_csv_results(m4_csv, "M4 (OCR - Greedy)")

# Stage 4: M4 - OCR (Beam Search)
m4_beam_csv = M4_BEAM_DIR / "ocr_results.csv"

# Handle beam search results safely
if m4_beam_csv.exists():
    try:
        m4_beam_results, m4_beam_df = load_csv_results(m4_beam_csv, "M4 (OCR - Beam Search)")
    except Exception as e:
        print(f"  [ERROR] Could not load beam search results: {e}")
        m4_beam_results = {}
        m4_beam_df = None
else:
    m4_beam_results = {}
    m4_beam_df = None

# ============================================
# COMPARE GREEDY vs BEAM SEARCH
# ============================================

print(f"\n[2/4] Comparing Greedy vs Beam Search...")
print("-" * 80)

if m4_df is not None and m4_beam_df is not None:
    # Compare success rates
    greedy_success = m4_results['success_count']
    greedy_total = m4_results['row_count']
    greedy_rate = greedy_success / greedy_total * 100

    beam_success = m4_beam_results.get('success_count', 0)
    beam_total = m4_beam_results.get('row_count', 0)
    beam_rate = beam_success / beam_total * 100 if beam_total > 0 else 0

    print(f"\nSuccess Rate Comparison:")
    print(f"  Greedy:  {greedy_success}/{greedy_total} ({greedy_rate:.1f}%)")
    print(f"  Beam:    {beam_success}/{beam_total} ({beam_rate:.1f}%)")
    print(f"  Improvement: +{beam_rate - greedy_rate:.1f}%")

    # Compare confidence scores
    if 'confidence' in m4_df.columns and 'confidence' in m4_beam_df.columns:
        greedy_conf = m4_df[m4_df['status'] == 'success']['confidence'].mean()
        beam_conf = m4_beam_df[m4_beam_df['status'] == 'success']['confidence'].mean()

        print(f"\nAverage Confidence:")
        print(f"  Greedy:  {greedy_conf:.4f}")
        print(f"  Beam:    {beam_conf:.4f}")

    # Find cases where beam search fixed errors
    if 'predicted_text' in m4_df.columns and 'predicted_text' in m4_beam_df.columns:
        # This would need ground truth labels for proper comparison
        print(f"\nNote: To see which cases were fixed, run compare_m4_decoders.py with ground truth labels")
else:
    print("  Cannot compare - missing one or both result files")

# ============================================
# GENERATE SUMMARY REPORT
# ============================================

print(f"\n[3/4] Generating summary report...")

summary = {
    'timestamp': datetime.now().isoformat(),
    'pipeline_stages': {
        'M1': m1_results,
        'M2': m2_results,
        'M3': m3_results,
        'M3.5': m3_5_results,
        'M4_Greedy': m4_results,
        'M4_BeamSearch': m4_beam_results,
    },
    'comparison': {
        'greedy_success_rate': greedy_success / greedy_total * 100 if m4_df is not None else 0,
        'beam_success_rate': beam_success / beam_total * 100 if m4_beam_df is not None else 0,
        'improvement': beam_rate - greedy_rate if m4_df is not None and m4_beam_df is not None else 0,
    } if m4_df is not None and m4_beam_df is not None else {}
}

# Save summary as JSON
summary_json = LOG_DIR / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"  Summary saved to: {summary_json}")

# ============================================
# CREATE DETAILED REPORT
# ============================================

print(f"\n[4/4] Creating detailed report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("PIPELINE EXECUTION REPORT")
report_lines.append("=" * 80)
report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"\nPipeline Stages:")
report_lines.append(f"  M1:  Water Meter Detection")
report_lines.append(f"  M2:  Image Alignment")
report_lines.append(f"  M3:  ROI Detection")
report_lines.append(f"  M3.5: Black Digit Extraction")
report_lines.append(f"  M4:  OCR Reading (Greedy + Beam Search)")
report_lines.append(f"\n" + "=" * 80)

# Stage statistics
report_lines.append(f"\nSTAGE STATISTICS:")
report_lines.append(f"-" * 80)
report_lines.append(f"{'Stage':<20} {'Files':<10} {'Status':<30}")
report_lines.append(f"-" * 80)

stages_info = [
    ("M1 (Detection)", m1_results.get('file_count', 0), "OK" if m1_results else "NOT FOUND"),
    ("M2 (Alignment)", m2_results.get('file_count', 0), "OK" if m2_results else "NOT FOUND"),
    ("M3 (ROI)", m3_results.get('file_count', 0), "OK" if m3_results else "NOT FOUND"),
    ("M3.5 (Black Digits)", m3_5_results.get('file_count', 0), "OK" if m3_5_results else "NOT FOUND"),
]

for stage_name, count, status in stages_info:
    report_lines.append(f"{stage_name:<20} {count:<10} {status:<30}")

# OCR comparison
if m4_df is not None and m4_beam_df is not None:
    report_lines.append(f"\n" + "=" * 80)
    report_lines.append(f"OCR DECODER COMPARISON:")
    report_lines.append(f"-" * 80)

    report_lines.append(f"\nGreedy Decoder:")
    report_lines.append(f"  Total:     {m4_results['row_count']}")
    report_lines.append(f"  Success:   {m4_results['success_count']} ({greedy_rate:.1f}%)")
    report_lines.append(f"  Errors:    {m4_results['error_count']}")

    report_lines.append(f"\nBeam Search Decoder:")
    report_lines.append(f"  Total:     {m4_beam_results['row_count']}")
    report_lines.append(f"  Success:   {m4_beam_results['success_count']} ({beam_rate:.1f}%)")
    report_lines.append(f"  Errors:    {m4_beam_results['error_count']}")

    report_lines.append(f"\nImprovement: +{beam_rate - greedy_rate:.1f}%")

# Recommendations
report_lines.append(f"\n" + "=" * 80)
report_lines.append(f"RECOMMENDATIONS:")
report_lines.append(f"-" * 80)

if beam_rate > greedy_rate + 5:
    report_lines.append(f"  [OK] Beam search shows significant improvement")
    report_lines.append(f"       → Use beam search decoder in production")
elif beam_rate > greedy_rate:
    report_lines.append(f"  [INFO] Beam search shows moderate improvement")
    report_lines.append(f"       → Consider using beam search for better accuracy")
else:
    report_lines.append(f"  [WARN] Beam search does not show improvement")
    report_lines.append(f"       → Investigate beam width and decoder parameters")

report_lines.append(f"\n" + "=" * 80)
report_lines.append(f"END OF REPORT")
report_lines.append(f"=" * 80)

# Save report
report_txt = LOG_DIR / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"  Report saved to: {report_txt}")

# Print report to console
print("\n")
for line in report_lines:
    print(line)

print("\n" + "=" * 80)
print("[OK] PIPELINE LOGGING COMPLETED!")
print("=" * 80)

print(f"\nLog files:")
print(f"  Summary JSON: {summary_json.name}")
print(f"  Report TXT:   {report_txt.name}")
print("=" * 80)

print("\n💡 Next steps:")
print("  1. Review the pipeline report")
print("  2. Check for any failed stages")
print("  3. Use beam search decoder if it shows improvement")
print("  4. Investigate any errors in individual stages")
print("=" * 80)

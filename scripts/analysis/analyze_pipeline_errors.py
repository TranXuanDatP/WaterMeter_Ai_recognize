#!/usr/bin/env python3
"""
Pipeline Error Analysis Tool

Phân tích chi tiết lỗi từ pipeline results:
1. Phân loại lỗi theo độ lệch (off-by-1, off-by-2-5, large errors)
2. Phân tích lỗi theo confidence score thấp
3. Phân tích lỗi theo stage (M1, M2, M3, M4)
4. Export report để debug

Usage:
    python analyze_pipeline_errors.py --csv results/pipeline_detailed_results.csv
    python analyze_pipeline_errors.py --csv results/pipeline_results.csv --output analysis/
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import argparse

# Fix encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


class PipelineErrorAnalyzer:
    """Analyzer for pipeline errors"""

    def __init__(self, csv_path: str, output_dir: str = None):
        """
        Initialize analyzer

        Args:
            csv_path: Path to pipeline results CSV
            output_dir: Directory to save analysis reports
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir else self.csv_path.parent / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print(f"Loading results from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} records\n")

        # Analysis results
        self.analysis = {}

    def analyze_all(self):
        """Run all analysis"""
        print("=" * 80)
        print("PIPELINE ERROR ANALYSIS")
        print("=" * 80)

        self._analyze_overall()
        self._analyze_error_by_difference()
        self._analyze_low_confidence()
        self._analyze_error_by_stage()
        self._analyze_timing()
        self._export_detailed_errors()
        self._generate_recommendations()

        # Save analysis
        self._save_analysis()

        self._print_summary()

    def _analyze_overall(self):
        """Overall statistics"""
        print("\n[1] OVERALL STATISTICS")
        print("-" * 40)

        total = len(self.df)
        correct = self.df['correct'].sum() if 'correct' in self.df.columns else 0
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0

        self.analysis['overall'] = {
            'total': total,
            'correct': int(correct),
            'incorrect': int(incorrect),
            'accuracy': accuracy
        }

        print(f"Total images: {total}")
        print(f"Correct: {correct} ({accuracy:.2%})")
        print(f"Incorrect: {incorrect} ({1-accuracy:.2%})")

        # Error rate
        if 'error_stage' in self.df.columns:
            error_count = self.df['error_stage'].notna().sum()
            print(f"Pipeline errors: {error_count} ({error_count/total:.2%})")

    def _analyze_error_by_difference(self):
        """Analyze errors by prediction difference"""
        print("\n[2] ERROR CLASSIFICATION BY DIFFERENCE")
        print("-" * 40)

        if 'true_value' not in self.df.columns or 'predicted_value' not in self.df.columns:
            print("Skip: Missing true_value or predicted_value columns")
            return

        # Calculate differences
        df_errors = self.df[self.df['correct'] == False].copy()

        if len(df_errors) == 0:
            print("No incorrect predictions found")
            return

        # Convert to numeric
        df_errors['true_num'] = pd.to_numeric(df_errors['true_value'], errors='coerce')
        df_errors['pred_num'] = pd.to_numeric(df_errors['predicted_value'], errors='coerce')
        df_errors['diff'] = (df_errors['true_num'] - df_errors['pred_num']).abs()

        # Classify errors
        off_by_1 = df_errors[df_errors['diff'] <= 1]
        off_by_2_5 = df_errors[(df_errors['diff'] > 1) & (df_errors['diff'] <= 5)]
        off_by_6_10 = df_errors[(df_errors['diff'] > 5) & (df_errors['diff'] <= 10)]
        large_errors = df_errors[df_errors['diff'] > 10]

        total_errors = len(df_errors)

        self.analysis['error_by_difference'] = {
            'off_by_1': {
                'count': len(off_by_1),
                'percentage': len(off_by_1) / total_errors if total_errors > 0 else 0,
                'examples': off_by_1.head(5)[['filename', 'true_value', 'predicted_value']].to_dict('records')
            },
            'off_by_2_5': {
                'count': len(off_by_2_5),
                'percentage': len(off_by_2_5) / total_errors if total_errors > 0 else 0,
                'examples': off_by_2_5.head(5)[['filename', 'true_value', 'predicted_value']].to_dict('records')
            },
            'off_by_6_10': {
                'count': len(off_by_6_10),
                'percentage': len(off_by_6_10) / total_errors if total_errors > 0 else 0,
                'examples': off_by_6_10.head(5)[['filename', 'true_value', 'predicted_value']].to_dict('records')
            },
            'large_errors': {
                'count': len(large_errors),
                'percentage': len(large_errors) / total_errors if total_errors > 0 else 0,
                'examples': large_errors.head(10)[['filename', 'true_value', 'predicted_value', 'diff']].to_dict('records')
            }
        }

        print(f"Off-by-1 errors: {len(off_by_1)} ({len(off_by_1)/total_errors:.1%})")
        print(f"Off-by-2-5 errors: {len(off_by_2_5)} ({len(off_by_2_5)/total_errors:.1%})")
        print(f"Off-by-6-10 errors: {len(off_by_6_10)} ({len(off_by_6_10)/total_errors:.1%})")
        print(f"Large errors (>10): {len(large_errors)} ({len(large_errors)/total_errors:.1%})")

    def _analyze_low_confidence(self):
        """Analyze low confidence predictions"""
        print("\n[3] LOW CONFIDENCE ANALYSIS")
        print("-" * 40)

        for stage in ['m1', 'm3']:
            conf_col = f'{stage}_confidence'

            if conf_col not in self.df.columns:
                continue

            # Low confidence thresholds
            thresholds = [0.2, 0.3, 0.4, 0.5]

            self.analysis[f'{stage}_confidence'] = {}

            for thresh in thresholds:
                low_conf = self.df[self.df[conf_col] < thresh]

                # Count errors in low confidence
                if 'correct' in self.df.columns:
                    low_conf_errors = low_conf[low_conf['correct'] == False]
                    error_rate = len(low_conf_errors) / len(low_conf) if len(low_conf) > 0 else 0
                else:
                    low_conf_errors = pd.DataFrame()
                    error_rate = 0

                self.analysis[f'{stage}_confidence'][f'below_{thresh}'] = {
                    'count': len(low_conf),
                    'percentage': len(low_conf) / len(self.df),
                    'error_count': len(low_conf_errors),
                    'error_rate': error_rate,
                    'examples': low_conf.head(5)[['filename', conf_col]].to_dict('records')
                }

                print(f"{stage.upper()} confidence < {thresh}: {len(low_conf)} ({len(low_conf)/len(self.df):.1%}) - Error rate: {error_rate:.1%}")

    def _analyze_error_by_stage(self):
        """Analyze errors by pipeline stage"""
        print("\n[4] ERROR BY STAGE")
        print("-" * 40)

        if 'error_stage' not in self.df.columns:
            print("Skip: Missing error_stage column")
            return

        stage_errors = self.df[self.df['error_stage'].notna()]

        if len(stage_errors) == 0:
            print("No stage errors found")
            return

        # Count by stage
        stage_counts = stage_errors['error_stage'].value_counts()

        self.analysis['errors_by_stage'] = {}

        for stage, count in stage_counts.items():
            stage_df = stage_errors[stage_errors['error_stage'] == stage]
            self.analysis['errors_by_stage'][stage] = {
                'count': count,
                'percentage': count / len(self.df),
                'examples': stage_df.head(5)[['filename', 'error_stage', 'error_message']].to_dict('records')
            }

            print(f"{stage}: {count} ({count/len(self.df):.1%})")

    def _analyze_timing(self):
        """Analyze execution timing"""
        print("\n[5] TIMING ANALYSIS")
        print("-" * 40)

        if 'total_time_ms' not in self.df.columns:
            print("Skip: Missing total_time_ms column")
            return

        timing_stats = self.df['total_time_ms'].describe()

        self.analysis['timing'] = {
            'mean_ms': timing_stats['mean'],
            'std_ms': timing_stats['std'],
            'min_ms': timing_stats['min'],
            'max_ms': timing_stats['max'],
            'median_ms': timing_stats['50%']
        }

        print(f"Mean: {timing_stats['mean']:.1f}ms")
        print(f"Median: {timing_stats['50%']:.1f}ms")
        print(f"Std: {timing_stats['std']:.1f}ms")
        print(f"Min: {timing_stats['min']:.1f}ms")
        print(f"Max: {timing_stats['max']:.1f}ms")

        # Per-stage timing
        for stage in ['m1', 'm2', 'm3', 'm3_5', 'm4']:
            time_col = f'{stage}_time_ms'

            if time_col in self.df.columns:
                stage_time = self.df[time_col].dropna()
                if len(stage_time) > 0:
                    print(f"{stage.upper()}: {stage_time.mean():.1f}ms avg")

    def _export_detailed_errors(self):
        """Export detailed error lists"""
        print("\n[6] EXPORTING ERROR LISTS")
        print("-" * 40)

        # Incorrect predictions
        if 'correct' in self.df.columns:
            incorrect_df = self.df[self.df['correct'] == False]

            if len(incorrect_df) > 0:
                # Export all incorrect predictions
                error_csv = self.output_dir / "incorrect_predictions.csv"
                incorrect_df.to_csv(error_csv, index=False)
                print(f"Exported {len(incorrect_df)} incorrect predictions to: {error_csv}")

                # Export large errors
                if 'true_value' in self.df.columns and 'predicted_value' in self.df.columns:
                    incorrect_df = incorrect_df.copy()
                    incorrect_df['true_num'] = pd.to_numeric(incorrect_df['true_value'], errors='coerce')
                    incorrect_df['pred_num'] = pd.to_numeric(incorrect_df['predicted_value'], errors='coerce')
                    incorrect_df['diff'] = (incorrect_df['true_num'] - incorrect_df['pred_num']).abs()

                    large_errors = incorrect_df[incorrect_df['diff'] > 10]
                    if len(large_errors) > 0:
                        large_error_csv = self.output_dir / "large_errors.csv"
                        large_errors.to_csv(large_error_csv, index=False)
                        print(f"Exported {len(large_errors)} large errors to: {large_error_csv}")

        # Low confidence images
        for stage in ['m1', 'm3']:
            conf_col = f'{stage}_confidence'

            if conf_col in self.df.columns:
                low_conf = self.df[self.df[conf_col] < 0.3]

                if len(low_conf) > 0:
                    low_conf_txt = self.output_dir / f"low_{stage}_confidence.txt"
                    with open(low_conf_txt, 'w') as f:
                        for fname in low_conf['filename']:
                            f.write(f"{fname}\n")
                    print(f"Exported {len(low_conf)} low {stage.upper()} confidence images to: {low_conf_txt}")

        # Stage errors
        if 'error_stage' in self.df.columns:
            stage_errors = self.df[self.df['error_stage'].notna()]

            if len(stage_errors) > 0:
                stage_error_txt = self.output_dir / "stage_errors.txt"
                with open(stage_error_txt, 'w') as f:
                    for _, row in stage_errors.iterrows():
                        f.write(f"{row['filename']} | {row['error_stage']} | {row.get('error_message', '')}\n")
                print(f"Exported {len(stage_errors)} stage errors to: {stage_error_txt}")

    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""
        print("\n[7] RECOMMENDATIONS")
        print("-" * 40)

        recommendations = []

        # Check low confidence
        if 'm3_confidence' in self.df.columns:
            low_m3 = (self.df['m3_confidence'] < 0.3).sum()
            if low_m3 > len(self.df) * 0.1:
                recommendations.append({
                    'priority': 'HIGH',
                    'issue': f'{low_m3/len(self.df):.1%} images have M3 confidence < 0.3',
                    'recommendation': 'Retrain M3 model with more diverse data or increase training epochs. Check if ROI crop quality is poor.'
                })

        # Check large errors
        if 'error_by_difference' in self.analysis:
            large_error_pct = self.analysis['error_by_difference']['large_errors']['percentage']
            if large_error_pct > 0.05:
                recommendations.append({
                    'priority': 'HIGH',
                    'issue': f'{large_error_pct:.1%} of errors are large (>10 digit difference)',
                    'recommendation': 'Investigate M3 ROI detection (may be cropping wrong region) and M4 OCR quality.'
                })

        # Check off-by-1 errors
        if 'error_by_difference' in self.analysis:
            off_by_1_pct = self.analysis['error_by_difference']['off_by_1']['percentage']
            if off_by_1_pct > 0.3:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'issue': f'{off_by_1_pct:.1%} of errors are off-by-1',
                    'recommendation': 'Minor OCR tuning needed. Beam search parameters or character recognition threshold adjustment.'
                })

        # Check stage failures
        if 'errors_by_stage' in self.analysis:
            for stage, data in self.analysis['errors_by_stage'].items():
                if data['count'] > 10:
                    recommendations.append({
                        'priority': 'HIGH',
                        'issue': f'{data["count"]} images failing at {stage} stage',
                        'recommendation': f'Check {stage} model quality and input data. Consider lowering confidence threshold or retraining.'
                    })

        self.analysis['recommendations'] = recommendations

        if not recommendations:
            print("No major issues detected!")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n[{i}] {rec['priority']} PRIORITY")
                print(f"    Issue: {rec['issue']}")
                print(f"    Action: {rec['recommendation']}")

    def _save_analysis(self):
        """Save analysis to JSON"""
        analysis_path = self.output_dir / "error_analysis.json"

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj

        # Convert recursively
        def clean_data(data):
            if isinstance(data, dict):
                return {k: clean_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_data(item) for item in data]
            else:
                return convert_types(data)

        clean_analysis = clean_data(self.analysis)

        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(clean_analysis, f, indent=2, ensure_ascii=False)

        print(f"\nAnalysis saved to: {analysis_path}")

    def _print_summary(self):
        """Print final summary"""
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")
        print(f"- incorrect_predictions.csv: All incorrect predictions")
        print(f"- large_errors.csv: Errors with >10 digit difference")
        print(f"- low_m1_confidence.txt: Images with M1 confidence < 0.3")
        print(f"- low_m3_confidence.txt: Images with M3 confidence < 0.3")
        print(f"- stage_errors.txt: Images that failed at specific pipeline stages")
        print(f"- error_analysis.json: Full analysis report")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze pipeline errors')
    parser.add_argument('--csv', type=str, required=True, help='Path to pipeline results CSV')
    parser.add_argument('--output', type=str, default=None, help='Output directory for analysis')

    args = parser.parse_args()

    analyzer = PipelineErrorAnalyzer(args.csv, args.output)
    analyzer.analyze_all()


if __name__ == "__main__":
    main()

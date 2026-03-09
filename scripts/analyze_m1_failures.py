"""
Analyze M1 failures - visualize why easy images are not detected
"""
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Paths
DATA_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
LABELS_FILE = Path(r"F:\Workspace\Project\data\images_4digit2.csv")
M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
SUCCESS_CSV = r"F:\Workspace\Project\results\pipeline_fixed_m2\pipeline_results.csv"

print("="*70)
print("M1 FAILURE ANALYSIS")
print("="*70)

# Load data
labels_df = pd.read_csv(LABELS_FILE)
success_df = pd.read_csv(SUCCESS_CSV)
success_files = set(success_df['filename'].values)

all_files = list(DATA_DIR.glob("*.jpg"))
failed_files = [f for f in all_files if f.name not in success_files]

print(f"\nTotal images: {len(all_files)}")
print(f"Successful: {len(success_files)}")
print(f"Failed: {len(failed_files)}")

# Load M1 model
m1_model = YOLO(M1_MODEL)

# Test some failed images
print(f"\nTesting {min(20, len(failed_files))} failed images...")
print("-"*70)

# Create output directory for visualization
viz_dir = Path(r"F:\Workspace\Project\results\m1_failure_analysis")
viz_dir.mkdir(exist_ok=True)

# Analysis
detection_results = []

for i, img_path in enumerate(failed_files[:20], 1):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    # Try to detect
    results = m1_model(img, verbose=False, conf=0.25)

    if len(results) > 0 and len(results[0].boxes) > 0:
        # Has detections with conf=0.25
        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()

        # Try with lower threshold
        results_low = m1_model(img, verbose=False, conf=0.10)

        if len(results_low) > 0 and len(results_low[0].boxes) > 0:
            # Has detections with conf=0.10
            boxes_low = results_low[0].boxes
            confs_low = boxes_low.conf.cpu().numpy()

            print(f"{i}. {img_path.name[:50]}")
            print(f"   Image: {w}x{h}")
            print(f"   Detections@0.25: {len(confs)} (max conf: {np.max(confs):.3f})")
            print(f"   Detections@0.10: {len(confs_low)} (max conf: {np.max(confs_low):.3f})")
            print(f"   >>> PROBLEM: Threshold too high!")

            detection_results.append({
                'filename': img_path.name,
                'size': f"{w}x{h}",
                'max_conf_025': np.max(confs),
                'num_detections_025': len(confs),
                'max_conf_010': np.max(confs_low),
                'num_detections_010': len(confs_low),
                'issue': 'Threshold too high'
            })

            # Visualize
            vis = img.copy()
            for j, box in enumerate(boxes_low):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()

                color = (0, 255, 0) if conf >= 0.25 else (0, 165, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imwrite(str(viz_dir / f"{img_path.stem}_detections.jpg"), vis)

        else:
            print(f"{i}. {img_path.name[:50]}")
            print(f"   Image: {w}x{h}")
            print(f"   Detections@0.25: {len(confs)} (max conf: {np.max(confs):.3f})")
            print(f"   Detections@0.10: 0")
            print(f"   >>> PROBLEM: Model really cannot detect!")

            detection_results.append({
                'filename': img_path.name,
                'size': f"{w}x{h}",
                'max_conf_025': np.max(confs),
                'num_detections_025': len(confs),
                'max_conf_010': 0,
                'num_detections_010': 0,
                'issue': 'Model cannot detect'
            })

            # Visualize original
            cv2.imwrite(str(viz_dir / f"{img_path.stem}_original.jpg"), img)
    else:
        print(f"{i}. {img_path.name[:50]}")
        print(f"   Image: {w}x{h}")
        print(f"   >>> PROBLEM: Zero detections even at low threshold!")
        print(f"   File might be corrupted or invalid")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

if detection_results:
    df_results = pd.DataFrame(detection_results)

    threshold_issues = df_results[df_results['issue'] == 'Threshold too high']
    model_issues = df_results[df_results['issue'] == 'Model cannot detect']

    print(f"\nIssue breakdown:")
    print(f"  Threshold too high: {len(threshold_issues)}")
    print(f"  Model cannot detect: {len(model_issues)}")

    if len(threshold_issues) > 0:
        print(f"\nThreshold too high - Statistics:")
        print(f"  Max confidence (0.25): {threshold_issues['max_conf_025'].max():.3f}")
        print(f"  Max confidence (0.10): {threshold_issues['max_conf_010'].max():.3f}")
        print(f"  Average detections@0.10: {threshold_issues['num_detections_010'].mean():.1f}")

    print(f"\nRecommendation:")
    if len(threshold_issues) > len(model_issues):
        print(f"  >>> LOWER M1 CONFIDENCE THRESHOLD: 0.25 -> 0.15")
    else:
        print(f"  >>> Model needs retraining or fine-tuning")

    print(f"\nVisualizations saved to: {viz_dir}")

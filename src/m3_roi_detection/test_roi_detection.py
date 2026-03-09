"""
M3: ROI Detection Pipeline (Corrected)
Use roi_boundingbox.pt (YOLOv8n) to detect digit region
"""
import os
import sys
import codecs
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Configuration
INPUT_DIR = Path(r"F:\Workspace\Project\data\m2_batch_test_results")
OUTPUT_DIR = Path(r"F:\Workspace\Project\data\m3_roi_detected")
RESULTS_CSV = Path(r"F:\Workspace\Project\data\m3_roi_results.csv")
MODEL_PATH = Path(r"F:\Workspace\Project\model\roi_boundingbox.pt")
CONF_THRESHOLD = 0.25

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("M3: ROI DETECTION (Using YOLOv8n roi_boundingbox.pt)")
print("=" * 70)
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Model: {MODEL_PATH}")
print(f"Confidence Threshold: {CONF_THRESHOLD}")
print("=" * 70)

# ==========================================
# LOAD MODEL
# ==========================================

print("\n[LOAD] Loading ROI detection model...")
model = YOLO(str(MODEL_PATH))
print(f"[LOAD] Model loaded: {type(model).__name__}")
print(f"[INFO] Classes: {model.names}")

# ==========================================
# PROCESSING FUNCTIONS
# ==========================================

def detect_and_crop_roi(model, img_path, output_dir, conf_threshold=0.25):
    """
    Detect ROI in image and crop it
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            'filename': img_path.name,
            'status': 'error',
            'error': 'Could not read image'
        }

    h, w = img.shape[:2]

    # Run inference
    results = model(img, verbose=False)

    # Get the first result (single image)
    if len(results) == 0:
        return {
            'filename': img_path.name,
            'status': 'no_detection',
            'original_size': f"{w}x{h}"
        }

    result = results[0]

    # Get boxes
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            'filename': img_path.name,
            'status': 'no_detection',
            'original_size': f"{w}x{h}"
        }

    # Get the box with highest confidence
    confidences = boxes.conf.cpu().numpy()
    best_idx = confidences.argmax()
    box = boxes[best_idx]

    confidence = float(box.conf[0])

    # Check confidence threshold
    if confidence < conf_threshold:
        return {
            'filename': img_path.name,
            'status': 'low_confidence',
            'confidence': confidence,
            'original_size': f"{w}x{h}"
        }

    # Get bbox coordinates (xyxy format)
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Crop ROI
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return {
            'filename': img_path.name,
            'status': 'empty_crop',
            'confidence': confidence,
            'bbox': f"{x1},{y1},{x2},{y2}",
            'original_size': f"{w}x{h}"
        }

    # Save ROI
    output_path = output_dir / img_path.name.replace('_corrected.jpg', '_roi.jpg')
    cv2.imwrite(str(output_path), roi)

    roi_h, roi_w = roi.shape[:2]

    return {
        'filename': img_path.name,
        'status': 'success',
        'confidence': confidence,
        'bbox': f"{x1},{y1},{x2},{y2}",
        'roi_size': f"{roi_w}x{roi_h}",
        'original_size': f"{w}x{h}"
    }

# ==========================================
# MAIN PROCESSING
# ==========================================

# Get all corrected images
image_files = sorted(list(INPUT_DIR.glob('*_corrected.jpg')))
print(f"\n[SCAN] Found {len(image_files)} corrected images")

if len(image_files) == 0:
    print("[ERROR] No corrected images found!")
    sys.exit(1)

# Process images
print(f"\n[PROCESS] Detecting ROI...")
print("-" * 70)

results = []
success_count = 0
no_detection_count = 0
low_conf_count = 0
error_count = 0

for img_path in tqdm(image_files, desc="Detecting ROI"):
    result = detect_and_crop_roi(model, img_path, OUTPUT_DIR, CONF_THRESHOLD)
    results.append(result)

    # Update counters
    if result['status'] == 'success':
        success_count += 1
    elif result['status'] == 'no_detection':
        no_detection_count += 1
    elif result['status'] == 'low_confidence':
        low_conf_count += 1
    else:
        error_count += 1

# Save results
df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Total images: {len(image_files)}")
print(f"Success: {success_count} ({success_count/len(image_files)*100:.1f}%)")
print(f"No detection: {no_detection_count} ({no_detection_count/len(image_files)*100:.1f}%)")
print(f"Low confidence: {low_conf_count} ({low_conf_count/len(image_files)*100:.1f}%)")
print(f"Errors: {error_count} ({error_count/len(image_files)*100:.1f}%)")
print(f"\nROI images saved to: {OUTPUT_DIR}")
print(f"Results saved to: {RESULTS_CSV}")

# Statistics for successful detections
if success_count > 0:
    success_df = df[df['status'] == 'success']
    print("\n" + "=" * 70)
    print("CONFIDENCE STATISTICS")
    print("=" * 70)
    print(f"Mean: {success_df['confidence'].mean():.4f}")
    print(f"Std: {success_df['confidence'].std():.4f}")
    print(f"Min: {success_df['confidence'].min():.4f}")
    print(f"Max: {success_df['confidence'].max():.4f}")
    print(f"Median: {success_df['confidence'].median():.4f}")

    # ROI size statistics
    roi_sizes = success_df['roi_size'].values
    print(f"\nROI SIZE STATISTICS")
    print(f"Unique sizes: {len(set(roi_sizes))}")
    print(f"Most common sizes:")
    from collections import Counter
    size_counts = Counter(roi_sizes)
    for size, count in size_counts.most_common(5):
        print(f"  {size}: {count} ({count/success_count*100:.1f}%)")

print("\n" + "=" * 70)
print("✅ M3 ROI DETECTION COMPLETED!")
print("=" * 70)

# Show sample results
print(f"\n[SAMPLE RESULTS] (First 10)")
print("-" * 70)
for i, result in enumerate(results[:10], 1):
    status_emoji = "✅" if result['status'] == 'success' else "❌"
    print(f"  {i:2d}. {result['filename'][:60]:60s} {status_emoji}")
    if result['status'] == 'success':
        print(f"      Confidence: {result['confidence']:.4f} | BBox: {result['bbox']}")
    else:
        print(f"      Status: {result['status']}")

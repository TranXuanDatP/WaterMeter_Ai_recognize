#!/usr/bin/env python3
"""
Test M3.5 extraction with debug output
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

INPUT_DIR = r"F:\Workspace\Project\data\images_4digit_xxxx"
OUTPUT_DIR = r"F:\Workspace\Project\data\test_m3_5_debug"
M1_MODEL = r"F:\Workspace\Project\model\M1_DetectWatermeter.pt"
M3_MODEL = r"F:\Workspace\Project\model\M3_Roi_Boundingbox.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Testing M3.5 extraction...")
print(f"Output: {OUTPUT_DIR}")

# Load models
m1_model = YOLO(M1_MODEL)
m3_model = YOLO(M3_MODEL)

# Test on first 5 images
image_files = sorted(list(Path(INPUT_DIR).glob('*.jpg')))[:5]

print(f"\nTesting on {len(image_files)} images...\n")

for idx, img_path in enumerate(image_files):
    filename = os.path.basename(img_path)
    print(f"[{idx+1}] {filename}")

    try:
        # M1
        img = cv2.imread(str(img_path))
        results = m1_model(img, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            print("    M1: No detection")
            continue

        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().cpu().numpy()
        x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy().astype(int)
        m1_crop = img[y1:y2, x1:x2]

        # M3
        results = m3_model(m1_crop, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            print("    M3: No detection")
            continue

        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().cpu().numpy()
        x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy().astype(int)
        m3_roi = m1_crop[y1:y2, x1:x2]

        # Save M3 ROI
        m3_path = os.path.join(OUTPUT_DIR, f"m3_{filename}")
        cv2.imwrite(m3_path, m3_roi)
        print(f"    M3 ROI saved: {m3_path}")

        # Simple digit extraction (gray threshold)
        gray = cv2.cvtColor(m3_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digits = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 20 < w < 300 and 20 < h < 300:
                aspect = w / h if h > 0 else 0
                if 0.3 < aspect < 3.0:
                    digit = gray[y:y+h, x:x+w]
                    digits.append((digit, x))

        digits.sort(key=lambda x: x[1])
        print(f"    Digits found: {len(digits)}")

        if len(digits) > 0:
            # Combine with hstack
            target_height = 64
            resized = []
            for digit, x in digits:
                h, w = digit.shape
                new_w = int(target_height * (w / h))
                resized_digit = cv2.resize(digit, (new_w, target_height))
                resized.append(resized_digit)

            combined = np.hstack(resized)

            # Save
            combined_path = os.path.join(OUTPUT_DIR, f"combined_{filename}")
            cv2.imwrite(combined_path, combined)
            print(f"    Combined saved: {combined_path}")
            print(f"    Shape: {combined.shape}")
        else:
            print("    No digits to combine")

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\nTest complete. Check {OUTPUT_DIR}")

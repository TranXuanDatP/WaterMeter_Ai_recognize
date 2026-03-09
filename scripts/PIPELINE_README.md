# Pipeline Training M1 -> M2 + Smart Rotate -> M3 -> M4

## 📋 Cấu trúc Pipeline Hoàn Chỉnh

```
Raw Images
    ↓
[M1] Water Meter Detection (YOLO)
    → Detect vùng đồng hồ nước trong ảnh
    → Crop vùng đồng hồ
    → Output: m1_crops/
    ↓
[M2] Orientation + Smart Rotate ⭐
    → Tính góc xoay (Angle Regression)
    → Smart Rotate: Xoay ảnh để thẳng hàng
    → Output: m2_crops_aligned/
    ↓
[M3] ROI Detection (YOLOv8n)
    → Detect vùng chứa dãy số (ROI)
    → Crop ROI
    → Output: m3_roi_crops/
    ↓
[M3.5] Black Digit Extraction
    → Detect và cắt từng digit màu đen
    → Output: m4_ocr_dataset_black_digits/
    ↓
[M4] CRNN OCR Reading
    → Đọc chỉ số 4 chữ số
    → Output: Final reading
```

---

## 🔧 Models

| Model | File Name | Type | Mô tả |
|-------|-----------|------|--------|
| **M1** | `M1_DetectWatermeter.pt` | YOLO | Detect đồng hồ nước trong ảnh |
| **M2** | `M2_Orientation.pth` | ResNet18+CBAM | Regression góc xoay (sin/cos) |
| **M3** | `M3_Roi_Boundingbox.pt` | YOLOv8n | Detect ROI (vùng dãy số) |
| **M4** | `M4_OCR.pth` | CRNN+BiLSTM+CTC | OCR 4 chữ số |

---

## 📂 Data Flow Chi Tiết

### Stage 1: M1 - Water Meter Detection
```
Input: data/raw_images/ (ảnh đồng hồ nước)
  ↓
M1_DetectWatermeter.pt (YOLO)
  ↓
Output: data/m1_crops/
  ├── crop_001.jpg  (vùng đồng hồ đã crop)
  ├── crop_002.jpg
  └── ...
  + results/m1_detections.json
```

### Stage 2: M2 - Orientation + Smart Rotate
```
Input: data/m1_crops/ (ảnh crop từ M1)
  ↓
M2_Orientation.pth (Angle Regressor)
  ↓
Predict angle (sin/cos → degrees)
  ↓
Smart Rotate: Xoay ảnh theo góc dự đoán
  ↓
Output: data/m2_crops_aligned/
  ├── aligned_001.jpg  (ảnh đã xoay thẳng)
  ├── aligned_002.jpg
  └── ...
  + results/m2_angles.json
```

**Smart Rotate Algorithm:**
```python
# 1. Predict angle using M2 model
sin_cos = m2_model(image)  # Output: [sin, cos]
angle = atan2(sin, cos)     # Convert to degrees

# 2. Rotate image to upright
rotated = rotate_image(image, angle)

# 3. Verify rotation quality
if is_rotated_correctly(rotated):
    save_rotated(rotated)
else:
    fallback_to_original(image)
```

### Stage 3: M3 - ROI Detection
```
Input: data/m2_crops_aligned/ (ảnh đã xoay thẳng)
  ↓
M3_Roi_Boundingbox.pt (YOLOv8n)
  ↓
Detect ROI (vùng chứa dãy số)
  ↓
Crop ROI
  ↓
Output: data/m3_roi_crops/
  ├── roi_001.jpg  (vùng ROI đã crop)
  ├── roi_002.jpg
  └── ...
  + results/m3_roi_results.csv
```

### Stage 3.5: Black Digit Extraction
```
Input: data/m3_roi_crops/ (vùng ROI)
  ↓
Digit Detection / Color-based Segmentation
  ↓
Extract black digits (4 chữ số)
  ↓
Output: data/m4_ocr_dataset_black_digits/
  ├── crop_meter4_00000_xxx.jpg
  ├── crop_meter4_00001_xxx.jpg
  └── ...
  + labels.csv
```

**Black Digit Extraction Methods:**
- **Method 1**: YOLO detection từng digit
- **Method 2**: Color segmentation (lọc pixel đen)
- **Method 3**: Contour detection
- **Method 4**: Horizontal projection

### Stage 4: M4 - CRNN OCR Reading
```
Input: data/m4_ocr_dataset_black_digits/ (ảnh 4 digit)
  ↓
M4_OCR.pth (CRNN+BiLSTM+CTC)
  ↓
Predict: 4-digit sequence
  ↓
Output: Final reading
  + confidence score
```

---

## 🚀 Chạy Pipeline

### Method 1: Chạy từng stage riêng lẻ

#### **Stage 1: M1 Detection**
```bash
# Detect đồng hồ nước
python scripts/run_m1_detection.py \
    --input data/raw_images \
    --output data/m1_crops \
    --model model/M1_DetectWatermeter.pt
```

**Output:**
- `data/m1_crops/` - Ảnh crop đồng hồ
- `results/m1_detections.json` - Bounding boxes

#### **Stage 2: M2 Orientation + Smart Rotate**
```bash
# Tính góc và xoay ảnh
python scripts/run_m2_orientation.py \
    --input data/m1_crops \
    --output data/m2_crops_aligned \
    --model model/M2_Orientation.pth
```

**Output:**
- `data/m2_crops_aligned/` -  ảnh đã xoay thẳng
- `results/m2_angles.json` - Góc xoay từng ảnh

#### **Stage 3: M3 ROI Detection**
```bash
# Detect ROI
python scripts/run_m3_roi_detection.py \
    --input data/m2_crops_aligned \
    --output data/m3_roi_crops \
    --model model/M3_Roi_Boundingbox.pt
```

**Output:**
- `data/m3_roi_crops/` - Ảnh ROI
- `results/m3_roi_results.csv` - Kết quả detection

#### **Stage 3.5: Extract Black Digits**
```bash
# Cắt black digits
python scripts/extract_black_digits.py \
    --input data/m3_roi_crops \
    --output data/m4_ocr_dataset_black_digits
```

**Output:**
- `data/m4_ocr_dataset_black_digits/images/` - Ảnh 4 digit
- `data/m4_ocr_dataset_black_digits/labels.csv`

#### **Stage 4: M4 OCR Reading**
```bash
# Đọc số
python scripts/run_m4_ocr.py \
    --input data/m4_ocr_dataset_black_digits/images \
    --labels data/m4_ocr_dataset_black_digits/labels.csv \
    --model model/M4_OCR.pth
```

**Output:**
- `results/final_readings.csv` - Kết quả OCR

---

### Method 2: Chạy toàn bộ pipeline tự động

```bash
# Run complete pipeline
python scripts/run_pipeline_complete.py \
    --input data/raw_images \
    --output results/pipeline_run
```

Script này sẽ tự động:
1. M1: Detect đồng hồ → crop
2. M2: Predict góc → smart rotate
3. M3: Detect ROI → crop
4. M3.5: Extract black digits
5. M4: CRNN OCR reading

---

## 📊 Monitoring & Logging

### M1 Logs
```
results/pipeline_run/M1/
├── m1_detections.json       # Tất cả detections
├── m1_crops/                # Ảnh crop
└── metrics.json             # Precision, Recall, mAP
```

### M2 Logs
```
results/pipeline_run/M2/
├── m2_angles.json           # Góc xoay từng ảnh
├── m2_crops_aligned/        #  ảnh đã xoay
├── rotation_stats.png       # Histogram góc xoay
└── metrics.json             # MAE, RMSE
```

### M3 Logs
```
results/pipeline_run/M3/
├── m3_roi_results.csv       # ROI detections
├── m3_roi_crops/            # Ảnh ROI
└── metrics.json             # Precision, Recall
```

### M3.5 Logs
```
results/pipeline_run/M3.5/
├── extraction_log.json      # Log extraction
├── m4_ocr_dataset_black_digits/
│   ├── images/              # Ảnh black digits
│   └── labels.csv           # Labels
└── extraction_stats.png     # Thống kê
```

### M4 Logs
```
results/pipeline_run/M4/
├── final_readings.csv       # Kết quả cuối
├── confidence_histogram.png # Phân phối confidence
└── metrics.json             # Accuracy, char accuracy
```

---

## 🎯 Smart Rotate (M2) Details

### Angle Prediction
```python
# M2 model outputs sin/cos representation
sin_cos = model(image)  # [sin_value, cos_value]

# Convert to degrees
angle = math.degrees(math.atan2(sin_cos[0], sin_cos[1]))
angle = (angle + 360) % 360  # Normalize to [0, 360)
```

### Smart Rotate Process
```python
def smart_rotate(image, angle):
    # 1. Rotate to upright
    rotated = rotate_image(image, angle)

    # 2. Verify quality
    if is_good_rotation(rotated):
        return rotated

    # 3. Fallback: try angle + 90, +180, +270
    for offset in [90, 180, 270]:
        alt_rotated = rotate_image(image, angle + offset)
        if is_good_rotation(alt_rotated):
            return alt_rotated

    # 4. Last resort: return original
    return image
```

### Rotation Quality Check
- Check if text lines are horizontal
- Verify aspect ratio
- Edge detection quality

---

## 🔍 Black Digit Extraction (M3.5)

### Methods Available

**Method 1: Color Segmentation**
```python
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Threshold black pixels
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract digit regions
digits = [crop_contour(c) for c in contours]
```

**Method 2: Horizontal Projection**
```python
# Sum pixels along rows
h_proj = np.sum(255 - gray, axis=1)

# Find digit rows (peaks in projection)
peaks = find_peaks(h_proj, distance=10)

# Crop each digit row
digits = [gray[y1:y2, :] for y1, y2 in peaks]
```

**Method 3: YOLO Detection**
```python
# Use YOLO to detect each digit
model = YOLO('digit_detector.pt')
results = model(image)

# Crop each detected digit
digits = [crop_box(r.boxes.xyxy) for r in results]
```

---

## 📈 Performance Metrics

### M1: Water Meter Detection
- **Precision**: % detections đúng là đồng hồ
- **Recall**: % đồng hồ được detect
- **mAP@0.5**: Mean Average Precision

### M2: Orientation + Smart Rotate
- **MAE**: Mean Absolute Error (degrees)
- **RMSE**: Root Mean Square Error
- **Accuracy**: % angles within ±5°

### M3: ROI Detection
- **Precision**: % detections đúng là ROI
- **Recall**: % ROI được detect
- **IoU**: Intersection over Union

### M3.5: Black Digit Extraction
- **Detection Rate**: % digits được extract
- **Quality Score**: Chất lượng crop

### M4: CRNN OCR
- **Accuracy**: % predictions đúng hoàn toàn
- **Char Accuracy**: % chữ số đúng
- **Edit Distance**: Levenshtein distance

---

## 🔧 Configuration

### M1 Config
```python
M1_CONFIDENCE = 0.25      # Ngưỡng confidence YOLO
M1_IOU_THRESHOLD = 0.45   # IoU threshold cho NMS
M1_IMG_SIZE = 640         # Input image size
```

### M2 Config
```python
M2_CONFIDENCE = 0.5       # Ngưỡng confidence xoay
M2_MAX_ANGLE = 45         # Góc xoay tối đa (độ)
M2_FALLBACK = True        # Cho phép fallback nếu xoay lỗi
```

### M3 Config
```python
M3_CONFIDENCE = 0.25      # Ngưỡng confidence ROI
M3_IMG_SIZE = 640         # Input image size
M3_MIN_WIDTH = 100        # Minimum ROI width
```

### M3.5 Config
```python
EXTRACTION_METHOD = "color"  # color, projection, yolo
MIN_DIGIT_SIZE = 20         # Minimum digit size
MAX_DIGIT_SIZE = 200        # Maximum digit size
```

### M4 Config
```python
M4_IMG_HEIGHT = 64         # Input height
M4_IMG_WIDTH = 256         # Input width
M4_BATCH_SIZE = 32         # Batch size
M4_CONFIDENCE_THRESHOLD = 0.7
```

---

## 💡 Tips

### Debug Smart Rotate (M2)
```python
# Visualize rotation
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3)
axes[0].imshow(original)
axes[0].set_title(f'Original (angle: {predicted_angle:.1f}°)')
axes[1].imshow(rotated)
axes[1].set_title('Rotated')
axes[2].imshow(verification)
axes[2].set_title('Verification')
plt.show()
```

### Verify ROI Detection (M3)
```python
# Draw bounding boxes
for result in results:
    x1, y1, x2, y2 = result.boxes.xyxy[0]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('roi_detection.jpg', image)
```

### Check Black Digit Extraction (M3.5)
```python
# Visualize extracted digits
fig, axes = plt.subplots(1, 4)
for i, digit in enumerate(digits):
    axes[i].imshow(digit, cmap='gray')
    axes[i].set_title(f'Digit {i+1}')
plt.show()
```

---

## 🎯 Summary

### Pipeline Flow:
1. **M1**: Detect đồng hồ → crop
2. **M2**: Predict góc → smart rotate → thẳng hàng
3. **M3**: Detect ROI → crop dãy số
4. **M3.5**: Extract black digits → data M4
5. **M4**: CRNN OCR → đọc chỉ số

### Key Features:
- ✅ **Smart Rotate** (M2): Tự động xoay ảnh thẳng hàng
- ✅ **ROI Detection** (M3): Detect vùng số chính xác
- ✅ **Black Digit Extraction** (M3.5): Tách từng digit
- ✅ **End-to-End**: Từ ảnh thô → chỉ số cuối

### Files:
- `scripts/run_pipeline_complete.py` - Full pipeline
- `scripts/run_m1_detection.py` - M1 standalone
- `scripts/run_m2_orientation.py` - M2 standalone
- `scripts/run_m3_roi_detection.py` - M3 standalone
- `scripts/extract_black_digits.py` - M3.5
- `scripts/run_m4_ocr.py` - M4 standalone

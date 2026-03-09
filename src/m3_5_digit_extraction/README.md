# M3.5: Black Digit Extraction

## 📋 Overview

M3.5 trích xuất 4 chữ số đen (phần nguyên) từ ảnh ROI của M3, loại bỏ chữ số đỏ (phần thập phân).

```
M3 ROI Output (contains 4 black digits + 1 red digit)
     ↓
M3.5: Detect and crop only black digits
     ↓
Output: 4 black digits → M4 OCR input
```

## 🎯 Chức năng

- Phát hiện chữ số đỏ ở bên phải (red digit/decimal part)
- Crop chỉ phần chữ số đen (left part)
- Tự động fallback nếu không phát hiện được red digit
- Đảm bảo giữ ít nhất 75% chiều rộng ảnh

## 📁 Files

### crop_black_digits.py
Script chính để crop black digits từ ROI images.

**Usage:**
```bash
cd src/m3_5_digit_extraction
python crop_black_digits.py
```

**Configuration:**
```python
INPUT_DIR = Path(r"F:\Workspace\Project\data\m3_roi_crops_all")
OUTPUT_DIR = Path(r"F:\Workspace\Project\data\m5_black_digits")
```

### test_m5.py
Script test với visualization.

## 🔬 Phương pháp phát hiện

### 1. Red Color Detection (HSV)
```python
# Red hue range in HSV (two ranges because red wraps around)
lower_red1 = [0, 100, 100]      # 0-10 degrees
upper_red1 = [10, 255, 255]

lower_red2 = [170, 100, 100]    # 170-180 degrees
upper_red2 = [180, 255, 255]
```

### 2. Morphological Operations
- **CLOSE**: Đóng lỗ hổng trong mask
- **OPEN**: Loại bỏ noise

### 3. Contour Detection
- Tìm tất cả contours trong red mask
- Lọc contours theo kích thước (width > 5, height > 10)
- Chọn rightmost region (chữ số đỏ ở bên phải nhất)

### 4. Safety Fallback
```python
# Nếu không phát hiện được red digit
red_x_start = int(w * 0.8)  # Dùng 80% width

# Đảm bảo giữ ít nhất 75% width
crop_x_end = max(red_x_start - 5, int(w * 0.75))
```

## 📊 Input/Output

### Input (from M3)
```
m3_roi_crops_all/
  ├── image1_roi.jpg       # 400×500 - 4 black digits + 1 red digit
  ├── image2_roi.jpg
  └── ...
```

### Output (to M4)
```
m5_black_digits/
  ├── image1_roi.jpg       # 300×500 - 4 black digits only
  ├── image2_roi.jpg
  └── ...

m5_crop_results.csv
  - filename
  - status
  - original_size
  - crop_size
  - red_x_start
  - crop_ratio
```

## 📈 Thống kê hiệu suất

Sample results từ test_pipeline:
- **Success rate:** 100% (50/50)
- **Mean crop ratio:** ~0.78 (giữ 78% width)
- **Std crop ratio:** ~0.05
- **Min crop ratio:** 0.75 (safety check working)

## 🚀 Integration với Pipeline

```
M1 (Detect) → M2 (Align) → M3 (ROI) → M3.5 (Black Digits) → M4 (OCR)
```

**Example trong complete pipeline:**
```python
from src.m3_5_digit_extraction.crop_black_digits import crop_black_digits

# Crop black digits từ M3 ROI
result = crop_black_digits(roi_image_path, output_dir)
```

## 🔧 Troubleshooting

### 1. Crop ratio quá thấp (<75%)
**Problem:** Red digit detection quá sớm
**Solution:** Safety check tự động giữ 75% width

### 2. Không phát hiện được red digit
**Cause:** Ảnh có thể chỉ có 4 chữ số đen (không có chữ số đỏ)
**Solution:** Fallback sang 80% width

### 3. Crop bị lệch
**Cause:** Red digit detection không chính xác
**Solution:**
- Kiểm tra illumination của ảnh
- Điều chỉnh HSV thresholds nếu cần
- Dùng manual crop cho edge cases

## 💡 Best Practices

1. **Input quality:** Đảm bảo M3 ROI đã được align tốt từ M2
2. **Review outputs:** Kiểm tra mẫu black digits trước khi train M4
3. **Augmentation:** Nếu crop ratio quá thấp, xem xét lại M3 ROI
4. **Batch processing:** Script hỗ trợ processing hàng loạt ảnh

## 📝 Notes

- M3.5 được cải tiến từ M5 cũ với better red detection
- Sử dụng HSV color space vì robust hơn với illumination changes
- Morphological operations giúp giảm noise
- Safety fallback đảm bảo không crop quá nhiều

## 🔗 Related Modules

- **M3:** ROI Detection (`src/m3_roi_detection/`)
- **M4:** CRNN OCR (`src/m4_crnn_reading/`)
- **M2:** Orientation Alignment (`src/m2_orientation/`)

---

**Created:** 2026-03-08
**Status:** ✅ Active - Replaced old M3.5 implementation
**Version:** 2.0 (Improved from M5)

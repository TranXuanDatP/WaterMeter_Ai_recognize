# Hướng dẫn Fix Labels với Zero-Padding cho M4 OCR

## 📋 Tổng quan

Script này đảm bảo tất cả labels trong dataset M4 OCR đều có đúng **4 chữ số** với zero-padding phía trước.

### Ví dụ chuyển đổi:
- `5` → `"0005"`
- `87` → `"0087"`
- `187` → `"0187"`
- `1234` → `"1234"`

## 🔧 Cách sử dụng

### Option 1: Chuẩn bị dataset mới (từ đầu)

Sử dụng script `prepare_dataset.py` để tạo dataset mới với labels đã được fix:

```bash
cd F:\Workspace\Project\src\m4_ocr
python prepare_dataset.py
```

**Kết quả:**
- Tạo thư mục `F:\Workspace\Project\data\m4_ocr_dataset_black_digits\`
- File `labels.csv` với tất cả labels đã được format 4-digit
- Copy tất cả images vào thư mục `images/`

### Option 2: Fix file labels.csv hiện tại

Nếu bạn **đã có dataset** và chỉ muốn fix labels:

```bash
cd F:\Workspace\Project\src\m4_ocr
python fix_labels_csv.py
```

**Trước khi chạy:**
1. Mở file `fix_labels_csv.py`
2. Tìm biến `LABELS_CSV` ở cuối file
3. Update đường dẫn đến file `labels.csv` của bạn:

```python
# Thay đổi đường dẫn này
LABELS_CSV = r"F:\Workspace\Project\data\m4_ocr_dataset_black_digits\labels.csv"
```

## ✅ Validation

Script sẽ tự động:

1. **Backup file gốc** trước khi fix
2. **Hiển thị sample** trước và sau khi fix
3. **Validate kết quả:**
   - Kiểm tra độ dài labels (phải = 4)
   - Kiểm tra non-digit characters
   - Hiển thị distribution

### Output sample:

```
==================================================================
FIX LABELS CSV - ZERO-PADDING
==================================================================

📂 Processing: F:\Workspace\Project\data\m4_ocr_dataset_black_digits\labels.csv
💾 Backup created: labels.csv.backup_20260306_143000

📊 Loaded 6193 rows from CSV
   Columns: ['filename', 'text']

📋 Sample labels BEFORE fixing:
----------------------------------------------------------------------
  crop_meter4_03907_9af88e4368594e9aaae6be8465a3...jpg | 187        (type: int)
  crop_meter4_05084_c9c0d66a37054224adca08eb4188...jpg | 5          (type: int)

🔧 Applying zero-padding fix...

✅ Sample labels AFTER fixing:
----------------------------------------------------------------------
  crop_meter4_03907_9af88e4368594e9aaae6be8465a3...jpg | 0187       (type: str)
  crop_meter4_05084_c9c0d66a37054224adca08eb4188...jpg | 0005       (type: str)

📊 VALIDATION RESULTS:
----------------------------------------------------------------------

Label length distribution:
  ✅ 4 digits:  6193 (100.0%)

✅ All labels contain only digits

💾 Saving fixed labels...
✅ Saved to: F:\Workspace\Project\data\m4_ocr_dataset_black_digits\labels.csv

==================================================================
✅ SUCCESS: All labels are now correctly formatted as 4-digit strings!
==================================================================
```

## 📁 Cấu trúc Dataset

Sau khi fix, dataset sẽ có cấu trúc:

```
m4_ocr_dataset_black_digits/
├── labels.csv          # File labels với format 4-digit
└── images/             # Thư mục chứa images
    ├── crop_meter4_xxx.jpg
    ├── crop_meter4_xyy.jpg
    └── ...
```

### Format labels.csv:

```csv
filename,text
crop_meter4_03907_9af88e4368594e9aaae6be8465a3...,0187
crop_meter4_05084_c9c0d66a37054224adca08eb4188...,0005
crop_meter4_05728_e1c6961a5e6f4a4a8c848248d239...,0187
```

## 🚀 Next Steps

Sau khi fix labels thành công:

1. **ZIP dataset:**
   ```bash
   cd F:\Workspace\Project\data
   tar -czf m4_ocr_dataset_black_digits.zip m4_ocr_dataset_black_digits/
   # Hoặc dùng Windows Explorer để ZIP
   ```

2. **Upload lên Google Drive:**
   - Upload file `m4_ocr_dataset_black_digits.zip`
   - Đặt trong thư mục `watermeter_Project/`

3. **Run training trong Colab:**
   - Sử dụng notebook: `M4_OCR_Training_BlackDigits_2Stages_CLEAN.ipynb`
   - Upload notebook lên Google Colab
   - Run các cell theo thứ tự

## 🔍 Troubleshooting

### Lỗi: "File không tồn tại"

**Nguyên nhân:** Đường dẫn đến file `labels.csv` không đúng.

**Giải pháp:**
1. Kiểm tra đường dẫn file
2. Update biến `LABELS_CSV` trong script
3. Sử dụng đường dẫn tuyệt đối (full path)

### Lỗi: "CSV không có cột 'text'"

**Nguyên nhân:** File CSV không có cột `text`.

**Giải pháp:**
1. Kiểm tra format CSV của bạn
2. Đảm bảo có cột `text` chứa labels
3. Nếu cần, đổi tên cột trong script

### Warning: "Labels không có 4 digits"

**Nguyên nhân:** Có labels không đúng format sau khi fix.

**Giải pháp:**
1. Kiểm tra output validation
2. Xem sample labels bị lỗi
3. Kiểm tra data source gốc

## 📊 Statistics

Dataset sau khi fix:
- **Total samples:** 6,193 images
- **Label format:** 4-digit strings (0000-9999)
- **Distribution:** 99% trong range 0-999
- **Validation:** ✅ All labels = 4 digits

## 🎯 Key Points

1. **Convert sang string TRƯỚC** khi xử lý labels
2. **Pad với zeros** để đảm bảo 4 chữ số
3. **Validate** sau khi fix
4. **Backup** file gốc trước khi modify
5. **Test** trên validation set trước khi train

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Check output validation của script
2. Kiểm tra backup file được tạo
3. Review sample labels trước và sau
4. Test với vài样本 trước khi chạy toàn bộ

---

**Author:** Claude Code Assistant
**Date:** 2026-03-06
**Version:** 1.0

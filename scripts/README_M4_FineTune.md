# M4 Fine-tuning - Hướng dẫn sử dụng

## 📋 Tóm tắt

File: `M4_FineTune_3Layer_Strategy.ipynb`

**Mục tiêu**: Sửa lỗi **6 → 0** (chiếm 20.4% tổng số lỗi)

**Kết quả kỳ vọng**:
- 6→0 Error Rate: 20.4% → <5%
- Overall Accuracy: 76.55% → ~85-90%
- Pipeline Success: 96.25% → ~98-99%

---

## 🚀 Cách sử dụng trên Google Colab

### Bước 1: Upload file lên Google Drive

1. Mở Google Drive: https://drive.google.com
2. Tạo thư mục: `MyDrive/Project/`
3. Upload các file sau:

```
Project/
├── data/
│   ├── data_4digit2/          # 6,527 ảnh gốc
│   └── images_4digit2.csv      # File labels
└── model/
    └── M4_OCR.pth              # Model hiện tại
```

4. Upload file notebook: `M4_FineTune_3Layer_Strategy.ipynb`

### Bước 2: Mở notebook trên Colab

1. Vào: https://colab.research.google.com
2. Click: File → Open Notebook → Upload
3. Chọn: `M4_FineTune_3Layer_Strategy.ipynb`

### Bước 3: Cấu hình paths

Trong cell **"📂 Prepare Data"**, cập nhật paths nếu cần:

```python
# Nếu structure của bạn khác nhau, sửa ở đây:
DRIVE_DATA_DIR = "/content/drive/MyDrive/Project/data/data_4digit2"
DRIVE_LABELS_FILE = "/content/drive/MyDrive/Project/data/images_4digit2.csv"
DRIVE_MODEL_PATH = "/content/drive/MyDrive/Project/model/M4_OCR.pth"
```

### Bước 4: Chạy tất cả cells

1. Menu: Runtime → Run all
2. Hoặc: Ctrl+F9 (Windows) hoặc Cmd+F9 (Mac)
3. Chờ ~30 phút (GPU T4)

---

## 🎯 3 Lớp Cải Tiến

### Layer 1: WeightedRandomSampler 🍽️

**Vấn đề**: Số 0, 1, 2 chiếm ưu thế tuyệt đối trong dataset
- AI "học vẹt" đoán số 0 để lấy accuracy cao

**Giải pháp**: Ep AI nhìn thấy số hiếm (6,7,8,9) nhiều ngang bằng
```python
# Samples with digit 6 get much higher weight
weight = 1.0 / (min_digit_frequency + 1e-6)
```

**Kết quả**: AI buộc phải học đặc trưng của từng số

### Layer 2: Sharpening Pre-processing 🔍

**Vấn đề**: 6 → 0 khi ảnh bị mờ (blur)
- Mất cái "móc" phía trên của số 6

**Giải pháp**: Làm sắc nét ảnh trước khi đưa vào model
```python
# Unsharp Masking + CLAHE
sharpened = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
enhanced = clahe.apply(sharpened)
```

**Kết quả**: "Móc" của số 6 rõ hơn, dễ phân biệt với 0

### Layer 3: Weighted Loss ⚖️

**Vấn đề**: Lỗi 6→0 vẫn xảy ra vì "phạt" chưa đủ nặng

**Giải pháp**: Tăng hình phạt dựa trên phân tích lỗi thực tế

📊 **Bảng trọng số đề xuất (Dựa trên phân tích 1,473 lỗi):**

| Chữ số | Trọng số | Lý do |
|--------|----------|-------|
| **6** | **4.0x** | Tỉ lệ lỗi 26% (CAO NHẤT) - Target chính! |
| **1** | **3.5x** | Số lượng lỗi tuyệt đối nhiều nhất (370 lần) |
| **8** | **3.0x** | Hay bị nhầm thành 0 và 1 |
| **9** | **2.5x** | Dữ liệu ít, dễ bị mô hình bỏ qua |
| **7** | **2.0x** | Hay nhầm với 1 |
| **5** | **2.0x** | Hay nhầm thành 8 (mirror) |
| **4** | **2.0x** | Hay nhầm thành 3 |
| **2** | **1.5x** | Tương đối ổn định |
| **3** | **1.5x** | Tương đối ổn định |
| **0** | **0.5x** | GIẢM TRỌNG SỐ - AI bớt "đoán mò" |

```python
digit_weights = [0.5, 3.5, 1.5, 1.5, 2.0, 2.0, 4.0, 2.0, 3.0, 2.5, 1.0]
#              0    1    2    3    4    5    6    7    8    9   blank
```

**Kết quả**: AI sợ sai số 6 → học kỹ hơn

---

## 📊 Theo dõi Training

### Metrics quan trọng

1. **6→0 Error Rate** → Mục tiêu: <5%
2. **Digit Accuracy** → Mục tiêu: >80%
3. **Val Accuracy** → Mục tiêu: >85%

### Output files

Training hoàn tất sẽ tạo:

```
/content/finetuned_models/
├── best_model.pth                    # Best overall accuracy
└── best_six_to_zero_model.pth       # Best 6→0 error rate
```

Tự động copy về Google Drive:

```
/content/drive/MyDrive/Project/model/M4_finetuned/
├── best_model.pth
└── best_six_to_zero_model.pth
```

---

## 🔧 Sử dụng model đã fine-tune

### Cập nhật pipeline script

```python
# Thay đổi path trong pipeline_m1_m2_m3_m3_5_m4.py
class Config:
    M4_MODEL = r"F:\Workspace\Project\model\M4_finetuned\best_six_to_zero_model.pth"
    # Hoặc dùng best_model.pth nếu accuracy tốt hơn
```

### Chạy pipeline

```bash
cd F:\Workspace\Project\scripts
python pipeline_m1_m2_m3_m3_5_m4.py
```

### So sánh kết quả

| Metric | Trước | Sau (dự kiến) |
|--------|-------|--------------|
| 6→0 Error | 20.4% | <5% |
| OCR Accuracy | 76.55% | ~85-90% |
| Overall Accuracy | 73.68% | ~82-88% |

---

## 🛠️ Troubleshooting

### Lỗi: "Cannot find images"

**Nguyên nhân**: Path sai trên Google Drive

**Giải pháp**:
1. Kiểm tra structure thư mục
2. Sửa paths trong cell "Prepare Data"
3. Uncomment dòng copy nếu muốn copy data sang Colab

### Lỗi: "CUDA out of memory"

**Nguyên nhân**: Batch size quá lớn

**Giải pháp**: Giảm batch size
```python
# Trong cell "Start Fine-tuning!"
batch_size = 16  # Thay vì 32
```

### Lỗi: "Model not found"

**Nguyên nhân**: Chưa upload model hoặc path sai

**Giải pháp**:
1. Upload M4_OCR.pth lên Drive
2. Kiểm tra path trong cell "Prepare Data"

---

## 📈 Thời gian & Chi phí

| Tài nguyên | Thời gian | Chi phí |
|------------|-----------|---------|
| **Colab GPU T4** | ~30 phút | Miễn phí |
| **Colab GPU Pro** | ~15 phút | $0.20 |
| **Local CPU** | ~8 tiếng | Miễn phí (chậm) |

**Khuyến nghị**: Dùng Colab GPU T4 (miễn phí, đủ nhanh)

---

## 🎓 Lý thuyết

### Tại sao 3 lớp này hoạt động?

**Vấn đề gốc**: Dataset imbalance
- Số 0, 1, 2 chiếm ~60%
- Số 6, 7, 8, 9 chỉ ~20%

**Hệ quả**: AI "lười" học đặc trưng số hiếm
- Đoán 0 → có 60% chance đúng
- Sai số 6 → chỉ mất ~2% accuracy

**3 lớp → Kích thích AI học kỹ hơn**:
1. Sampler: "Thấy số 6 nhiều lên!"
2. Sharpening: "Nhìn kỹ móc của số 6!"
3. Weighted Loss: "Sai số 6 bị phạt nặng!"

---

## 🚀 Next Steps

Sau khi fine-tune xong:

1. ✅ **Test trên full pipeline**
   ```bash
   python pipeline_m1_m2_m3_m3_5_m4.py
   ```

2. ✅ **Phân tích lỗi mới** (nếu có)
   ```bash
   python analyze_failures.py
   ```

3. ✅ **Cải thiện tiếp** (nếu cần)
   - Tăng beam_width: 10 → 20
   - Thêm post-processing
   - Ensemble models

4. ✅ **Backup kết quả**
   ```bash
   python backup_pipeline.py
   ```

---

## 📞 Support

Nếu gặp lỗi:

1. Kiểm tra Colab logs (output của từng cell)
2. Đảm bảo đủ RAM (Colab có 12GB)
3. Thử giảm batch size nếu OOM
4. Upload lại model nếu file corrupt

---

**Created**: 2026-03-09
**Author**: Claude Code
**Strategy**: 3-Layer Approach to Fix 6→0 Error

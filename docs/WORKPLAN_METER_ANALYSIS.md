# WORKPLAN - HỆ THỐNG PHÂN TÍCH ĐỒNG HỒ NƯỚC TỰ ĐỘNG

## 📋 TỔNG QUAN DỰ ÁN

**Mục tiêu**: Xây dựng hệ thống hoàn chỉnh để tự động hóa quá trình đọc và quản lý chỉ số đồng hồ nước bằng Computer Vision và AI.

**Công nghệ hiện có**:
- CRNN Model (ResNet18 + BiLSTM + CTC) cho OCR
- PyTorch framework
- OpenCV cho image processing
- Accuracy hiện tại: ~22% → cần cải thiện

---

## 🎯 CÁC MODULE CẦN PHÁT TRIỂN

### 1️⃣ TRÍCH XUẤT SERIAL ĐỒNG HỒ TỪ ẢNH
**Mục tiêu**: Nhận diện mã số serial trên đồng hồ

**Phân tích kỹ thuật**:
- **Vị trí serial**: Thường nằm ở mặt trước hoặc mặt sau đồng hồ
- **Đặc điểm**: Chữ số + chữ cái, có thể in nổi hoặc in chìm
- **Thách thức**:
  - Góc chụp không chuẩn
  - Ánh sáng kém, phản chiếu
  - Serial bị che khuất bới bụi/bẩn

**Cách tiếp cận**:
```
a) TEXT DETECTION
   - Sử dụng EAST/DBNet/FCENet để detect vùng text
   - Hoặc dùng contour detection với OpenCV
   - Tìm vùng có đặc điểm text (high gradient, consistent spacing)

b) TEXT RECOGNITION
   - Option 1: Fine-tune CRNN hiện có cho alphanumeric characters
   - Option 2: Sử dụng EasyOCR/PaddleOCR cho serial
   - Option 3: Tesseract OCR với custom preprocessing

c) PREPROCESSING
   - Adaptive thresholding
   - Morphological operations (dilation/erosion)
   - Perspective transformation nếu góc chụp nghiêng
```

**Deliverables**:
- [ ] `serial_detector.py` - Module detect vùng serial
- [ ] `serial_ocr.py` - Module OCR cho serial
- [ ] `test_serial_detection.py` - Test script
- [ ] Dataset serial (nếu cần train model riêng)

---

### 2️⃣ ĐỌC CHỈ SỐ NƯỚC SỬ DỤNG TỰ ĐỘNG
**Mục tiêu**: Cải thiện accuracy đọc chỉ số từ ~22% lên >90%

**Phân tích hiện trạng**:
- Model CRNN V5 đã có nhưng accuracy thấp
- Rolling digits (9→0) gây lỗi
- Cần post-processing tốt hơn

**Cách tiếp cận**:
```
a) IMPROVE EXISTING MODEL
   - Stage 1: Tăng dataset quality (clean data hơn)
   - Stage 2: Fine-tune với rolling digits augmentation
   - Stage 3: Ensemble models (CRNN + transformer-based)

b) ADVANCED PREPROCESSING
   - Deblooming - loại bỏ bloom effect
   - Shadow removal
   - Contrast enhancement (CLAHE)
   - Image inpainting cho vùng bị che

c) BETTER POST-PROCESSING
   - Rolling digit detection rule-based
   - Context-aware validation (không nhảy vọt quá 100)
   - Historical data comparison

d) MULTI-MODEL ENSEMBLE
   - Model 1: CRNN hiện tại
   - Model 2: Vision Transformer (ViT)
   - Model 3: Attention-based OCR
   - Voting mechanism
```

**Deliverables**:
- [ ] `preprocess_advanced.py` - Advanced preprocessing pipeline
- [ ] `crnn_v6_ensemble.py` - Ensemble model
- [ ] `postprocess_smart.py` - Smart post-processing
- [ ] `test_accuracy_enhanced.py` - Test script với metrics chi tiết
- [ ] `generate_synthetic_data.py` - Tool tạo dữ liệu nhân tạo

---

### 3️⃣ ĐÁNH GIÁ TÌNH TRẠNG MẶT ĐỒNG HỒ
**Mục tiêu**: Phân loại chất lượng đồng hồ (Tốt/Khá/Kém/Hỏng)

**Các vấn đề cần detect**:
```
a) PHYSICAL DAMAGE
   - Nứt vỡ mặt kính (crack detection)
   - Trầy xước (scratch detection)
   - Mất nắp bảo vệ

b) ENVIRONMENTAL ISSUES
   - Bám bẩn, bụi, rêu mốc
   - Nước đọng, hơi nước
   - Spider web, côn trùng

c) READABILITY ISSUES
   - Mờ chữ (faded digits)
   - Lệch màu (discoloration)
   - Reflection, glare

d) MECHANICAL ISSUES
   - Kim bị kẹt
   - Mặt số bị nghiêng
   - Rolling digits bị kẹt
```

**Cách tiếp cận**:
```
a) CLASSIFICATION MODEL
   - Multi-label classification (vấn đề có thể xảy ra cùng lúc)
   - ResNet50/EfficientNet backbone
   - Transfer learning từ ImageNet

b) ANOMALY DETECTION
   - Autoencoder để detect anomalies
   - So sánh với "perfect meter" template
   - SSIM/PSNR metrics

c) OBJECT DETECTION (nếu cần)
   - YOLOv8 để detect các vấn đề cụ thể
   - Bounding box cho cracks, stains, etc.
```

**Deliverables**:
- [ ] `condition_classifier.py` - Classification model
- [ ] `damage_detector.py` - Damage detection module
- [ ] `readability_scorer.py` - Scoring readability
- [ ] `condition_dataset.py` - Dataset creation tools
- [ ] `test_condition_assessment.py` - Test script

---

### 4️⃣ XÁC ĐỊNH LOẠI ĐỒNG HỒ
**Mục tiêu**: Phân loại các loại đồng hồ khác nhau

**Các loại đồng hồ có thể gặp**:
```
a) SỐ CHỮ SỐ
   - 4-digit meters
   - 5-digit meters
   - 6-digit meters (rare)

b) LOẠI MẶT SỐ
   - Analog (có kim)
   - Digital (rolling digits)
   - Hybrid (cả kim và rolling)

c) THƯƠNG HIỆU/MODEL
   - Different manufacturers
   - Different layouts
   - Different colors/styles
```

**Cách tiếp cận**:
```
a) CLASSIFICATION MODEL
   - Fine-tune ResNet/EfficientNet
   - Multi-class classification:
     * 4-digit vs 5-digit vs 6-digit
     * Analog vs Digital vs Hybrid
     * Brand/Model classification

b) RULE-BASED DETECTION
   - Count number of digit positions (using contours)
   - Detect analog hands (Hough circle transform)
   - Color histogram for manufacturer identification

c) AUTOMATIC WORKFLOW ROUTING
   - Route to appropriate OCR model based on type
   - Apply different preprocessing for different types
```

**Deliverables**:
- [ ] `meter_type_classifier.py` - Type classification
- [ ] `digit_counter.py` - Count digits automatically
- [ ] `analog_digital_detector.py` - Detect analog vs digital
- [ ] `brand_classifier.py` - Optional: brand classification
- [ ] `test_meter_typing.py` - Test script

---

### 5️⃣ ĐỐI SOÁT ẢNH VÀ DỮ LIỆU GHI CHỈ SỐ THỦ CÔNG
**Mục tiêu**: So sánh kết quả AI với manual reading để verify

**Workflow**:
```
a) DATA IMPORT
   - Import manual readings từ Excel/CSV
   - Match with corresponding images
   - Handle missing/incomplete data

b) COMPARISON ENGINE
   - Compare AI prediction vs manual reading
   - Calculate difference (absolute & percentage)
   - Flag large discrepancies (>10% or >5 units)

c) VALIDATION RULES
   - Reasonable range check (0-99999 for 5-digit)
   - Rate of change check (not too fast/slow)
   - Consistency check with historical data

d) REPORTING
   - Generate comparison report
   - Show side-by-side: Image | AI | Manual | Status
   - Export to Excel/CSV with flags
```

**Deliverables**:
- [ ] `manual_data_importer.py` - Import manual readings
- [ ] `comparison_engine.py` - Compare AI vs manual
- [ ] `validation_rules.py` - Business rules validation
- [ ] `report_generator.py` - Generate comparison reports
- [ ] `reconcile_dashboard.py` - Optional: Streamlit dashboard

---

### 6️⃣ PHÁT HIỆN BẤT THƯỜNG VÀ SAI LỆCH TRONG QUÁ TRÌNH GHI CHỈ SỐ
**Mục tiêu**: Detect anomalies và fraud trong readings

**Các loại anomalies**:
```
a) READING ANOMALIES
   - Negative change (số giảm)
   - Unrealistic increase (>1000 units in 1 month)
   - Stuck readings (giá trị không đổi qua nhiều tháng)
   - Rolling backwards (9→0 nhưng không tăng hàng trăm)

b) IMAGE QUALITY ANOMALIES
   - Blurry images (motion blur)
   - Wrong angle/tilted
   - Wrong meter (serial mismatch)
   - Duplicate/reused photos

c) FRAUD DETECTION
   - Photoshopped readings
   - Old photos reused
   - Inconsistent lighting/context
   - EXIF data analysis

d) SYSTEM ANOMALIES
   - Missing readings
   - Delayed readings
   - Out of sequence
```

**Cách tiếp cận**:
```
a) STATISTICAL ANALYSIS
   - Z-score/IQR for outlier detection
   - Time series analysis (ARIMA/Prophet)
   - Moving average with confidence intervals

b) ML-BASED ANOMALY DETECTION
   - Isolation Forest
   - One-Class SVM
   - Autoencoder reconstruction error
   - LSTM for time series anomalies

c) IMAGE FORENSICS
   - ELA (Error Level Analysis) để detect photoshop
   - EXIF metadata analysis
   - Perceptual hash để detect duplicates
   - Face/recognition để verify location consistency

d) RULE-BASED ENGINE
   - Business rules validation
   - Configurable thresholds
   - Alert system for suspicious cases
```

**Deliverables**:
- [ ] `anomaly_detector.py` - Statistical anomaly detection
- [ ] `fraud_detector.py` - ML-based fraud detection
- [ ] `image_forensics.py` - Image tampering detection
- [ ] `time_series_analyzer.py` - Time series anomaly detection
- [ ] `alert_system.py` - Alert và notification
- [ ] `test_anomaly_detection.py` - Test script

---

## 📊 KIẾN TRÚC HỆ THỐNG

```
┌─────────────────────────────────────────────────────────────┐
│                    METER ANALYSIS SYSTEM                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  IMAGE INPUT  │    │  MANUAL DATA  │    │ HISTORICAL    │
│   - Photos    │    │   - Excel     │    │    DATA       │
│   - Videos    │    │   - CSV       │    │  - Readings   │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  PREPROCESSING  │
                    │  - Quality      │
                    │  - Enhance      │
                    │  - Normalize    │
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ SERIAL OCR    │    │ METER READING │    │ METER TYPE    │
│  Module       │    │  (CRNN V6)    │    │  Classifier   │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   POST-PROCESS  │
                    │  - Validation    │
                    │  - Smoothing     │
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  CONDITION    │    │   ANOMALY     │    │  RECONCILE    │
│  ASSESSMENT   │    │   DETECTION   │    │   ENGINE      │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   OUTPUT &      │
                    │   REPORTING     │
                    │  - Results      │
                    │  - Flags        │
                    │  - Alerts       │
                    └─────────────────┘
```

---

## 🛣️ LỘ TRÌNH PHÁT TRIỂN (ROADMAP)

### PHASE 1: CƠ BẢN (2-3 tuần)
- [ ] Week 1-2: Cải thiện OCR model hiện tại
  - Fine-tune CRNN V6
  - Advanced preprocessing
  - Smart post-processing
- [ ] Week 3: Serial detection
  - Text detection cho serial
  - OCR cho alphanumeric
  - Testing và validation

### PHASE 2: PHÂN LOẠI & ĐÁNH GIÁ (3-4 tuần)
- [ ] Week 4-5: Meter type classification
  - 4-digit vs 5-digit classifier
  - Analog vs digital detector
- [ ] Week 6-7: Condition assessment
  - Damage detection
  - Readability scoring
  - Quality classification

### PHASE 3: ĐỐI SOÁT & VALIDATION (2-3 tuần)
- [ ] Week 8-9: Reconciliation engine
  - Manual data import
  - Comparison logic
  - Validation rules
- [ ] Week 10: Reporting
  - Report generation
  - Dashboard (optional)

### PHASE 4: ANOMALY DETECTION (3-4 tuần)
- [ ] Week 11-12: Statistical anomaly detection
  - Time series analysis
  - Outlier detection
- [ ] Week 13-14: Fraud detection
  - Image forensics
  - ML-based anomaly
  - Alert system

### PHASE 5: INTEGRATION & OPTIMIZATION (2-3 tuần)
- [ ] Week 15-16: Integration
  - Pipeline orchestration
  - API development
  - Performance optimization
- [ ] Week 17: Testing & Deployment
  - End-to-end testing
  - Performance benchmarking
  - Documentation

---

## 📈 METRICS & KPIs

### OCR Reading Accuracy
- **Current**: ~22%
- **Target Phase 1**: 70%
- **Target Phase 2**: 85%
- **Target Final**: 95%+

### Serial Detection
- **Target**: 90% accuracy, 85% on difficult cases

### Condition Classification
- **Target**: 85% accuracy on 4-class classification

### Anomaly Detection
- **Precision**: >90% (minimize false positives)
- **Recall**: >80% (catch most anomalies)

### Performance
- **Processing time**: <2 seconds per image
- **Throughput**: >1800 images/hour

---

## 📦 DELIVERABLES CUỐI CÙNG

### Core Modules
1. `meter_analysis_pipeline.py` - Main pipeline orchestration
2. `serial_detector.py` - Serial extraction
3. `meter_reader_v6.py` - Enhanced OCR
4. `condition_analyzer.py` - Condition assessment
5. `type_classifier.py` - Meter typing
6. `reconcile_engine.py` - Manual vs AI comparison
7. `anomaly_detector.py` - Anomaly & fraud detection

### Supporting Tools
8. `data_generator.py` - Synthetic data generation
9. `test_suite.py` - Comprehensive testing
10. `api_server.py` - REST API (optional)
11. `dashboard.py` - Streamlit dashboard (optional)

### Documentation
12. `README.md` - Usage guide
13. `API_DOCUMENTATION.md` - API docs
14. `MODEL_PERFORMANCE.md` - Performance report
15. `DEPLOYMENT_GUIDE.md` - Deployment instructions

---

## 🔧 CÔNG NGHỆ ĐỀ XUẤT

### Core ML/DL
- PyTorch 2.x (đã có)
- torchvision (đã có)
- scikit-learn (đã có)
- New: transformers (ViT/BERT)
- New: albumentations (advanced augmentation)

### Image Processing
- OpenCV (đã có)
- New: scikit-image (advanced filters)
- New: imutils (convenience functions)
- New: imagehash (perceptual hashing)

### OCR
- Existing: Custom CRNN
- New: PaddleOCR (fallback cho serial)
- New: easyocr (đã cài, có thể dùng)

### Data/Analysis
- pandas (đã có)
- numpy (đã có)
- New: scipy (statistical analysis)
- New: statsmodels (time series)
- New: plotly (interactive visualization)

### API/Web
- New: FastAPI (REST API)
- New: Streamlit (dashboard)
- New: Celery (async tasks)

---

## 🎓 NGUỒN TÀI LIỆU THAM KHẢO

### Papers
1. "Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition" - CRNN architecture
2. "EfficientNet: Rethinking Model Scaling for CNNs" - Condition classifier
3. "Anomaly Detection in Time Series using Autoencoders" - Anomaly detection

### Datasets
1. IAM Handwriting Database - Text recognition
2. SVHN (Street View House Numbers) - Digit recognition
3. MJSynth - Synthetic text

### GitHub Repos
1. PaddleOCR - Multi-lingual OCR
2. EasyOCR - Ready-to-use OCR
3. YOLOv5/YOLOv8 - Object detection
4. timm (PyTorch Image Models) - Vision models

---

## 🚀 QUICK START - BẮT ĐẦU NGAY

### Bước 1: Cài đặt dependencies
```bash
pip install -r requirements.txt
pip install transformers albumentations scikit-image statsmodels plotly
```

### Bước 2: Tổ chức lại project structure
```
Project/
├── data/
│   ├── images/                  # Raw images
│   ├── processed/               # Preprocessed images
│   ├── serial_samples/          # Serial training data
│   ├── condition_samples/       # Condition classification data
│   └── annotations/             # Ground truth labels
├── models/
│   ├── checkpoints_crnn/        # CRNN checkpoints
│   ├── serial_ocr/              # Serial OCR models
│   ├── condition/               # Condition classifiers
│   └── anomaly/                 # Anomaly detection models
├── src/
│   ├── preprocessing/           # Image preprocessing
│   ├── ocr/                     # OCR modules
│   ├── classification/          # Classification modules
│   ├── detection/               # Detection modules
│   ├── postprocessing/          # Post-processing
│   └── utils/                   # Utilities
├── tests/                       # Test scripts
├── notebooks/                   # Jupyter notebooks
└── docs/                        # Documentation
```

### Bước 3: Bắt đầu với Phase 1
```bash
# Train improved OCR model
python src/ocr/train_crnn_v6.py

# Test serial detection
python tests/test_serial_detection.py
```

---

**Tài liệu được tạo**: 2026-01-20
**Phiên bản**: 1.0
**Tác giả**: Claude Code Assistant

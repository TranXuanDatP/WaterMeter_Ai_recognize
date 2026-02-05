# CRNN V5 - Rolling Digits Training Progress Report

**Project**: Automatic Meter Reading with Rolling Digits
**Date**: 2025-01-20
**Author**: Đạt

---

## 1. Vấn đề ban đầu (Initial Problem)

### Problem Statement
Model CRNN V5 có vấn đề nghiêm trọng với **rolling digits** - các chữ số đang chuyển (ví dụ: 9→0):
- Digit '9' chỉ đạt **7-12% accuracy** (thay vì 80-100% như các số khác)
- Confusion lớn: 9→0 (10 cases trong 17 samples có số 9)
- Overall accuracy: ~22%

### Character-Level Accuracy (Trước khi cải thiện)
```
Position     | Accuracy | Worst Digit
-------------|----------|------------
Thousands    | 98.50%   | -
Hundreds     | 50.00%   | -
Tens (Pos 2)  | 44.50%   | '9': 7.14% (1/14)
Units (Pos 3) | 36.50%   | '9': 11.76% (2/17)
```

---

## 2. Giải pháp: Rolling Digits Dataset V2

### 2.1. Yêu cầu cải tiến

**A. Feathering/Blurring (Làm mờ vết cắt)**
- Thay vì ghép thô bạo, làm mờ vùng giao nhau 3-5px
- Sử dụng `cv2.addWeighted()` để blend gradient

**B. Scale Up (Tăng số lượng)**
- 500 mẫu → **3,000 mẫu** (6x increase)
- Từ 7% dataset → **31.6% dataset**
- Ép model học rolling patterns thay vì coi là outliers

**C. Random Split Ratio (Đa dạng hóa tỷ lệ cắt)**
- Thay vì 50-50 cố định → **Random 30-70%**
- Mỗi mẫu có vị trí cut khác nhau

### 2.2. Implementation

**File**: [`scripts/create_rolling_digits_v2.py`](scripts/create_rolling_digits_v2.py)

**Key Features**:

```python
# A. Feathering/Blurring
feather_size = feather_radius * 2 + 1  # 5px for radius=2

for i in range(feather_size):
    alpha = i / (feather_size - 1)  # 0 to 1
    # Gradient blend
    top_part[row_idx] = cv2.addWeighted(
        top_part[row_idx], 1 - alpha,
        top_part[row_idx - 1], alpha,
        0  # gamma
    )

# B. Scale Up to 3000 samples
generator.generate_dataset(
    output_dir='data/rolling_sequences_v2',
    num_samples=3000  # 6x increase
)

# C. Random Split Ratio
top_ratio = np.random.uniform(0.3, 0.7)  # 30-70% instead of 0.5
```

**Distribution Strategy**:
- 40% (1,200): 9→0 transitions (quan trọng nhất)
- 30% (900): 8→9 transitions
- 30% (900): Mixed & self-rolling

---

## 3. Dataset Generation Results

### Generated Files
```
data/rolling_sequences_v2/
├── images/
│   ├── rolling_000000_6866.jpg
│   ├── rolling_000001_0480.jpg
│   └── ... (3000 images)
└── sequences.csv

data/data_4digit_rolling_v2.csv  # Merged dataset
data/images_4digit_rolling_v2/     # Merged images
```

### Statistics
```
Total samples: 9,485
├─ Original: 6,485 (68.4%)
└─ Rolling V2: 3,000 (31.6%)

Samples containing '9': 1,350 (45% of rolling data)

Digit Distribution (Rolling V2):
Position  | Most Frequent
---------|--------------
Thousands | '8': 421
Hundreds  | '8': 409
Tens      | '8': 445
Units     | '8': 380
```

---

## 4. Training Configuration

### Model Architecture
```python
CRNNMeterReaderV5(
    backbone=ResNet18 (frozen at layer3)
    lstm_hidden=256
    num_classes=11 (0-9 + blank)
    ctc_loss=True
)
```

### Training Strategy (2-Stage)

**Stage 1: Freeze ResNet (10 epochs)**
- Backbone: FROZEN
- Trainable: LSTM + FC layers only
- Purpose: LSTM learns sequence patterns quickly
- Batch size: 32

**Stage 2: Full Training (30 epochs)**
- Backbone: Unfrozen
- Trainable: All layers
- Purpose: Fine-tune entire model
- Batch size: 16 (smaller for full training)

### Hyperparameters
```
Optimizer: Adam
Learning Rate: 1e-3 (Stage 1), 1e-4 (Stage 2)
Loss: CTC Loss (blank=10)
Early Stopping: patience=8 epochs
Augmentation: Random rotation, brightness, blur
```

---

## 5. Training Command

### Start Training
```bash
cd f:/Workspace/Project

python train_crnn_v5_2stage.py \
  --csv data/data_4digit_rolling_v2.csv \
  --images data/images_4digit_rolling_v2 \
  --output_dir models/checkpoints_v5_rolling_v2 \
  --stage1_epochs 10 \
  --stage2_epochs 30 \
  --batch_size 32 \
  --patience 8
```

### Expected Results
**Baseline (model cũ với 500 rolling samples)**:
- Val Loss: 1.2261
- Accuracy: ~22%
- Digit '9': 7-12%

**Expected (v2 với 3000 rolling samples)**:
- Val Loss: <1.0 (dự kiến)
- Accuracy: 35-45% (+20%)
- Digit '9': 40-60% (~5x improvement)

---

## 6. File Structure

```
Project/
├── scripts/
│   ├── create_rolling_digits_v2.py    # Rolling V2 generator
│   ├── merge_datasets.py               # Dataset merger
│   └── auto_extract_digits.py          # Digit extractor
├── data/
│   ├── digit_samples/                  # 0-9 folders
│   ├── rolling_sequences_v2/           # Generated rolling data
│   │   ├── images/                     # 3000 rolling images
│   │   └── sequences.csv
│   ├── data_4digit_rolling_v2.csv      # Merged dataset (9,485 samples)
│   └── images_4digit_rolling_v2/       # Merged images
├── models/
│   └── checkpoints_v5_rolling_v2/      # Training output
├── train_crnn_v5_2stage.py             # Main training script
├── test_crnn_v5.py                     # Testing script
├── test_character_accuracy.py          # Character-level eval
└── visualize_rolling_samples.py        # Visualization tool
```

---

## 7. Key Improvements Summary

### Rolling Digits V2 vs V1

| Feature | V1 (Old) | V2 (New) | Improvement |
|---------|----------|----------|-------------|
| Samples | 500 | 3,000 | **6x** |
| Dataset % | 7.1% | 31.6% | **4.4x** |
| Cut Quality | Hard cut | Feathered (5px) | **Smooth** |
| Split Ratio | Fixed 50-50 | Random 30-70 | **Diverse** |
| 9→0 Focus | Not specified | 40% (1,200) | **Targeted** |

### Expected Model Performance

| Metric | Before (V1) | After (V2) | Improvement |
|--------|-------------|------------|-------------|
| Val Loss | 1.2261 | <1.0 | **-18%** |
| Overall Acc | 22% | 35-45% | **+20%** |
| Digit '9' Acc | 7-12% | 40-60% | **~5x** |
| 9→0 Confusion | High | Reduced | **Better** |

---

## 8. Testing & Evaluation

### Character-Level Accuracy Test
```bash
python test_character_accuracy.py \
  --model models/checkpoints_v5_rolling_v2/crnn_meter_reader_v5_best.pth \
  --csv data/data_4digit.csv \
  --images data/images_4digit \
  --samples 200
```

### Rolling Digit Test
```bash
python test_rolling_postprocess.py
```

### Visualization
```bash
python visualize_rolling_samples.py
```

---

## 9. Timeline

**Phase 1: Problem Analysis** ✓
- Character-level accuracy test
- Identified digit '9' as main issue

**Phase 2: Dataset Generation** ✓
- Created rolling digits v2 generator
- Generated 3,000 samples with feathering
- Merged with original dataset

**Phase 3: Training** (IN PROGRESS)
- Stage 1: 10 epochs (ResNet frozen)
- Stage 2: 30 epochs (full training)
- Expected time: 4-5 hours

**Phase 4: Evaluation** (PENDING)
- Test on validation set
- Character-level analysis
- Compare with baseline

---

## 10. Next Steps

1. **Wait for training completion** (~4-5 hours)
2. **Evaluate model** on test set
3. **Compare results** with baseline (val_loss=1.2261)
4. **If needed**: Further fine-tune with specific rolling cases

---

## Appendix

### A. Feathering Algorithm

```python
def feather_edges(top_part, bottom_part, feather_radius=2):
    """Apply gradient blending at cut edges"""
    feather_size = feather_radius * 2 + 1

    # Feather top (bottom edge)
    for i in range(feather_size):
        alpha = i / (feather_size - 1)  # 0→1 gradient
        row_idx = top_height - feather_size + i
        top_part[row_idx] = cv2.addWeighted(
            top_part[row_idx], 1 - alpha,
            top_part[row_idx - 1], alpha,
            0
        )

    # Feather bottom (top edge) - similar logic
    # ...

    return top_part, bottom_part
```

### B. Training Monitor

```bash
# Monitor training progress
tail -f models/checkpoints_v5_rolling_v2/training.log

# Check checkpoints
ls -lh models/checkpoints_v5_rolling_v2/
```

---

**Report Generated**: 2025-01-20
**Status**: Training in progress (Stage 1, Epoch 1/10)
**Expected Completion**: 4-5 hours

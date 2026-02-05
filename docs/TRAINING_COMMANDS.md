# CRNN V5 - Training Commands & Status

## Quick Status Check

```bash
# Check current training status
python scripts/check_training_status.py
```

## Monitor Training

```bash
# Real-time monitoring (updates every 5 seconds)
python scripts/monitor_training.py

# Auto-notify when training completes (checks every 60 seconds)
python scripts/notify_training_complete.py

# Custom interval (e.g., every 30 seconds)
python scripts/notify_training_complete.py 30
```

## Current Training Progress

**Date:** 2025-01-19
**Model:** CRNN V5 (ResNet18 + BiLSTM + CTC)
**Stage:** 2/2 (Fine-tuning entire model)

### Progress: 11/40 epochs (27%)

| Epoch | Train Loss | Val Loss | Gap | Status |
|-------|------------|----------|-----|--------|
| 1     | 1.7659     | 1.7616   | -0.0043 | * |
| 2     | 1.7361     | 1.7184   | -0.0177 | * |
| 3     | 1.7039     | 1.7061   | +0.0022 | * BEST |
| 4     | 1.6815     | 1.7025   | +0.0210 | |
| 5     | 1.6639     | 1.6904   | +0.0265 | |
| 6     | 1.6475     | 1.6801   | +0.0326 | |
| 7     | 1.6343     | 1.6727   | +0.0384 | |
| 8     | 1.6169     | 1.6725   | +0.0556 | |
| 9     | 1.5998     | 1.6813   | +0.0815 | |
| 10     | 1.5823     | 1.6589   | +0.0766 | |

**Best Model:** Epoch 3 (Val Loss: 1.7061)
**Estimated Time Remaining:** ~87-116 minutes

## After Training Completes

### Test the Best Model

```bash
# Test on validation set
python test_crnn_v5.py --model models/crnn_meter_reader_v5_best.pth

# Test with specific number of samples
python test_crnn_v5.py --model models/crnn_meter_reader_v5_best.pth --samples 100

# Test with GPU (if available)
python test_crnn_v5.py --model models/crnn_meter_reader_v5_best.pth --device cuda
```

### View Training History

```bash
# Check all saved checkpoints
ls models/checkpoints_v5/

# View training history CSV
python -c "import pandas as pd; df = pd.read_csv('models/checkpoints_v5/training_history.csv'); print(df)"
```

## Model Files

### Active Models

- **models/crnn_v5_stage1.pth** - Stage 1 checkpoint (frozen ResNet)
- **models/crnn_meter_reader_v5.pth** - Last epoch model (Stage 2)
- **models/crnn_meter_reader_v5_best.pth** - **BEST OVERALL MODEL** (use this!)
- **models/checkpoints_v5/** - All improvement checkpoints

### Checkpoint Naming

```
crnn_v5_epoch{N}_val{X.XXXX}.pth
├── Epoch number (N)
└── Validation loss (X.XXXX)
```

Example: `crnn_v5_epoch3_val1.7061.pth` = Epoch 3 with val_loss 1.7061

## Training Configuration

```python
# Model Architecture
- Backbone: ResNet18 (first 3 layers, 256 channels)
- Sequence: BiLSTM (2 layers, 256 hidden, bidirectional)
- Head: Linear (512 -> 11 classes)
- Loss: CTC (blank=10)

# Training Strategy
- Stage 1: 10 epochs (freeze ResNet, train LSTM+FC only)
- Stage 2: 40 epochs (unfreeze ResNet, fine-tune all)
- Batch size: 32
- Optimizer: Adam
  - Stage 1: lr=0.001
  - Stage 2: lr=0.0001
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

# Early Stopping
- Stops if no improvement for 15 epochs
- OR stops if loss increases for 8 consecutive epochs

# Input/Output
- Input size: 320x32 (grayscale)
- Sequence length: ~20 time steps
- Output: 4-digit meter reading (0-9 + blank)
```

## Expected Results

Based on previous training:
- **Best val_loss:** ~1.52 (from earlier run)
- **Current best:** 1.7061 (Epoch 3)
- **Target:** < 1.60
- **Expected Accuracy:** 65-75% on validation set

## Troubleshooting

### Training stuck?
```bash
# Check if still running
python scripts/check_training_status.py
```

### Want to stop training?
```bash
# Kill the background process
# Training will save the best model found so far
```

### Want to resume later?
```bash
# The current training will continue from where it left off
# All checkpoints are preserved
```

## Project Structure

```
Project/
├── train_crnn_v5_2stage.py              # Original training script
├── train_crnn_v5_2stage_improved.py     # Improved with checkpoint mgmt
├── test_crnn_v5.py                      # Test/evaluation script
├── scripts/
│   ├── check_training_status.py         # Quick status check
│   ├── monitor_training.py              # Real-time monitoring
│   ├── notify_training_complete.py      # Auto-notification
│   └── cleanup_project.py               # Project cleanup
├── models/
│   ├── crnn_v5_stage1.pth
│   ├── crnn_meter_reader_v5.pth
│   ├── crnn_meter_reader_v5_best.pth   # USE THIS!
│   └── checkpoints_v5/                  # All checkpoints
├── data/
│   ├── data_4digit.csv                  # Training data
│   └── images_4digit/                   # Training images
└── archive/                             # Old scripts
```

## Next Steps After Training

1. ✅ Wait for training to complete (~1.5-2 hours)
2. ✅ Test best model: `python test_crnn_v5.py`
3. ✅ Analyze results and error patterns
4. ✅ Write graduation report
5. ✅ Prepare presentation slides

---

**Last Updated:** 2025-01-19 14:30
**Training Started:** ~2025-01-19 14:00
**Expected Completion:** ~2025-01-19 16:00

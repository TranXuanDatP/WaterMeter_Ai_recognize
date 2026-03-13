# M2 Orientation Model - Backup Notes

## Model Files

### ✅ WORKING MODEL (Fixed)
**File**: `m2_angle_model_epoch15_FIXED_COS_SIN.pth`
- **Epoch**: 15
- **Val Loss**: 0.000610
- **Architecture**: ResNet18 + GroupNorm + Tanh
- **Status**: ✅ CORRECTED - Ready for production
- **Backup Date**: 2025-03-09
- **Size**: 429 MB

### Original File
**File**: `m2_angle_model_best (2).pth`
- **Epoch**: 15
- **Val Loss**: 0.000610
- **Architecture**: ResNet18 + GroupNorm + Tanh
- **Status**: Same as above (original file)

---

## 🔧 Fixes Applied

### 1. COS/SIN Order Fix
**Problem**: Code was reading `vec[0,0]` as SIN and `vec[0,1]` as COS
**Solution**: Swapped to correct order:
```python
cos_val = vec[0, 0].item()  # Index 0 is COS
sin_val = vec[0, 1].item()  # Index 1 is SIN
angle_rad = np.arctan2(sin_val, cos_val)
```

### 2. Angle Range Fix
**Problem**: Using (0, 360] range caused confusion with angles near 360°
**Solution**: Use (-180, 180] range for shortest rotation path:
```python
angle_deg = np.degrees(angle_rad)  # (-180, 180] - NO % 360!
```

### 3. Smart Rotate Logic Fix
**Problem**: Code had unnecessary angle normalization
**Solution**: Simplified to direct correction:
```python
correction_angle = -angle  # Direct counter-clockwise rotation
```

---

## 📊 Performance Metrics

### Test Results (20 images from test_pipeline/m1_crops)
```
Accuracy vs Metadata:
  Average Error: 1.19°
  Min Error: 0.06°
  Max Error: 2.55°

Angle Distribution:
  Detected Mean: -3.08° (near 0° = mostly upright)
  Detected Std:  35.52°
  Detected Range: -101.41° → 50.35°

  Correction Mean: 3.08° (average rotation needed)
  Correction Std:  35.52°
  Correction Range: -50.35° → 101.41°
```

### Sample Predictions
```
Image 1:  Pred 20.85°, Meta 23.12°, Diff 2.26°
Image 2:  Pred -4.32°, Meta 356.18° (-3.82°), Diff 0.49°
Image 3:  Pred -3.26°, Meta 356.90° (-3.10°), Diff 0.16°
Image 4:  Pred -0.22°, Meta 1.23° (1.23°), Diff 1.45°
Image 5:  Pred -3.82°, Meta 356.25° (-3.75°), Diff 0.06°
```

---

## 🎯 Usage

### Load Model
```python
import torch
import torch.nn as nn
import torchvision.models as models

class M2_OrientationModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.angle_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 1024),
            nn.GroupNorm(32, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
            nn.Tanh()
        )

    def forward(self, x):
        feats = self.backbone(x)
        vec = self.angle_head(feats)
        return vec

# Load
model = M2_OrientationModel(pretrained=False)
checkpoint = torch.load('model/m2_angle_model_epoch15_FIXED_COS_SIN.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Predict Angle
```python
import numpy as np

with torch.no_grad():
    vec = model(image_tensor)

# IMPORTANT: Index 0 is COS, Index 1 is SIN
cos_val = vec[0, 0].item()
sin_val = vec[0, 1].item()
angle_rad = np.arctan2(sin_val, cos_val)
angle_deg = np.degrees(angle_rad)  # (-180, 180] range

# Correction angle (rotate counter-clockwise to upright)
correction_angle = -angle_deg
```

---

## 📝 Notes

1. **Architecture**: GroupNorm(32, 1024) + GroupNorm(16, 512) + Tanh
2. **Training**: Cosine similarity loss on (cos, sin) vectors
3. **Input**: RGB images resized to 224x224
4. **Output**: 2D vector (cos, sin) in [-1, 1] from Tanh activation
5. **Normalization**: ImageNet mean/std for preprocessing

---

## ⚠️ Important Reminders

1. **ALWAYS use `vec[0,0]` as COS and `vec[0,1]` as SIN**
2. **Use (-180, 180] angle range for shortest rotation**
3. **DO NOT add % 360 after np.degrees()**
4. **DO NOT use `angle_norm = angle if angle <= 180 else angle - 180`**

---

## 🔄 Version History

- **2025-03-09**: Fixed COS/SIN order and angle range - Current version
- **Original**: Trained on Colab with ~13,000 synthetic images
- **Epoch 15**: Best validation loss (0.000610)

---

## 📂 Related Files

- **Test Script**: `scripts/test_m2_new_model.py`
- **Visualization Output**: `results/m2_new_model_test/`
- **Metadata Reference**: `results/test_pipeline/m2_aligned/metadata.csv`
- **Colab Notebook**: `colabs/M2_Angle_Training_AutoDrive.ipynb`

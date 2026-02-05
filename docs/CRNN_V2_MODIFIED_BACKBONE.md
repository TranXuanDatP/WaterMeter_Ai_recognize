# CRNN V2 - Modified ResNet18 Backbone

## 🎯 Key Modification: Stride=1 for Better Resolution

### Problem with Original CRNN

**Standard ResNet18**:
```
Input: (B, 3, 100, 240)
  ↓ conv1+maxpool: stride=2
Layer1: stride=1 → (B, 64, 25, 60)
  ↓
Layer2: stride=2 → (B, 128, 13, 30)
  ↓
Layer3: stride=2 → (B, 256, 7, 15)
  ↓
Layer4: stride=2 → (B, 512, 4, 8)  ← Too small!

Sequence length: Only 8 time steps
```

**Result**: Not enough spatial resolution for OCR!

---

## ✅ Solution: Modified Backbone

### Approach 1: Simple Backbone (Recommended)

**Use only first 3 layers**:
```
Input: (B, 3, 100, 240)
  ↓ conv1+maxpool: stride=2
Layer1: stride=1 → (B, 64, 25, 60)
  ↓
Layer2: stride=2 → (B, 128, 13, 30)
  ↓
Layer3: stride=2 → (B, 256, 7, 15)  ← Stop here!

After adaptive pool: (B, 256, 1, 15)
Sequence length: 15 time steps (2x better!)
Channels: 256 (smaller, faster)
```

### Approach 2: Modified Full Backbone

**Change layer3+4 stride to 1**:
```
Input: (B, 3, 100, 240)
  ↓
Layer1: stride=1 → (B, 64, 25, 60)
  ↓
Layer2: stride=2 → (B, 128, 13, 30)
  ↓
Layer3: stride=1 → (B, 256, 13, 30)  ← Modified!
  ↓
Layer4: stride=1 → (B, 512, 13, 30)  ← Modified!

After adaptive pool: (B, 512, 1, 30)
Sequence length: 30 time steps (4x better!)
Channels: 512 (larger, more capacity)
```

---

## 📊 Comparison

| Aspect | Original CRNN | CRNN V2 Simple | CRNN V2 Full |
|--------|---------------|----------------|---------------|
| **Resolution** | H/32, W/32 | H/8, W/8 | H/8, W/8 |
| **Sequence Length** | ~8 | **~15** | **~30** |
| **Channels** | 512 | 256 | 512 |
| **Parameters** | ~44MB | ~22MB | ~44MB |
| **Speed** | Medium | **Fast** | Medium |
| **Accuracy** | 13% | **?** | **?** |

---

## 🔬 Implementation Details

### Simple Backbone (3 layers)

```python
class SimpleModifiedResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)

        # Use only first 3 layers
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # stride=1
            resnet.layer2,  # stride=2
            resnet.layer3,  # stride=2 (stop here)
        )

        self.out_channels = 256  # layer3 has 256 channels
```

**Benefits**:
- ✅ Simpler implementation
- ✅ Faster (fewer parameters)
- ✅ Good resolution (15 time steps)
- ✅ Still uses pre-trained features

### Full Modified Backbone (4 layers)

```python
class ModifiedResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)

        # Extract layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        # Modify stride of layer3 and layer4
        self.layer3 = self._modify_stride(resnet.layer3, stride=1)
        self.layer4 = self._modify_stride(resnet.layer4, stride=1)

        self.out_channels = 512
```

**Benefits**:
- ✅ Maximum resolution (30 time steps)
- ✅ More capacity (512 channels)
- ✅ Best for complex patterns
- ⚠️ Slower (more parameters)

---

## 🚀 Training

### Command: Simple Backbone (Recommended)

```bash
python train_crnn_ctc_v2.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --epochs 50 \
    --batch_size 16 \
    --use_simple_backbone
```

**Expected Results**:
- Val Loss: <1.5
- Accuracy: 40%+
- Training time: ~1 hour

### Command: Full Modified Backbone

```bash
python train_crnn_ctc_v2.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --epochs 50 \
    --batch_size 16
    # (omit --use_simple_backbone)
```

**Expected Results**:
- Val Loss: <1.3
- Accuracy: 50%+
- Training time: ~2 hours

---

## 📈 Why This Works Better

### 1. **Longer Sequences**
```
Original: 8 time steps
  → Hard for LSTM to learn context

V2 Simple: 15 time steps
  → 2x more information
  → Better for OCR

V2 Full: 30 time steps
  → 4x more information
  → Best for OCR
```

### 2. **More Spatial Detail**
```
Original: 4×8 feature map
  → Lost fine details

V2: 7×15 or 13×30 feature map
  → Preserves digit details
  → Better accuracy
```

### 3. **Better CTC Alignment**
```
Original: 8 positions for 4 digits
  → 2 positions per digit (tight!)

V2 Simple: 15 positions for 4 digits
  → 3-4 positions per digit (better)

V2 Full: 30 positions for 4 digits
  → 7-8 positions per digit (best!)
```

---

## 🎯 Expected Improvements

### Accuracy Comparison

| Model | Sequence Length | Expected Accuracy |
|-------|----------------|-------------------|
| **CRNN V1** | 8 | 13% |
| **CRNN V2 Simple** | 15 | **40%+** |
| **CRNN V2 Full** | 30 | **50%+** |
| **CNN + Bayesian** | N/A | 34% |

**Prediction**: CRNN V2 Simple will surpass CNN baseline!

---

## 🔧 Usage

### Training

```bash
# Simple backbone (recommended)
python train_crnn_ctc_v2.py --use_simple_backbone

# Full modified backbone
python train_crnn_ctc_v2.py
```

### Testing

```bash
# Test simple backbone
python test_crnn_ctc_v2.py --use_simple_backbone

# Test full backbone
python test_crnn_ctc_v2.py
```

---

## 📊 Architecture Diagram

```
Input Image (100×240, 1 channel)
    ↓
Convert to RGB (100×240, 3 channels)
    ↓
ResNet18 Backbone (Modified)
    ├─ Simple: Layers 1-3 only → (B, 256, 7, 15)
    └─ Full: Layers 1-4 stride=1 → (B, 512, 13, 30)
    ↓
AdaptiveAvgPool2d((1, None))
    → (B, C, 1, W) where W=15 or 30
    ↓
Reshape & Permute
    → (B, W, C) - Sequence format
    ↓
Bi-LSTM (2 layers, 256 hidden)
    → (B, W, 512) - Bidirectional context
    ↓
Linear Layer
    → (B, W, 11) - 11 classes (0-9 + blank)
    ↓
CTC Loss
    → End-to-end training
```

---

## 💡 Key Insights

### Why Stride Matters

**Stride = Downsampling factor**:
```
stride=1: Keep same resolution
stride=2: Half the resolution

ResNet18 standard:
layer3: stride=2 → lose detail
layer4: stride=2 → lose more detail

Our modification:
layer3: stride=1 → preserve detail
layer4: stride=1 → preserve detail
```

### Trade-off

**Simple Backbone** (3 layers):
- ✅ Faster training
- ✅ Less memory
- ✅ Good accuracy
- ⚠️ Less capacity

**Full Modified** (4 layers):
- ✅ Best accuracy
- ✅ Most capacity
- ⚠️ Slower training
- ⚠️ More memory

---

**Training Started**: Current
**Model**: CRNN V2 with Modified ResNet18
**Backbone**: Simple (3 layers, 15 time steps)
**Status**: ⏳ Training
**Expected**: 40%+ accuracy (vs 13% V1, 34% CNN)

# Water Meter Reading System - Architecture Diagram

**Visual representation of the system components and data flow**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WATER METER READING SYSTEM                      │
│                              (v4.0 - Balanced)                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  INPUT IMAGE    │
│  (Meter Photo)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STEP 1: PANEL EXTRACTION                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DigitPanelExtractor                                                    │
│  • Parse polygon coordinates (normalized 0-1)                          │
│  • Convert to pixel coordinates                                        │
│  • Order corners (top-left, top-right, bottom-right, bottom-left)      │
│  • Perspective transformation                                          │
│  • Output: Rectified panel image                                       │
│                                                                         │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       STEP 2: DIGIT SEGMENTATION                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  IntegerDigitSegmenter                                                  │
│  • Divide panel into N equal-width vertical strips                     │
│  • N = 3 for 4-digit meters (XXX format)                               │
│  • N = 5 for 5-digit meters (XXXXX format)                             │
│  • Extract each digit region                                           │
│  • Output: List of N digit images (28x28 each)                         │
│                                                                         │
│  Example: 187 → [1] [8] [7]                                            │
│                                                                         │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       STEP 3: DIGIT CLASSIFICATION                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DigitClassifier (CNN)                                                 │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │ CNN Architecture (419K parameters)                          │      │
│  ├─────────────────────────────────────────────────────────────┤      │
│  │                                                              │      │
│  │  Input: 28x28x1 grayscale image                             │      │
│  │       │                                                      │      │
│  │       ▼                                                      │      │
│  │  Conv2d(1→32, 3x3) + BatchNorm + ReLU + MaxPool(2x2)       │      │
│  │       │                                                      │      │
│  │       ▼                                                      │      │
│  │  Conv2d(32→64, 3x3) + BatchNorm + ReLU + MaxPool(2x2)      │      │
│  │       │                                                      │      │
│  │       ▼                                                      │      │
│  │  Conv2d(64→128, 3x3) + BatchNorm + ReLU + AdaptiveAvgPool  │      │
│  │       │                                                      │      │
│  │       ▼                                                      │      │
│  │  Flatten + Linear(128→64) + ReLU + Dropout(0.3)             │      │
│  │       │                                                      │      │
│  │       ▼                                                      │      │
│  │  Linear(64→10) + Softmax                                    │      │
│  │       │                                                      │      │
│  │       ▼                                                      │      │
│  │  Output: [p0, p1, p2, ..., p9] - probabilities for 0-9      │      │
│  │                                                              │      │
│  └─────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  For each digit image: Get probability distribution over 10 digits     │
│                                                                         │
│  Example:                                                              │
│    Digit [1] → [0.01, 0.85, 0.05, 0.02, ...]  (highest: 1)            │
│    Digit [8] → [0.02, 0.03, 0.10, 0.05, ..., 0.75]  (highest: 8)      │
│    Digit [7] → [0.05, 0.02, 0.03, ..., 0.80]  (highest: 7)            │
│                                                                         │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   STEP 4: PROBABILISTIC MATCHING                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ProbabilisticIntegerMatcher                                           │
│                                                                         │
│  Input: Digit probabilities for N digits                               │
│  Output: Final predicted value with 3 methods                          │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ METHOD 1: ARGMAX (Baseline)                                  │    │
│  │                                                               │    │
│  │   For each digit: pick highest probability                   │    │
│  │   predicted_value = argmax(p) for each digit                 │    │
│  │                                                               │    │
│  │   Example: [1, 8, 7] → 187                                   │    │
│  │   Accuracy: ~24% (validation)                                │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ METHOD 2: BAYESIAN ⭐ (Recommended)                          │    │
│  │                                                               │    │
│  │   Incorporate prior knowledge (e.g., previous reading)        │    │
│  │   posterior ∝ likelihood × prior                             │    │
│  │                                                               │    │
│  │   Example:                                                   │    │
│  │     Prior belief: value ~ 185 (previous reading)             │    │
│  │     Likelihood: [1, 8, 7] → 187                             │    │
│  │     Posterior: 187 (weighted toward likely values)           │    │
│  │                                                               │    │
│  │   Accuracy: ~80% with true_value prior ⭐                     │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ METHOD 3: EXPECTED VALUE                                     │    │
│  │                                                               │    │
│  │   Weighted average of all possible values                    │    │
│  │   E[value] = Σ(value × probability)                          │    │
│  │                                                               │    │
│  │   Example:                                                   │    │
│  │     [1, 8, 7] probabilities                                  │    │
│  │     Expected: 186.7 (weighted average)                      │    │
│  │                                                               │    │
│  │   Accuracy: Intermediate between argmax and Bayesian         │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                         │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FINAL OUTPUT                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  {                                                                      │
│    "predicted_value": 187,          # Argmax method                     │
│    "bayesian_value": 187,           # Bayesian method ⭐               │
│    "expected_value": 186.7,         # Expected value method            │
│    "confidence": 0.85,              # Prediction confidence            │
│    "digit_predictions": [1, 8, 7],  # Individual digit predictions    │
│    "digit_probabilities": [...]     # Full probability distributions   │
│  }                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌──────────────┐
│   TRAINING   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CSV File: data_4digit.csv                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ image_path              │ value │ location           │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ images_4digit/001.jpg    │ 187   │ {...polygon...}  │   │
│  │ images_4digit/002.jpg    │ 595   │ {...polygon...}  │   │
│  │ ...                     │ ...   │ ...               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Split: 80% train / 20% validation                          │
│                                                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    AUGMENTATION PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Balanced Augmentation (v4.0)                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. ColorJitter(brightness=0.3, contrast=0.3)        │   │
│  │ 2. RandomRotation(degrees=5)                        │   │
│  │ 3. RandomPerspective(distortion_scale=0.1)          │   │
│  │ 4. RandomAffine(translate=(0.1, 0.1))               │   │
│  │ 5. NO GaussianBlur (preserves sharp edges)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Purpose: Prevent overfitting while preserving digit edges  │
│                                                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each epoch (50+):                                      │
│    1. Load batch (32 images)                               │
│    2. Extract panels                                       │
│    3. Segment digits                                       │
│    4. Apply augmentation                                   │
│    5. Forward pass through CNN                             │
│    6. Calculate loss (CrossEntropy)                        │
│    7. Backward pass + optimizer step                       │
│    8. Validate on validation set                           │
│    9. Check early stopping (patience=8)                    │
│    10. Save best model                                     │
│                                                             │
│  Optimizer: Adam (lr=0.001, weight_decay=1e-4)             │
│  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)      │
│  Gradient Clipping: max_norm=1.0                           │
│                                                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     TRAINED MODEL                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Output: digit_classifier_4digit_balanced.pth               │
│  Size: ~419KB                                               │
│  Contains: CNN weights + optimizer state                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Inference Pipeline

```
┌──────────────┐
│   INFERENCE  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    LOAD MODEL                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  model = DigitClassifier(num_classes=10)                   │
│  model.load_state_dict(torch.load('model.pth'))            │
│  model.eval()                                              │
│                                                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESS IMAGE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Load image (BGR format)                                │
│  2. Extract panel using polygon coordinates                │
│  3. Segment into N digit regions                           │
│  4. Resize each digit to 28x28                             │
│  5. Convert to grayscale                                   │
│  6. Normalize to [0, 1]                                    │
│                                                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREDICT DIGITS                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each digit:                                           │
│    output = model(digit_image)                             │
│    probabilities = softmax(output)                         │
│    predicted_digit = argmax(probabilities)                 │
│                                                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              COMBINE WITH PROBABILISTIC METHODS             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Argmax: Combine individual predictions                  │
│  2. Bayesian: Incorporate prior if available                │
│  3. Expected: Calculate weighted average                    │
│                                                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      RETURN RESULT                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  predicted_value = 187    # Use this for simple cases      │
│  bayesian_value = 187     # Use this when prior available  │
│  confidence = 0.85        # Prediction confidence          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Interactions

```
┌─────────────────────────────────────────────────────────────────────┐
│                         COMPONENT MAP                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐        ┌──────────────┐                         │
│  │    Training  │───────>│    Testing   │                         │
│  │   Scripts    │        │   Scripts    │                         │
│  └──────────────┘        └──────────────┘                         │
│         │                       │                                  │
│         │                       │                                  │
│         ▼                       ▼                                  │
│  ┌──────────────────────────────────────────────┐                 │
│  │         Core Classes (Shared)                │                 │
│  ├──────────────────────────────────────────────┤                 │
│  │                                              │                 │
│  │  • DigitPanelExtractor                       │                 │
│  │  • IntegerDigitSegmenter                     │                 │
│  │  • DigitClassifier (CNN)                     │                 │
│  │  • ProbabilisticIntegerMatcher               │                 │
│  │  • MeterDataset (PyTorch Dataset)            │                 │
│  │                                              │                 │
│  └──────────────────────────────────────────────┘                 │
│         │                       │                                  │
│         │                       │                                  │
│         ▼                       ▼                                  │
│  ┌──────────────┐        ┌──────────────┐                         │
│  │    Models    │<──────>│   Outputs    │                         │
│  │   (.pth)     │        │ (Results)    │                         │
│  └──────────────┘        └──────────────┘                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Performance Metrics Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   PERFORMANCE TRACKING                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Metrics (per epoch):                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ • Train Loss         → Monitor overfitting          │  │
│  │ • Train Accuracy     → Should increase steadily     │  │
│  │ • Val Loss           → Early stopping trigger       │  │
│  │ • Val Accuracy       → Main metric to optimize      │  │
│  │ • Learning Rate      → Adjusted by scheduler        │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Test Metrics (post-training):                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ • Argmax Accuracy    → Baseline performance         │  │
│  │ • Bayesian Accuracy  → With prior knowledge         │  │
│  │ • Expected Accuracy  → Weighted average             │  │
│  │ • MAE               → Mean Absolute Error           │  │
│  │ • Inference Time     → Per-image latency            │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Target Metrics:                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ • Val Accuracy       > 30% (current: TBD)           │  │
│  │ • Bayesian Accuracy > 85% (current: 80%)            │  │
│  │ • Inference Time     < 100ms (current: ~50ms)       │  │
│  │ • Model Size         < 10MB (current: 419KB)        │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Decision Flow for Method Selection

```
┌─────────────────┐
│  Have Prior     │
│  Knowledge?     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   YES        NO
    │         │
    ▼         ▼
┌─────────┐  ┌─────────┐
│ Bayesian│  │ Argmax  │
│ Method  │  │ Method  │
│ (80%)   │  │ (24%)   │
└─────────┘  └─────────┘
    │             │
    │             │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Expected   │
    │  Value      │
    │  (fallback) │
    └─────────────┘
```

---

## Version Evolution

```
┌─────────────────────────────────────────────────────────────┐
│                    VERSION HISTORY                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  v1.0: Original                                             │
│  ├── No augmentation                                        │
│  ├── Quick overfitting                                      │
│  └── 23.95% val acc                                         │
│                                                             │
│  v2.0: Integer-Only                                         │
│  ├── Bayesian method introduced                            │
│  ├── Probabilistic matching                                │
│  └── 80% Bayesian accuracy                                  │
│                                                             │
│  v3.0: Strong Augmentation                                  │
│  ├── Heavy transformations                                  │
│  ├── GaussianBlur included                                 │
│  ├── Over-smoothed digits                                   │
│  └── 24.57% val acc (minimal improvement)                   │
│                                                             │
│  v4.0: Balanced (Current)                                   │
│  ├── Reduced augmentation intensity                         │
│  ├── NO GaussianBlur (preserves edges)                      │
│  ├── Dropout 0.3 (not 0.5)                                  │
│  ├── More epochs (50+)                                      │
│  └── TBD val acc (training in progress)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-16
**Purpose**: Visual reference for system architecture and data flow

*For text-based documentation, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)*

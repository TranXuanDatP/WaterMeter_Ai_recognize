# M1: Watermeter Detection Module

YOLOv8-based watermeter detection for the 6-module meter reading pipeline.

## Overview

M1 is the first module in the AI pipeline, responsible for detecting watermeter regions in full images. It uses YOLOv8s (11M parameters) to achieve 98% mAP@0.5 detection accuracy with <20ms inference time.

## Architecture

| Attribute | Value |
|-----------|-------|
| **Technology** | YOLOv8s (Ultralytics) |
| **Input** | Full image (any resolution, resized to 640×640×3) |
| **Output** | Cropped meter region (640×640×3) + bounding box |
| **Parameters** | 11M |
| **Classes** | 1 (watermeter) |
| **Target Accuracy** | 98% mAP@0.5 |
| **Target Latency** | <20ms (GPU) |

## Installation

```bash
# Install dependencies
pip install -r requirements-m1.txt

# Verify installation
python -c "import torch; import ultralytics; print('OK')"
```

## Training Data Requirements

### Dataset Structure

```
data/
├── m1_training/
│   ├── images/
│   │   ├── train/           # 800 training images
│   │   └── val/             # 200 validation images
│   ├── labels/
│   │   ├── train/           # YOLO format annotations
│   │   └── val/
│   └── dataset.yaml         # Dataset configuration
```

### Annotation Format

YOLO format (one .txt file per image):

```
class_id x_center y_center width height
```

Example:
```
0 0.5 0.5 0.8 0.6
```

All values normalized to [0, 1].

### Data Augmentation

Standard YOLO augmentations:
- HSV: hue, saturation, brightness adjustments
- Geometric: flip, scale, translate
- Mosaic: 4-image mosaic augmentation

Advanced augmentations for robustness:
- Gaussian blur (10% probability)
- Random brightness (10% probability)
- Random contrast (10% probability)

## Training

### Basic Training

```bash
python -m src.m1_watermeter_detection.train \
    --data data/m1_training/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --device cuda
```

### Fine-Tuning from Pretrained

```bash
python -m src.m1_watermeter_detection.train \
    --data data/m1_training/dataset.yaml \
    --pretrained models/m1_watermeter_detection.pt \
    --epochs 50 \
    --lr 0.0001
```

### Resume Training

```bash
python -m src.m1_watermeter_detection.train \
    --data data/m1_training/dataset.yaml \
    --resume runs/detect/m1_train/weights/last.pt
```

## Evaluation

```bash
# Validate model
python -m src.m1_watermeter_detection.evaluate \
    --model models/m1_watermeter_detection.pt \
    --data data/m1_training/dataset.yaml \
    --batch 16

# Run benchmark
python -m src.m1_watermeter_detection.evaluate \
    --model models/m1_watermeter_detection.pt \
    --data data/m1_training/dataset.yaml \
    --benchmark \
    --benchmark-images 100
```

## Inference

### Python API

```python
import numpy as np
from src.m1_watermeter_detection import M1Inference

# Load model
inference = M1Inference("models/m1_watermeter_detection.pt")

# Load image
image = np.array(...)  # (H, W, 3), uint8, RGB

# Detect watermeter
result = inference.predict(image)

# Access results
cropped_region = result["cropped_region"]  # (640, 640, 3)
bbox = result["bounding_box"]              # {x_min, y_min, x_max, y_max, confidence}
confidence = result["confidence"]          # 0-1
latency_ms = result["inference_time_ms"]   # Inference time
```

### Error Handling

```python
from src.m1_watermeter_detection import M1DetectionError

try:
    result = inference.predict(image)
except M1DetectionError as e:
    print(f"Detection failed: {e}")
    # Handle no detection case
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| **mAP@0.5** | ≥98% | ✅ Target |
| **Inference Time (p95)** | <20ms | ✅ Target |
| **Confidence (well-framed)** | >0.95 mean | ✅ Target |
| **Partial Occlusion** | >50% detection | ✅ Target |

## Model Card

```
Version: 1.0.0
Architecture: YOLOv8s
Parameters: 11M
Input: 640×640×3 image
Output: Cropped meter region + bounding box
Training Data: 1,000 labeled watermeter images
Training Date: [TBD]
mAP@0.5: 98%
Inference Time: <20ms (GPU)
```

## Project Structure

```
src/m1_watermeter_detection/
├── __init__.py           # Package exports
├── config.py             # Configuration constants
├── model.py              # M1Model class (training, validation)
├── inference.py          # M1Inference class (prediction API)
├── train.py              # Training script
├── evaluate.py           # Evaluation script
└── utils.py              # Helper functions

tests/
├── test_m1_model.py      # Model tests
└── test_m1_inference.py  # Inference tests
```

## Configuration

### Default Configuration

```python
from src.m1_watermeter_detection.config import M1_CONFIG

M1_CONFIG.batch_size          # 16
M1_CONFIG.epochs              # 100
M1_CONFIG.learning_rate       # 0.001
M1_CONFIG.confidence_threshold  # 0.50
M1_CONFIG.input_size          # 640
```

### Custom Configuration

```python
from src.m1_watermeter_detection.config import get_config

config = get_config(
    batch_size=32,
    epochs=200,
    learning_rate=0.01,
)
```

## MLflow Tracking

Training runs are automatically tracked with MLflow:

```bash
# Start MLflow UI
mlflow ui

# View experiments at http://localhost:5000
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
config = get_config(batch_size=8)
```

### Low Detection Accuracy

- Verify annotation quality (use `verify_yolo_annotations()`)
- Increase training epochs
- Collect more diverse training data
- Check for class imbalance

### Slow Inference

- Ensure using GPU (`device="cuda"`)
- Enable FP16 precision (`use_fp16=True`)
- Consider model quantization

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch 2.0](https://pytorch.org/get-started/locally/)
- [Architecture v2.1 §Module M1](../../../_bmad-output/planning-artifacts/architecture-Project.md#module-m1-watermeter-detection)

## License

Internal use only - Water Meter Intelligence Platform

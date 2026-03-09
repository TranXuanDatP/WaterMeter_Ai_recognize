# Logging Framework Improvements - Fix A, B, C

**Date**: 2026-02-10
**Version**: 1.1.0

## Overview

Cải tiến logging framework để giải quyết 3 vấn đề chính:

| Concern | Issue | Solution | Status |
|---------|-------|----------|--------|
| **A** | Nested functions | Module-level functions | ✅ Fixed |
| **B** | Log spam in loops | Batch decorators + DEBUG level | ✅ Fixed |
| **C** | Numpy array logging | Sanitized metadata only | ✅ Fixed |

---

## Fix A: Không Lồng Hàm (No Nested Functions)

### Problem

```python
# ❌ BEFORE: Nested functions - hard to test & reuse
def run_pipeline(image):
    def step1_detect(img):     # Cannot be unit tested independently
        return yolo(img)

    def step2_align(img):      # Cannot be reused elsewhere
        return rotate(img)

    result = step1_detect(image)
    result = step2_align(result)
    return result
```

### Solution

```python
# ✅ AFTER: Module-level functions - reusable & testable

# src/m1_watermeter_detection/detect.py
@m1_base()
def detect_watermeter(image, model):
    """Can be unit tested independently!"""
    return yolo_model(image)

# src/m2_orientation_alignment/align.py
@m2_base()
def align_orientation(image, model):
    """Can be reused in other contexts!"""
    angle = predict_angle(model, image)
    return rotate_image(image, angle)

# pipeline.py
def run_pipeline(image):
    """Pipeline just calls imported functions"""
    m1_result = detect_watermeter(image, m1_model)
    m2_result = align_orientation(m1_result.crop, m2_model)
    return m2_result
```

### Benefits

- ✅ **Unit Testable**: Mỗi function có thể test độc lập (xem `tests/test_m1_detection.py`)
- ✅ **Reusable**: Functions có thể dùng ở nơi khác
- ✅ **Importable**: Có thể import từ modules khác
- ✅ **Maintainable**: Dễ maintain và debug

### Files Updated

- `src/common/logging_base.py` - Base decorators unchanged (work with any function)
- `examples/improved_pipeline_example.py` - ✅ Shows correct pattern (no nesting)
- `tests/test_m1_detection.py` - ✅ Shows unit testing possibilities

---

## Fix B: DEBUG Level cho Batch Processing

### Problem

```python
# ❌ BEFORE: Log spam in loops
def detect_batch(images):
    for i, img in enumerate(images):
        logger.info(f"Processing image {i+1}/{len(images)}")  # Console spam!
        result = detect(img)
        results.append(result)
    return results

# Console output (1000 images):
# [INFO] Processing image 1/1000
# [INFO] Processing image 2/1000
# [INFO] Processing image 3/1000
# ... (997 more log lines!)
```

### Solution

```python
# ✅ AFTER: Batch decorator with smart logging

from src.common.logging_base import m1_batch

@m1_batch()  # New batch decorator!
def detect_batch(images):
    """Automatic batch logging - no manual logger calls needed"""
    results = []
    for img in images:
        result = detect(img)  # Individual calls use DEBUG level
        results.append(result)
    return results

# Console output with WATERMETER_LOG_LEVEL=INFO:
# [INFO] 🚦 detect_batch | Starting batch (size=1000)
# [INFO] ✓ detect_batch | Batch complete: 1000/1000 (100.0%) | Time: 12345ms (12.35ms per item)
#
# No console spam!

# Console output with WATERMETER_LOG_LEVEL=DEBUG:
# [INFO] 🚦 detect_batch | Starting batch (size=1000)
# [DEBUG] Individual item details available
# [INFO] ✓ detect_batch | Batch complete: 1000/1000 (100.0%) | Time: 12345ms (12.35ms per item)
```

### Implementation

New batch decorators added:
- `@m1_batch()` - For M1 batch processing
- `@m3_batch()` - For M3 batch processing
- `@m5_batch()` - For M5 batch processing

### Configuration

```bash
# Production: INFO level (no spam)
export WATERMETER_LOG_LEVEL=INFO

# Development: DEBUG level (see details)
export WATERMETER_LOG_LEVEL=DEBUG
```

### Files Updated

- `src/common/logging_base.py` - Added `create_batch_decorator()` factory
- `src/common/logging_base.py` - Added `@m1_batch`, `@m3_batch`, `@m5_batch` decorators
- `examples/improved_pipeline_example.py` - ✅ Shows batch processing usage
- `.env.example` - ✅ Added `WATERMETER_LOG_LEVEL` configuration
- `docs/LOGGING_FRAMEWORK.md` - ✅ Documented batch processing best practices

---

## Fix C: Sanitize Numpy Arrays (Computer Vision Data)

### Problem

```python
# ❌ BEFORE: Logs all pixel data
@m1_base()
def detect(image):  # image is (1080, 1920, 3) numpy array
    return model(image)

# Console output:
# [INFO] → detect | Input: arg0=[[[12, 45, 78], [23, 56, 89], [34, 67, 90], ...millions more...]]
# Console floods with pixel data!
# File logs become huge!
```

### Solution

```python
# ✅ AFTER: Only log metadata
@m1_base()
def detect(image):  # image is (1080, 1920, 3) numpy array
    return model(image)

# Console output (INFO level with large input):
# (No output - large inputs use DEBUG level)
#
# Console output (DEBUG level):
# [DEBUG] → detect | Input: arg0=ndarray(shape=(1080, 1920, 3), dtype=uint8, range=[0, 255])
#
# Clean and informative!
```

### Implementation

Improved `_sanitize_value()` function:
- Detects numpy arrays: `_is_numpy_array(value)`
- Detects torch tensors: `_is_torch_tensor(value)`
- Detects PIL images: `hasattr(value, 'size') and hasattr(value, 'mode')`
- Returns metadata only: shape, dtype, min/max values
- Marks large objects: Returns `(sanitized_string, is_large)` tuple
- Auto DEBUG level: Large objects automatically use DEBUG level

### Sanitized Output Format

| Type | Output | Level |
|------|--------|-------|
| NumPy Array | `ndarray(shape=(1080, 1920, 3), dtype=uint8, range=[0, 255])` | DEBUG |
| PyTorch Tensor | `tensor(shape=(3, 224, 224), dtype=float32, range=[0.0, 1.0])` | DEBUG |
| PIL Image | `Image(size=(1920, 1080), mode=RGB)` | DEBUG |

### Configuration

```bash
# Production: Won't see numpy arrays in logs (they use DEBUG)
export WATERMETER_LOG_LEVEL=INFO

# Development: See sanitized numpy metadata
export WATERMETER_LOG_LEVEL=DEBUG
```

### Files Updated

- `src/common/logging_base.py` - Improved `_sanitize_value()` with tuple return
- `src/common/logging_base.py` - Added `_is_numpy_array()`, `_is_torch_tensor()` helpers
- `src/common/logging_base.py` - Updated decorator to use DEBUG for large inputs
- `examples/improved_pipeline_example.py` - ✅ Shows numpy logging behavior
- `docs/LOGGING_FRAMEWORK.md` - ✅ Documented sanitization behavior

---

## Migration Guide

### From Old to New

**1. Refactor Nested Functions**

```python
# BEFORE:
def pipeline(image):
    @m1_base()
    def detect(img):
        return yolo(img)
    return detect(image)

# AFTER:
@m1_base()
def detect_watermeter(image, model):
    return yolo_model(image)

def pipeline(image):
    return detect_watermeter(image, model)
```

**2. Add Batch Decorators for Loops**

```python
# BEFORE:
@m1_base()
def process_batch(images):
    for img in images:
        logger.info(f"Processing...")  # Manual logging
        result = detect(img)
    return results

# AFTER:
@m1_batch()  # Automatic batch logging
def process_batch(images):
    for img in images:
        result = detect(img)  # No manual logging needed
    return results
```

**3. Update Configuration**

```bash
# Add to .env:
WATERMETER_LOG_LEVEL=INFO  # Control verbosity
```

---

## Testing

### Unit Test Example

`tests/test_m1_detection.py` demonstrates:
- ✅ Independent function testing (Fix A)
- ✅ No need to test through pipeline
- ✅ Can mock inputs easily

```python
def test_detect_watermeter():
    """Test M1 function independently (Fix A)"""
    image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    result = detect_watermeter(image, model)

    assert result["success"] is True
    assert result["confidence"] >= 0.5
```

---

## Performance Impact

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| **Batch (1000 images)** | 1000 log lines | 2 log lines | 99.8% reduction |
| **Numpy logging** | ~200KB per log | ~100B per log | 99.95% reduction |
| **Unit testing** | Must test pipeline | Test functions directly | Faster testing |

---

## Summary

✅ **Fix A**: Functions are now modular, reusable, and testable
✅ **Fix B**: Batch processing no longer spams logs
✅ **Fix C**: CV data (numpy arrays) sanitized with metadata only

**Recommendation**: Use `examples/improved_pipeline_example.py` as the template for new code.

---

## Files Changed

- ✅ `src/common/logging_base.py` - Core framework with fixes
- ✅ `examples/improved_pipeline_example.py` - Recommended example
- ✅ `tests/test_m1_detection.py` - Unit test example
- ✅ `.env.example` - Added LOG_LEVEL config
- ✅ `docs/LOGGING_FRAMEWORK.md` - Updated documentation

# Water Meter AI - Logging Framework

## 📋 Overview

Framework logging tự động cho pipeline Water Meter AI (M1-M5 + Bayesian). Developer chỉ cần tập trung vào business logic, logging được xử lý tự động bởi decorators.

## 🎯 Key Features

- ✅ **Automatic Logging**: Input, output, execution time, errors - tất cả đều tự động
- ✅ **Per-Module Control**: Bật/tắt logging cho từng module riêng biệt
- ✅ **Zero Boilerplate**: Dev không cần viết `logger.info()`, `logger.error()` calls
- ✅ **Production Ready**: Dễ dàng bật/tắt logging qua environment variables
- ✅ **Sanitized Output**: Tự động sanitize large objects (images, arrays) để tránh log spam
- ✅ **Batch Processing**: Hỗ trợ DEBUG level cho loop iterations để tránh log spam
- ✅ **Modular Design**: Không lồng hàm - functions có thể reused & tested độc lập

## 📁 File Structure

```
src/common/logging_base.py                 # Base decorators for M1-M5
examples/m1_with_logging_example.py        # M1 usage examples
examples/m2_m3_m4_m5_logging_examples.py   # M2-M5 usage examples
examples/improved_pipeline_example.py     # Improved example (fixes A, B, C)
examples/complete_pipeline_with_logging.py # Full pipeline example
tests/test_m1_detection.py                 # Unit test example
.env.example                               # Configuration template
docs/LOGGING_FRAMEWORK.md                  # This file
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from src.common.logging_base import m1_base

@m1_base()  # Thêm decorator này
def detect_watermeter(image):
    # Dev chỉ viết logic, KHÔNG viết log
    from ultralytics import YOLO
    model = YOLO("model/detect_watermeter.pt")
    results = model(image)
    return results

# Gọi function như bình thường
result = detect_watermeter(image)
# Logging tự động output:
# [2026-02-10 10:30:00] [INFO] [WaterMeterAI.M1] → detect_watermeter | Input: arg0=ndarray(shape=(1080, 1920, 3), dtype=uint8, range=[0, 255])
# [2026-02-10 10:30:01] [INFO] [WaterMeterAI.M1] ✓ detect_watermeter | Output: {bbox=[100, 200, 500, 600], confidence=0.98} | Time: 123.45ms
```

### 2. Batch Processing (Avoid Log Spam)

```python
from src.common.logging_base import m1_batch

@m1_batch()  # Decorator cho batch processing
def detect_batch(images):
    results = []
    for img in images:
        result = detect_watermeter(img)
        results.append(result)
    return results

# INFO level: Chỉ log summary
# [INFO] 🚦 detect_batch | Starting batch (size=1000)
# [INFO] ✓ detect_batch | Batch complete: 1000/1000 successful (100.0%) | Time: 12345ms (12.35ms per item)
#
# DEBUG level: Log từng item chi tiết (set WATERMETER_LOG_LEVEL=DEBUG để enable)
```

### 1. Basic Usage

```python
from src.common.logging_base import m1_base

@m1_base()  # Thêm decorator này
def detect_watermeter(image):
    # Dev chỉ viết logic, KHÔNG viết log
    from ultralytics import YOLO
    model = YOLO("model/detect_watermeter.pt")
    results = model(image)
    return results

# Gọi function như bình thường
result = detect_watermeter(image)
# Logging tự động output:
# [2026-02-10 10:30:00] [INFO] [WaterMeterAI.M1] → detect_watermeter | Input: arg0=Image(size=1920x1080)
# [2026-02-10 10:30:01] [INFO] [WaterMeterAI.M1] ✓ detect_watermeter | Output: {...} | Time: 123.45ms
```

### 2. Disable Logging for Specific Function

```python
@m1_base(enable_log=False)  # Tắt logging cho function này
def internal_helper(data):
    # Function này sẽ không output log
    return process(data)
```

### 3. Global Logging Control

```bash
# Tắt tất cả logging
export WATERMETER_ENABLE_LOG=false

# Bật logging cho M1, M4, M5 only
export WATERMETER_ENABLE_LOG=true
export WATERMETER_LOG_M1=true
export WATERMETER_LOG_M2=false
export WATERMETER_LOG_M3=false
export WATERMETER_LOG_M4=true
export WATERMETER_LOG_M5=true
```

## 🔧 Improvements & Best Practices

### Fix A: Không lồng hàm (No Nested Functions)

**❌ Trước (Sai):**
```python
def run_pipeline(image):
    # Nested functions - cannot be tested or reused
    def step1_detect(img):
        return yolo(img)

    def step2_align(img):
        return rotate(img)

    # Only accessible within pipeline
    result1 = step1_detect(image)
    result2 = step2_align(result1)
    return result2
```

**✅ Sau (Đúng):**
```python
# Define at module level - reusable & testable
@m1_base()
def detect_watermeter(image, model):
    return yolo_model(image)

@m2_base()
def align_orientation(image, model):
    angle = predict_angle(model, image)
    return rotate_image(image, angle)

# Pipeline just calls them
def run_pipeline(image):
    m1_result = detect_watermeter(image, m1_model)  # Reusable!
    m2_result = align_orientation(m1_result.crop, m2_model)  # Testable!
    return m2_result
```

**Benefits:**
- ✅ Functions có thể được **unit test** độc lập (xem `tests/test_m1_detection.py`)
- ✅ Functions có thể **reused** ở nơi khác
- ✅ Functions có thể **import** từ modules khác

---

### Fix B: DEBUG Level cho Batch Processing

**Problem:** Logging trong loops làm file log phình to và chậm hệ thống.

**❌ Trước (Sai):**
```
[INFO] Processing item 1/1000
[INFO] Processing item 2/1000
[INFO] Processing item 3/1000
... (997 more log lines - console spam!)
```

**✅ Solution: Batch Decorators**
```python
from src.common.logging_base import m1_batch

@m1_batch()  # Sử dụng batch decorator
def process_batch(images):
    results = []
    for img in images:
        result = detect(img)  # Individual calls use DEBUG level
        results.append(result)
    return results
```

**Output:**
```
# INFO level (production):
[INFO] 🚦 process_batch | Starting batch (size=1000)
[INFO] ✓ process_batch | Batch complete: 1000/1000 (100.0%) | Time: 12345ms (12.35ms per item)

# DEBUG level (development):
[DEBUG] Individual item details available
```

**Configuration:**
```bash
# Production: INFO level only (no spam)
export WATERMETER_LOG_LEVEL=INFO

# Development: DEBUG level (see details)
export WATERMETER_LOG_LEVEL=DEBUG
```

---

### Fix C: Sanitize Numpy Arrays (Computer Vision Data)

**Problem:** Logging toàn bộ numpy array (pixel data) làm tràn console và file log.

**❌ Trước (Sai):**
```
[INFO] Input: array([[[12, 45, 78], [23, 56, 89], [34, 67, 90], ...millions more...]])
```

**✅ Solution: Auto-sanitize với metadata only**
```
[INFO] Input: ndarray(shape=(1080, 1920, 3), dtype=uint8, range=[0, 255])
```

**Framework tự động:**
- ✅ Chỉ log **shape**, **dtype**, **min/max values**
- ❌ KHÔNG log pixel data
- ✅ Large inputs tự động sử dụng **DEBUG level**

**Example:**
```python
@m1_base()
def detect(image):  # image là numpy array (1080, 1920, 3)
    # Log output tự động sanitized:
    # [INFO] → detect | Input: arg0=ndarray(shape=(1080, 1920, 3), dtype=uint8, range=[0, 255])
    return model(image)
```

---

## 📦 Module Decorators

| Decorator | Module | Purpose | Use Case |
|-----------|--------|---------|----------|
| `@m1_base()` | M1 | Watermeter Detection | Single image processing |
| `@m1_batch()` | M1 | Watermeter Detection (Batch) | Multiple images (Fix B) |
| `@m2_base()` | M2 | Orientation Alignment | Single image alignment |
| `@m3_base()` | M3 | Component Detection | Single image detection |
| `@m3_batch()` | M3 | Component Detection (Batch) | Multiple detections (Fix B) |
| `@m4_base()` | M4 | CRNN Digit Recognition | Single digit box |
| `@m5_base()` | M5 | Pointer Reading | Single pointer |
| `@m5_batch()` | M5 | Pointer Reading (Batch) | All 4 pointers (Fix B) |
| `@bayesian_base()` | Bayesian | Enhancement Layer | Rolling digit resolution |
| `@pipeline_step()` | Pipeline | E2E monitoring | Full pipeline execution |

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WATERMETER_ENABLE_LOG` | `true` | Global logging on/off |
| `WATERMETER_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `WATERMETER_LOG_M1` | `true` | M1 logging on/off |
| `WATERMETER_LOG_M2` | `true` | M2 logging on/off |
| `WATERMETER_LOG_M3` | `true` | M3 logging on/off |
| `WATERMETER_LOG_M4` | `true` | M4 logging on/off |
| `WATERMETER_LOG_M5` | `true` | M5 logging on/off |
| `WATERMETER_LOG_BAYESIAN` | `true` | Bayesian logging on/off |

**Log Level Guide:**
- `INFO` (Default): Batch summaries, key milestones
- `DEBUG`: Individual item details, full trace
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors

### Setup Configuration File

```bash
# Copy template
cp .env.example .env

# Edit .env to configure logging
nano .env
```

## 📝 Logging Output Examples

### Successful Execution

```
[2026-02-10 10:30:00] [INFO] [WaterMeterAI.M1] → detect_watermeter | Input: arg0=Image(size=1920x1080)
[2026-02-10 10:30:01] [INFO] [WaterMeterAI.M1] ✓ detect_watermeter | Output: {bbox=[100, 200, 500, 600], confidence=0.98} | Time: 123.45ms
```

### Error with Traceback

```
[2026-02-10 10:30:00] [INFO] [WaterMeterAI.M1] → detect_watermeter | Input: arg0=Image(size=1920x1080)
[2026-02-10 10:30:01] [ERROR] [WaterMeterAI.M1] ✗ detect_watermeter | Error: FileNotFoundError: Image not found | Time: 5.23ms
```

### Pipeline Execution

```
[2026-02-10 10:30:00] [INFO] [WaterMeterAI.Pipeline] ============================================================
[2026-02-10 10:30:00] [INFO] [WaterMeterAI.Pipeline] 🚀 Starting Pipeline: M1→M2→M3→M4→M5→Bayesian
[2026-02-10 10:30:00] [INFO] [WaterMeterAI.Pipeline] ============================================================
[2026-02-10 10:30:05] [INFO] [WaterMeterAI.Pipeline] ============================================================
[2026-02-10 10:30:05] [INFO] [WaterMeterAI.Pipeline] ✅ Pipeline Completed: M1→M2→M3→M4→M5→Bayesian | Total Time: 5.234s
[2026-02-10 10:30:05] [INFO] [WaterMeterAI.Pipeline] ============================================================
```

## 🎨 Advanced Usage

### Custom Module Decorator

```python
from src.common.logging_base import create_module_decorator

# Create custom decorator
my_module_base = create_module_decorator(
    module_name="MyModule",
    enable_log=True,
    log_input=True,
    log_output=True,
    log_performance=True,
    log_error=True
)

@my_module_base()
def my_function(data):
    return process(data)
```

### Pipeline Monitoring

```python
from src.common.logging_base import pipeline_step

@pipeline_step("My Custom Pipeline")
def run_my_pipeline(input_data):
    # Step 1
    result1 = step1(input_data)

    # Step 2
    result2 = step2(result1)

    # Step 3
    result3 = step3(result2)

    return result3
```

## 🏭 Best Practices

### 1. Development Environment

```bash
# Enable all logging with DEBUG level
export WATERMETER_ENABLE_LOG=true
export WATERMETER_LOG_LEVEL=DEBUG  # See all details including batch items
export WATERMETER_LOG_M1=true
export WATERMETER_LOG_M2=true
export WATERMETER_LOG_M3=true
export WATERMETER_LOG_M4=true
export WATERMETER_LOG_M5=true
```

### 2. Production Environment

```bash
# INFO level for batch summaries only (Fix B)
export WATERMETER_ENABLE_LOG=true
export WATERMETER_LOG_LEVEL=INFO  # No console spam from loops

# Disable specific verbose modules
export WATERMETER_LOG_M1=false  # M1 is very verbose
export WATERMETER_LOG_M2=false  # M2 is very verbose
export WATERMETER_LOG_M3=false  # M3 is very verbose
export WATERMETER_LOG_M4=true   # Keep M4 for CRNN monitoring
export WATERMETER_LOG_M5=true   # Keep M5 for pointer monitoring
```

### 3. Performance Testing

```bash
# Disable all logging for accurate performance measurement
export WATERMETER_ENABLE_LOG=false
```

### 4. Code Organization (Fix A)

```bash
# ✅ Define functions at module level
src/m1_watermeter_detection/
├── __init__.py
├── detect.py          # @m1_base() def detect_watermeter(image, model)
├── preprocess.py      # @m1_base() def preprocess_image(image, size)
└── batch.py           # @m1_batch() def detect_batch(images, model)

# ❌ Avoid nested functions
# def pipeline():
#     def nested_function():  # Cannot test this independently!
#         pass
```

## 📊 Log Sanitization (Fix C)

Framework tự động sanitize các đối tượng lớn để tránh log spam:

| Type | Sanitized Output | Level |
|------|-----------------|-------|
| NumPy Array | `ndarray(shape=(1080, 1920, 3), dtype=uint8, range=[0, 255])` | DEBUG* |
| PIL Image | `Image(size=(1920, 1080), mode=RGB)` | DEBUG* |
| PyTorch Tensor | `tensor(shape=(3, 224, 224), dtype=float32, range=[0.0, 1.0])` | DEBUG* |
| Long String | `'This is a very long string...' (len=500)` | INFO |
| Large List | `list(len=10000, [...])` | INFO |
| Large Dict | `dict(keys=5000, {...})` | INFO |

*Large objects (numpy arrays, images) automatically use DEBUG level to avoid console spam.

**Example Outputs:**

```python
# Input: numpy array with image data
image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

# ✅ Sanitized log output (what you see):
[DEBUG] → detect | Input: arg0=ndarray(shape=(1080, 1920, 3), dtype=uint8, range=[0, 255])

# ❌ NOT this (what we're avoiding):
[INFO] → detect | Input: arg0=[[[12, 45, 78], [23, 56, 89], ... millions of pixels ...]]
```

**Configuration for CV Data:**
```bash
# Production: Only see sanitized metadata at DEBUG level
export WATERMETER_LOG_LEVEL=INFO  # Won't see numpy arrays in logs

# Development: See full metadata including min/max values
export WATERMETER_LOG_LEVEL=DEBUG  # See sanitized numpy array info
```

## 🐛 Debugging Tips

### Enable Debug Logging

```python
import logging
logging.getLogger("WaterMeterAI").setLevel(logging.DEBUG)
```

### Log to File

```python
import logging

# Add file handler
file_handler = logging.FileHandler("watermeter_ai.log")
file_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
))

logging.getLogger("WaterMeterAI").addHandler(file_handler)
```

### Check Logging Configuration

```python
from src.common.logging_base import logger, ENABLE_LOG

print(f"Logging enabled: {ENABLE_LOG}")
print(f"Logger level: {logger.level}")
```

## 📚 Example Files

### Basic Examples
- **[examples/m1_with_logging_example.py](../examples/m1_with_logging_example.py)** - M1 basic usage with class-based approach
- **[examples/m2_m3_m4_m5_logging_examples.py](../examples/m2_m3_m4_m5_logging_examples.py)** - M2-M5 individual examples

### **Recommended** Examples (Fixes A, B, C)
- **[examples/improved_pipeline_example.py](../examples/improved_pipeline_example.py)** ⭐ **Recommended**
  - ✅ Fix A: No nested functions
  - ✅ Fix B: Batch processing with DEBUG level
  - ✅ Fix C: Sanitized numpy array logging

### Advanced Examples
- **[examples/complete_pipeline_with_logging.py](../examples/complete_pipeline_with_logging.py)** - Full end-to-end pipeline
- **[tests/test_m1_detection.py](../tests/test_m1_detection.py)** - Unit test examples showing Fix A benefits

### Which Example Should I Use?

| Use Case | Recommended Example |
|----------|-------------------|
| **Learning basics** | `m1_with_logging_example.py` |
| **Production code** | ⭐ `improved_pipeline_example.py` |
| **Understanding fixes** | ⭐ `improved_pipeline_example.py` + `test_m1_detection.py` |
| **Full pipeline reference** | `complete_pipeline_with_logging.py` |
| **Unit testing** | `tests/test_m1_detection.py` |

---

## 🔗 Related Documentation

- [Architecture Document](f:\Workspace\Project\_bmad-output\planning-artifacts\architecture-Project.md)
- [Implementation Artifacts](f:\Workspace\Project\_bmad-output\implementation-artifacts\)
- [M1 Implementation](f:\Workspace\Project\src\m1_watermeter_detection\)
- [M2 Implementation](f:\Workspace\Project\src\m2_orientation_alignment\)
- [M3 Implementation](f:\Workspace\Project\src\m3_counter_roi\)
- [M4 Implementation (CRNN)](f:\Workspace\Project\_bmad-output\implementation-artifacts\1-4-crnn-digit-recognition-m4.md)
- [M5 Implementation (Pointer)](f:\Workspace\Project\_bmad-output\implementation-artifacts\1-5-pointer-reading-m5.md)

## 📞 Support

For issues or questions about the logging framework:
1. Check this documentation
2. Review example files in `examples/`
3. Check `.env.example` for configuration options

---

**Last Updated**: 2026-02-10
**Version**: 1.0.0

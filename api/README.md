# Meter Reading API - FastAPI Server

AI Pipeline for reading water meter values (M1 → M2 → M3 → M3.5 → M4)

## Quick Start

### 1. Install Dependencies
```bash
cd f:\Workspace\Project\api
pip install -r requirements.txt
```

### 2. Run Server
```bash
python main.py
```

Server will start at: **http://localhost:8000**

### 3. Test API

#### Interactive Swagger UI
Open browser: **http://localhost:8000/docs**

#### cURL Example
```bash
# Convert image to base64 first
IMAGE_BASE64=$(base64 -w 0 path/to/your/image.jpg)

# Send prediction request
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_BASE64\"}"
```

#### Python Example
```python
import base64
import requests

# Read and encode image
with open("meter_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/predict",
    json={"image": image_base64}
)

print(response.json())
# Output: {
#   "prediction": "1234",
#   "success": true,
#   "pipeline_data": {
#     "m1_bbox": [100, 200, 600, 800],
#     "m2_angle": 45.5,
#     "m3_bbox": [150, 250, 500, 400]
#   },
#   "error": null,
#   "timestamp": "2025-01-16T10:30:00"
# }
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | Health check |
| `/predict` | POST | Main prediction endpoint |

## Pipeline Architecture

```
Base64 Input
    ↓
[M1] YOLO Detection → Find meter in image
    ↓
[M2] Angle Detection → Rotate meter upright
    ↓
[M3] ROI Detection → Find digit region
    ↓
[M3.5] Digit Extraction → Extract black digits
    ↓
[M4] CRNN OCR → Read final number
    ↓
Output: "1234"
```

## Response Format

**Success:**
```json
{
  "prediction": "1234",
  "success": true,
  "pipeline_data": {
    "m1_bbox": [100, 200, 600, 800],
    "m2_angle": 45.5,
    "m3_bbox": [150, 250, 500, 400]
  },
  "error": null,
  "timestamp": "2025-01-16T10:30:00"
}
```

**Error:**
```json
{
  "prediction": null,
  "success": false,
  "pipeline_data": null,
  "error": "M1: No meter detected",
  "timestamp": "2025-01-16T10:30:00"
}
```

## Configuration

Edit `main.py` to adjust:

```python
# Model paths
MODEL_PATHS = {
    'm1': r"path/to/M1_DetectWatermeter.pt",
    'm2': r"path/to/m2_angle_model_epoch15_FIXED_COS_SIN.pth",
    'm3': r"path/to/M3_Roi_Boundingbox.pt",
    'm4': r"path/to/M4_OCR.pth",
}

# Detection thresholds
m1_confidence=0.15,  # Lower = more detections
m3_confidence=0.10,  # Lower = more detections
beam_width=10,       # Higher = more accurate but slower
```

## Performance

- **Startup time**: ~10-15 seconds (model loading)
- **Inference time**: ~2-4 seconds per image
- **Capacity**: ~10,000 requests/day on single GPU

## Troubleshooting

**Models not found?**
- Check `MODEL_PATHS` in `main.py`
- Verify all 4 model files exist

**CUDA out of memory?**
- Reduce batch size or use CPU mode
- Close other GPU applications

**Import errors?**
- Run `pip install -r requirements.txt`
- Ensure Python 3.8+

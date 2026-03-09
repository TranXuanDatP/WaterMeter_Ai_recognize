# API Testing UI

A Streamlit-based web interface for testing the Water Meter Reading API.

## Installation

```bash
# Install dependencies
pip install -r src/api_testing_ui/requirements.txt
```

## Running

```bash
# From project root
streamlit run src/api_testing_ui/app.py
```

The UI will open at http://localhost:8501

## Configuration

Edit `src/api_testing_ui/config.py` to change:
- API base URL (default: http://localhost:8000)
- API timeout (default: 30 seconds)
- Max image size (default: 10MB)

## Usage

1. Open the UI in your browser
2. Upload a water meter image (JPEG/PNG)
3. Optionally enter meter ID and GPS coordinates
4. Click "Test API" to submit the request
5. View results with confidence scores and digit breakdown

## Features

- Image upload with preview
- Color-coded confidence (GREEN/YELLOW/RED)
- Digit and pointer breakdown
- Debug panel with raw JSON response
- Session history (last 10 tests)
- Error handling with helpful messages

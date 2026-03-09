"""
Configuration for API Testing UI
"""

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30  # seconds
API_ENDPOINT = "/api/v1/read-meter"

# File Upload Configuration
MAX_IMAGE_SIZE_MB = 10
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

# Session History Configuration
MAX_HISTORY_SIZE = 10

# UI Configuration
PAGE_TITLE = "Water Meter AI - Testing UI"
PAGE_LAYOUT = "wide"

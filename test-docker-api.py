#!/usr/bin/env python3
"""
Test script for Water Meter AI Docker API
Run this after starting the Docker container
"""

import requests
import base64
import json
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        print(f"✓ Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_predict_with_base64(image_path: str):
    """Test predict endpoint with base64 encoded image"""
    print(f"\nTesting predict endpoint with {image_path}...")

    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"image": image_data},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        print(f"✓ Prediction successful!")
        print(f"  Reading: {result.get('prediction', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text}")
        return False

def test_docs():
    """Test API documentation endpoint"""
    print("\nTesting API documentation...")
    try:
        response = requests.get(f"{API_URL}/docs")
        response.raise_for_status()
        print(f"✓ API docs available at: {API_URL}/docs")
        return True
    except Exception as e:
        print(f"✗ API docs check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Water Meter AI - Docker API Test Suite")
    print("=" * 60)

    # Test 1: Health check
    if not test_health():
        print("\n✗ API is not responding. Is Docker container running?")
        print("  Start with: docker-build.bat run")
        sys.exit(1)

    # Test 2: API docs
    test_docs()

    # Test 3: Predict endpoint (if test image available)
    test_images = [
        "data/data_4digit2/meter4_00000_00385501ab4d419fa7b0bdf0d9f8451f.jpg",
        "data/test_image.jpg",
    ]

    image_found = False
    for img_path in test_images:
        if Path(img_path).exists():
            if test_predict_with_base64(img_path):
                image_found = True
            break

    if not image_found:
        print("\n⚠ No test images found. Skipping prediction test.")
        print("  Place a test image in data/ directory to test prediction.")

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

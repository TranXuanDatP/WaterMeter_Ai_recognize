#!/usr/bin/env python3
"""
Quick test script for Meter Reading API

Tests the API with a sample image (if available)
"""
import base64
import requests
import json
from pathlib import Path


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def test_api(base_url: str = "http://localhost:8000"):
    """Test API endpoints"""
    print("="*70)
    print("Testing Meter Reading API")
    print("="*70)
    print()

    # Test 1: Root endpoint
    print("[Test 1] Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()

    # Test 2: Health endpoint
    print("[Test 2] Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()

    # Test 3: Predict endpoint (if sample image exists)
    print("[Test 3] Testing predict endpoint...")

    # Look for sample image
    sample_paths = [
        r"F:\Workspace\Project\data\data_4digit2\meter4_00000_00385501ab4d419fa7b0bdf0d9f8451f.jpg",
        r"F:\Workspace\Project\test_image.jpg",
    ]

    sample_image = None
    for path in sample_paths:
        if Path(path).exists():
            sample_image = path
            break

    if sample_image:
        print(f"Found sample image: {sample_image}")
        try:
            # Encode image
            image_base64 = encode_image_to_base64(sample_image)
            print(f"Image size: {len(image_base64)} chars")

            # Send request
            response = requests.post(
                f"{base_url}/predict",
                json={"image": image_base64},
                timeout=30
            )

            print(f"✓ Status: {response.status_code}")
            print(f"✓ Response:")
            print(json.dumps(response.json(), indent=2))

            # Check if prediction successful
            if response.json()['success']:
                print(f"\n🎉 Prediction: {response.json()['prediction']}")
            else:
                print(f"\n⚠️ Error: {response.json()['error']}")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ No sample image found for testing")
        print("To test prediction manually:")
        print("  1. Prepare an image file")
        print("  2. Convert to base64: base64 -w 0 image.jpg > image.b64")
        print("  3. Send POST request to /predict with JSON: {'image': '...'}")

    print()
    print("="*70)
    print("Test completed!")
    print("="*70)
    print("\nFor interactive testing, open: http://localhost:8000/docs")


if __name__ == "__main__":
    import sys

    # Check if server is running
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"Target API: {base_url}")
    print("Make sure the API server is running first!")
    print("Start server with: python main.py")
    print()

    input("Press Enter to start testing...")

    test_api(base_url)

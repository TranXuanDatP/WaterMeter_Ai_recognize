#!/usr/bin/env python3
"""
Test Meter Reading API with a real sample image
"""
import base64
import requests
import json
import time
import sys
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Convert image to base64"""
    print(f"[*] Reading image: {image_path}")
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    print(f"    ✓ Image size: {len(image_data)} characters")
    return image_data


def test_prediction(api_url: str, image_path: str):
    """Test prediction endpoint"""
    print("\n" + "="*70)
    print("METER READING API - LIVE TEST")
    print("="*70)
    print(f"API URL: {api_url}")
    print(f"Image: {image_path}")
    print("="*70)
    print()

    # Check if image exists
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        print("\nTrying alternative images...")

        # Try to find any sample image
        data_dir = Path(r"F:\Workspace\Project\data\data_4digit2")
        if data_dir.exists():
            images = list(data_dir.glob("*.jpg"))[:5]
            if images:
                image_path = str(images[0])
                print(f"✓ Found: {image_path}")
            else:
                print("✗ No images found")
                return
        else:
            print("✗ Data directory not found")
            return

    # Step 1: Check server health
    print("[Step 1/3] Checking server health...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"  ✓ Server Status: {health['status']}")
            print(f"  ✓ Pipeline Loaded: {health['pipeline_loaded']}")
            print(f"  ✓ Device: {health['device']}")
        else:
            print(f"  ✗ Server returned: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to server at {api_url}")
        print(f"\n  Please start the server first:")
        print(f"    cd api")
        print(f"    python main.py")
        return
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return

    # Step 2: Prepare image
    print("\n[Step 2/3] Preparing image...")
    try:
        image_base64 = encode_image(image_path)
        print(f"  ✓ Base64 encoded")
    except Exception as e:
        print(f"  ✗ Error encoding image: {e}")
        return

    # Step 3: Send prediction request
    print("\n[Step 3/3] Sending prediction request...")
    print(f"  Endpoint: POST {api_url}/predict")

    start_time = time.time()

    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"image": image_base64},
            timeout=30
        )

        elapsed_time = time.time() - start_time

        print(f"  ✓ Response received in {elapsed_time:.2f}s")
        print(f"  ✓ Status Code: {response.status_code}")

        # Parse response
        result = response.json()

        # Display results
        print("\n" + "="*70)
        print("RESULT")
        print("="*70)

        if result['success']:
            print(f"\n  🎉 PREDICTION SUCCESS!")
            print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  📊 Meter Reading: {result['prediction']}")
            print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"\n  Pipeline Details:")

            if result['pipeline_data']:
                data = result['pipeline_data']
                print(f"    ├─ M1 (Detection): {data['m1_bbox']}")
                print(f"    ├─ M2 (Angle):      {data['m2_angle']:.2f}°")
                print(f"    └─ M3 (ROI):        {data['m3_bbox']}")
        else:
            print(f"\n  ⚠️ PREDICTION FAILED")
            print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  Error: {result['error']}")

        print(f"\n  Timestamp: {result['timestamp']}")
        print(f"  Processing Time: {elapsed_time:.2f}s")

        # Additional info
        print("\n" + "="*70)
        print("PERFORMANCE STATS")
        print("="*70)
        print(f"  Total Time: {elapsed_time:.2f}s")
        print(f"  Throughput: {3600/elapsed_time:.0f} requests/hour (single-threaded)")
        print(f"  Daily Capacity: {86400/elapsed_time:.0f} requests/day (theoretical)")

    except requests.exceptions.Timeout:
        print(f"  ✗ Request timeout (>30s)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)


if __name__ == "__main__":
    # Configuration
    API_URL = "http://localhost:8000"
    SAMPLE_IMAGE = r"F:\Workspace\Project\data\data_4digit2\meter4_00000_00385501ab4d419fa7b0bdf0d9f8451f.jpg"

    # Allow command line overrides
    if len(sys.argv) > 1:
        SAMPLE_IMAGE = sys.argv[1]
    if len(sys.argv) > 2:
        API_URL = sys.argv[2]

    # Run test
    test_prediction(API_URL, SAMPLE_IMAGE)

#!/usr/bin/env python3
"""
Test Meter Reading API with direct Base64 input
"""
import requests
import json
import sys


def test_with_base64(api_url: str, base64_string: str):
    """
    Send prediction request with base64 string

    Args:
        api_url: API endpoint URL
        base64_string: Base64 encoded image (with or without data URL prefix)
    """
    print("\n" + "="*70)
    print("METER READING API - DIRECT BASE64 TEST")
    print("="*70)

    # Clean base64 string (remove data URL prefix if present)
    cleaned_base64 = base64_string.strip()
    if ',' in cleaned_base64:
        # Remove data URL prefix like "data:image/jpeg;base64,"
        cleaned_base64 = cleaned_base64.split(',', 1)[1]

    # Remove newlines and extra spaces
    cleaned_base64 = cleaned_base64.replace('\n', '').replace('\r', '').replace(' ', '')

    print(f"\n[*] API Endpoint: {api_url}/predict")
    print(f"[*] Base64 Length: {len(cleaned_base64):,} characters")
    print(f"[*] Estimated Size: {len(cleaned_base64) * 3 / 4 / 1024:.2f} KB")

    # Validate base64
    print(f"\n[*] Validating base64...")
    try:
        import base64
        decoded = base64.b64decode(cleaned_base64)
        print(f"  ✓ Valid base64")
        print(f"  ✓ Decoded size: {len(decoded):,} bytes")

        # Check if it's an image
        import io
        from PIL import Image
        try:
            img = Image.open(io.BytesIO(decoded))
            print(f"  ✓ Image Format: {img.format}")
            print(f"  ✓ Image Size: {img.size[0]}x{img.size[1]} pixels")
            print(f"  ✓ Image Mode: {img.mode}")
        except:
            print(f"  ⚠️  Warning: Cannot verify image format")
    except Exception as e:
        print(f"  ✗ Invalid base64: {e}")
        return

    # Send request
    print(f"\n[*] Sending prediction request...")
    try:
        import time
        start_time = time.time()

        response = requests.post(
            f"{api_url}/predict",
            json={"image": cleaned_base64},
            timeout=30
        )

        elapsed_time = time.time() - start_time

        print(f"  ✓ Response received ({elapsed_time:.2f}s)")
        print(f"  ✓ Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # Display result
            print("\n" + "="*70)
            print("RESULT")
            print("="*70)

            if result['success']:
                print(f"\n  🎉 PREDICTION SUCCESS!")
                print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"  📊 Meter Reading: {result['prediction']}")
                print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                if result['pipeline_data']:
                    data = result['pipeline_data']
                    print(f"\n  Pipeline Details:")
                    print(f"    ├─ M1 (Detection): {data['m1_bbox']}")
                    print(f"    ├─ M2 (Angle):      {data['m2_angle']:.2f}°")
                    print(f"    └─ M3 (ROI):        {data['m3_bbox']}")
            else:
                print(f"\n  ⚠️  PREDICTION FAILED")
                print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"  Error: {result['error']}")

            print(f"\n  Timestamp: {result['timestamp']}")
            print(f"  Processing Time: {elapsed_time:.2f}s")

        else:
            print(f"  ✗ HTTP Error {response.status_code}")
            print(f"  Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to server")
        print(f"\n  Make sure server is running:")
        print(f"    cd api")
        print(f"    python main.py")
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timeout (>30s)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)


def interactive_mode():
    """Interactive mode - paste base64"""
    print("="*70)
    print("METER READING API - BASE64 INPUT MODE")
    print("="*70)
    print("\nPaste your Base64 string below, then press Enter twice:")
    print("(Or type 'exit' to quit)")
    print("="*70)

    # Read multiline input
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == '' and len(lines) > 0:
                break
            if line.strip().lower() == 'exit':
                return
            lines.append(line)
        except EOFError:
            break

    base64_input = ''.join(lines)

    if not base64_input.strip():
        print("\n✗ No input provided")
        return

    # Test with the base64
    test_with_base64("http://localhost:8000", base64_input)


def file_mode(file_path: str):
    """Read base64 from file"""
    print(f"[*] Reading base64 from file: {file_path}")

    try:
        with open(file_path, 'r') as f:
            base64_content = f.read().strip()

        test_with_base64("http://localhost:8000", base64_content)
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
    except Exception as e:
        print(f"✗ Error reading file: {e}")


def main():
    if len(sys.argv) > 1:
        # File mode
        file_path = sys.argv[1]
        file_mode(file_path)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()

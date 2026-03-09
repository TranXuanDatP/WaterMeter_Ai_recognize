#!/usr/bin/env python3
"""
Interactive Test Script - Choose image to test
"""
import base64
import requests
import json
from pathlib import Path
import sys


def list_available_images(data_dir: str, limit: int = 10):
    """List available images in data directory"""
    data_path = Path(data_dir)

    if not data_path.exists():
        return []

    images = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    return images[:limit]


def display_image_menu(images):
    """Display images as menu"""
    print("\n" + "="*70)
    print("AVAILABLE IMAGES")
    print("="*70)

    for idx, img_path in enumerate(images, 1):
        # Try to get true value from CSV if available
        img_name = img_path.name
        print(f"  [{idx}] {img_name}")

    print(f"  [0] Exit")
    print("="*70)


def get_true_value_from_csv(image_name: str) -> str:
    """Get true value from labels CSV if available"""
    csv_path = Path(r"F:\Workspace\Project\data\images_4digit2.csv")

    if not csv_path.exists():
        return None

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        row = df[df['photo_name'] == image_name]
        if len(row) > 0:
            return str(row.iloc[0]['value'])
    except:
        pass

    return None


def test_single_image(api_url: str, image_path: Path):
    """Test prediction with single image"""
    print("\n" + "="*70)
    print(f"TESTING: {image_path.name}")
    print("="*70)

    # Get true value if available
    true_value = get_true_value_from_csv(image_path.name)
    if true_value:
        print(f"  ✓ True Value: {true_value}")

    # Encode image
    print(f"\n[*] Encoding image to base64...")
    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        print(f"  ✓ Image size: {len(image_base64):,} characters")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return

    # Send request
    print(f"\n[*] Sending prediction request to {api_url}/predict...")

    try:
        import time
        start_time = time.time()

        response = requests.post(
            f"{api_url}/predict",
            json={"image": image_base64},
            timeout=30
        )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()

            print(f"\n  ✓ Response received in {elapsed_time:.2f}s")

            # Display result
            print("\n" + "━"*70)
            if result['success']:
                prediction = result['prediction']
                print(f"  🎉 PREDICTION: {prediction}")

                if true_value:
                    if prediction == true_value:
                        print(f"  ✅ CORRECT! (True: {true_value})")
                    else:
                        print(f"  ❌ WRONG! (True: {true_value})")

                print("\n  Pipeline Details:")
                if result['pipeline_data']:
                    data = result['pipeline_data']
                    print(f"    M1 (Detection): {data['m1_bbox']}")
                    print(f"    M2 (Angle):      {data['m2_angle']:.2f}°")
                    print(f"    M3 (ROI):        {data['m3_bbox']}")
            else:
                print(f"  ⚠️  PREDICTION FAILED")
                print(f"  Error: {result['error']}")

            print("━"*70)
            print(f"  Processing Time: {elapsed_time:.2f}s")
        else:
            print(f"  ✗ HTTP Error: {response.status_code}")
            print(f"  Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to server")
        print(f"  Make sure server is running: python main.py")
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timeout (>30s)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def batch_test(api_url: str, images: list, max_tests: int = 5):
    """Test multiple images in batch"""
    print("\n" + "="*70)
    print(f"BATCH TESTING - {min(max_tests, len(images))} images")
    print("="*70)

    results = []

    for i, img_path in enumerate(images[:max_tests], 1):
        print(f"\n[{i}/{min(max_tests, len(images))}] {img_path.name}")

        # Encode and test
        try:
            with open(img_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()

            import time
            start = time.time()

            response = requests.post(
                f"{api_url}/predict",
                json={"image": image_base64},
                timeout=30
            )

            elapsed = time.time() - start
            result = response.json()

            true_value = get_true_value_from_csv(img_path.name)
            prediction = result.get('prediction') if result['success'] else None
            correct = (prediction == true_value) if (prediction and true_value) else None

            results.append({
                'image': img_path.name,
                'true': true_value,
                'predicted': prediction,
                'correct': correct,
                'success': result['success'],
                'time': elapsed,
                'error': result.get('error')
            })

            status = "✅" if correct else "❌" if correct is not None else "⚠️"
            print(f"  {status} Pred: {prediction} | True: {true_value} | Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'image': img_path.name,
                'error': str(e),
                'success': False
            })

    # Summary
    print("\n" + "="*70)
    print("BATCH TEST SUMMARY")
    print("="*70)

    successful = sum(1 for r in results if r['success'])
    correct_predictions = sum(1 for r in results if r.get('correct') == True)
    total_with_ground_truth = sum(1 for r in results if r.get('correct') is not None)
    avg_time = sum(r['time'] for r in results if 'time' in r) / len(results)

    print(f"  Total Tested: {len(results)}")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Accuracy: {correct_predictions}/{total_with_ground_truth} "
          f"({100*correct_predictions/total_with_ground_truth:.1f}%)" if total_with_ground_truth > 0 else "")
    print(f"  Avg Time: {avg_time:.2f}s")

    print("\nDetailed Results:")
    print(f"{'Image':<40} {'True':>8} {'Pred':>8} {'Status':>10} {'Time':>8}")
    print("─"*70)

    for r in results:
        img_short = r['image'][:37] + '...' if len(r['image']) > 40 else r['image']
        true = r.get('true', 'N/A')
        pred = r.get('predicted', 'N/A')
        status = '✅' if r.get('correct') == True else '❌' if r.get('correct') == False else '⚠️'
        time_str = f"{r.get('time', 0):.2f}s"

        print(f"{img_short:<40} {true:>8} {pred:>8} {status:>10} {time_str:>8}")


def main():
    API_URL = "http://localhost:8000"
    DATA_DIR = r"F:\Workspace\Project\data\data_4digit2"

    # Check server first
    print(f"[*] Checking server at {API_URL}...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            print(f"  ✓ Server is running")
        else:
            print(f"  ✗ Server returned status {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to server")
        print(f"\n  Please start the server first:")
        print(f"    cd api")
        print(f"    python main.py")
        return
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return

    # List available images
    print(f"\n[*] Scanning images from {DATA_DIR}...")
    images = list_available_images(DATA_DIR, limit=20)

    if not images:
        print(f"  ✗ No images found in {DATA_DIR}")
        print(f"\n  You can also provide custom image path:")
        print(f"    python interactive_test.py \"path\\to\\image.jpg\"")
        return

    print(f"  ✓ Found {len(images)} images")

    # If image provided as argument
    if len(sys.argv) > 1:
        custom_image = Path(sys.argv[1])
        if custom_image.exists():
            test_single_image(API_URL, custom_image)
            return

    # Interactive menu
    display_image_menu(images)

    while True:
        try:
            choice = input(f"\nSelect image [1-{len(images)}], 'b' for batch test, or '0' to exit: ").strip()

            if choice.lower() == '0' or choice.lower() == 'exit' or choice.lower() == 'q':
                print("👋 Goodbye!")
                break

            elif choice.lower() == 'b':
                # Batch test
                num = input(f"How many images to test? [1-{len(images)}] (default: 5): ").strip()
                try:
                    max_tests = int(num) if num else 5
                    max_tests = min(max_tests, len(images))
                except:
                    max_tests = 5

                batch_test(API_URL, images, max_tests)
                break

            elif choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(images):
                    test_single_image(API_URL, images[idx - 1])

                    # Ask if continue
                    cont = input("\nTest another image? [y/n]: ").strip().lower()
                    if cont != 'y':
                        break
                else:
                    print(f"  ✗ Invalid choice. Please enter 1-{len(images)}")
            else:
                print("  ✗ Invalid input")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    main()

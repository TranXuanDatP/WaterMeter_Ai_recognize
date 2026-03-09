#!/usr/bin/env python3
"""
Test API with pre-generated Base64 samples
"""
import requests
import json
import time
from pathlib import Path


def test_base64_samples(
    api_url: str,
    samples_file: str,
    max_tests: int = None
):
    """
    Test API with base64 samples from JSON file

    Args:
        api_url: API endpoint URL
        samples_file: Path to JSON file with base64 samples
        max_tests: Maximum number of samples to test (None = all)
    """
    print("="*70)
    print("API TEST - BASE64 SAMPLES")
    print("="*70)

    # Load samples
    print(f"\n[*] Loading samples from: {samples_file}")
    try:
        with open(samples_file, 'r') as f:
            samples = json.load(f)
        print(f"  ✓ Loaded {len(samples)} samples")
    except FileNotFoundError:
        print(f"  ✗ File not found: {samples_file}")
        print(f"\n  Run this first to generate samples:")
        print(f"    python generate_base64_samples.py")
        return
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return

    # Limit tests if specified
    if max_tests:
        samples = samples[:max_tests]
        print(f"  [*] Testing first {len(samples)} samples")

    # Test each sample
    results = []

    print(f"\n[*] Starting tests...")
    print("─"*70)

    for i, sample in enumerate(samples, 1):
        filename = sample['filename']
        true_value = sample.get('true_value')
        base64_str = sample['base64_raw']

        print(f"\n[{i}/{len(samples)}] {filename[:50]}...")

        if true_value:
            print(f"  True Value: {true_value}")

        try:
            start_time = time.time()

            # Send request
            response = requests.post(
                f"{api_url}/predict",
                json={"image": base64_str},
                timeout=30
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                prediction = result.get('prediction')
                success = result.get('success')

                # Check if correct
                correct = None
                if true_value and success:
                    correct = (prediction == true_value)

                # Display result
                status = "✅" if correct else "❌" if correct is not None else "⚠️"
                pred_str = prediction if success else "FAILED"

                print(f"  {status} Prediction: {pred_str} | Time: {elapsed_time:.2f}s")

                results.append({
                    'filename': filename,
                    'true_value': true_value,
                    'predicted': prediction,
                    'success': success,
                    'correct': correct,
                    'time': elapsed_time,
                    'error': result.get('error')
                })
            else:
                print(f"  ✗ HTTP {response.status_code}: {response.text[:100]}")
                results.append({
                    'filename': filename,
                    'true_value': true_value,
                    'success': False,
                    'error': f"HTTP {response.status_code}"
                })

        except requests.exceptions.Timeout:
            print(f"  ✗ Timeout (>30s)")
            results.append({
                'filename': filename,
                'true_value': true_value,
                'success': False,
                'error': 'Timeout'
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'filename': filename,
                'true_value': true_value,
                'success': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total = len(results)
    successful = sum(1 for r in results if r['success'])
    correct = sum(1 for r in results if r.get('correct') == True)
    with_ground_truth = sum(1 for r in results if r.get('true_value'))
    avg_time = sum(r['time'] for r in results if 'time' in r) / len(results) if results else 0

    print(f"\n  Total Tests:       {total}")
    print(f"  Successful:        {successful}/{total} ({100*successful/total:.1f}%)")

    if with_ground_truth > 0:
        print(f"  Correct:           {correct}/{with_ground_truth} ({100*correct/with_ground_truth:.1f}%)")

    print(f"  Avg Processing:    {avg_time:.2f}s")
    print(f"  Throughput:        {3600/avg_time:.0f} req/hour (single)")

    # Detailed results table
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    print(f"{'Filename':<35} {'True':>8} {'Pred':>8} {'Status':>8} {'Time':>8}")
    print("─"*70)

    for r in results:
        fname = r['filename'][:32] + '...' if len(r['filename']) > 35 else r['filename']
        true = r.get('true_value', 'N/A')
        pred = r.get('predicted') if r['success'] else 'FAIL'
        status = '✅' if r.get('correct') == True else '❌' if r.get('correct') == False else '⚠️'
        time_str = f"{r.get('time', 0):.2f}s"

        print(f"{fname:<35} {true:>8} {pred:>8} {status:>8} {time_str:>8}")

    print("="*70)

    # Save results
    results_file = Path(samples_file).parent / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total': total,
                'successful': successful,
                'correct': correct,
                'with_ground_truth': with_ground_truth,
                'accuracy': 100*correct/with_ground_truth if with_ground_truth > 0 else 0,
                'avg_time': avg_time
            },
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")


def main():
    API_URL = "http://localhost:8000"
    SAMPLES_FILE = r"F:\Workspace\Project\api\test_samples_base64.json"

    import sys

    max_tests = None
    if len(sys.argv) > 1:
        try:
            max_tests = int(sys.argv[1])
        except:
            pass

    test_base64_samples(API_URL, SAMPLES_FILE, max_tests)


if __name__ == "__main__":
    main()

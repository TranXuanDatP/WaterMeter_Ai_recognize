"""
Test Complete Pipeline with Beam Search Decoder

Quick test to verify beam search integration works correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pipeline_m1_m2_m3_m4 import CompletePipeline, Config

# ============================================
# CONFIGURATION
# ============================================

# Test image (one of the previously failed cases)
TEST_IMAGE = r"F:\Workspace\Project\results\test_pipeline\m5_black_digits\meter4_00003_validate_00003_0070e981653c4e0eb2209b78fb3f9ce2_black_digits.jpg"
GROUND_TRUTH = "441"

# ============================================
# TEST PIPELINE
# ============================================

print("=" * 80)
print("TESTING COMPLETE PIPELINE WITH BEAM SEARCH DECODER")
print("=" * 80)

# Test configurations
test_configs = [
    {'name': 'Greedy (Baseline)', 'method': 'greedy', 'beam_width': 1},
    {'name': 'Beam Search (Width=5)', 'method': 'beam', 'beam_width': 5},
    {'name': 'Beam Search (Width=10)', 'method': 'beam', 'beam_width': 10},
    {'name': 'Prefix Beam (Width=10)', 'method': 'prefix_beam', 'beam_width': 10},
]

results = []

for config in test_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"{'='*80}")

    # Create config
    pipeline_config = Config()
    pipeline_config.BEAM_SEARCH_METHOD = config['method']
    pipeline_config.BEAM_WIDTH = config['beam_width']
    pipeline_config.OUTPUT_DIR = f"F:/Workspace/Project/results/test_pipeline_beam_search/{config['method']}_{config['beam_width']}"

    # Initialize pipeline
    pipeline = CompletePipeline(pipeline_config)

    # Process image
    result = pipeline.process_single_image(TEST_IMAGE, save_intermediates=False)

    if 'error' in result:
        print(f"ERROR: {result['error']}")
        results.append({
            'config': config['name'],
            'predicted': 'ERROR',
            'correct': False,
            'confidence': 0.0
        })
        continue

    predicted = result['final_reading']
    confidence = result['final_confidence']
    is_correct = predicted == GROUND_TRUTH

    print(f"\nResults:")
    print(f"  Ground Truth:  {GROUND_TRUTH}")
    print(f"  Predicted:     {predicted}")
    print(f"  Confidence:    {confidence:.4f}")
    print(f"  Status:        {'✓ CORRECT' if is_correct else '✗ WRONG'}")

    results.append({
        'config': config['name'],
        'predicted': predicted,
        'correct': is_correct,
        'confidence': confidence
    })

# ============================================
# SUMMARY
# ============================================

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nGround Truth: {GROUND_TRUTH}")
print(f"\n{'Configuration':<30} {'Predicted':<15} {'Status':<10} {'Confidence':<10}")
print(f"{'-'*80}")

for r in results:
    status = "✓ PASS" if r['correct'] else "✗ FAIL"
    print(f"{r['config']:<30} {r['predicted']:<15} {status:<10} {r['confidence']:.4f}")

print(f"{'='*80}")

# Count passes
passes = sum(1 for r in results if r['correct'])
print(f"\nPass Rate: {passes}/{len(results)} ({passes/len(results)*100:.1f}%)")

print("\n" + "=" * 80)
print("BEAM SEARCH INTEGRATION TEST COMPLETE")
print("=" * 80)

print("\n💡 Next Steps:")
print("  1. If all tests pass, beam search is working correctly")
print("  2. Run full pipeline on test set to verify 96% accuracy")
print("  3. Update production pipeline to use beam search by default")
print("=" * 80)

"""
Quick Full Pipeline Test on data_4digit2 (Sample Only)

This script runs a quick test on the first 100 images from data_4digit2
to verify the full pipeline works correctly before running on the full dataset.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Config and main from the full pipeline
# Modify SAMPLE_SIZE before running
import scripts.full_pipeline_data_4digit2 as fp

# Override SAMPLE_SIZE
fp.Config.SAMPLE_SIZE = 100

print("=" * 80)
print("QUICK PIPELINE TEST (First 100 images)")
print("=" * 80)
print("This will run the full pipeline on the first 100 images.")
print("If successful, run the full pipeline with:")
print("  python scripts/full_pipeline_data_4digit2.py")
print("=" * 80)
print()

# Execute main
if __name__ == "__main__":
    fp.main()

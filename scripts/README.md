# Scripts Directory

## Organization

This directory contains Python scripts organized by purpose:

### Main Scripts (Current)
- `pipeline_m1_m2_m3_m3_5_m4.py` - **Main pipeline** with all stages (M1→M2→M3→M3.5→M4)
- `pipeline_m1_m2_m3_m4.py` - Alternative pipeline (without M3.5)
- `pipeline_with_fixed_m2.py` - Pipeline with fixed M2 orientation
- `pipeline_with_lower_threshold.py` - Pipeline with lower detection threshold

### M2 Orientation Scripts
- `m2_orientation_fixed.py` - M2 orientation correction
- `m2_align_crops.py` - M2 alignment for cropped images
- `m2_rotate_data_4digit2.py` - M2 rotation for 4-digit data
- `m2_rotate_data_4digit2_correct.py` - Corrected M2 rotation
- `test_m2orientation_model.py` - Test M2 orientation model
- `test_both_m2_models.py` - Compare M2 models
- `test_grouprnorm_vs_layernorm.py` - Test normalization approaches
- `inspect_m2orientation.py` - Inspect M2 orientation
- `inspect_m2_model.py` - Inspect M2 model

### Utility Scripts
- `log_pipeline_results.py` - Log pipeline results
- `colab_config.py` - Colab configuration

### Subdirectories

#### `analysis/`
Analysis scripts for pipeline results and failures:
- `analyze_failures.py`
- `analyze_m1_failures.py`
- `analyze_pipeline_failures.py`
- `analyze_pipeline_results.py`
- `count_failures_detailed.py`

#### `archive/`
Old/deprecated scripts:
- `extract_*.py` - M3 extraction scripts
- `show_*.py` - Visualization scripts
- `visualize_*.py` - Visualization scripts
- `compare_*.py` - Comparison scripts
- `full_pipeline_data_4digit2*.py` - Old pipeline versions
- And more...

#### `debug/`
Debugging and verification scripts:
- `debug_m2_*.py` - M2 debugging
- `verify_*.py` - Verification scripts

#### `test_old/`
Old test scripts (archived):
- `test_m2_*.py` - M2 tests
- `test_m3*.py` - M3 tests
- `test_m4*.py` - M4/OCR tests
- `test_pipeline*.py` - Pipeline tests
- And more...

## Usage

### Run Main Pipeline
```bash
python scripts/pipeline_m1_m2_m3_m3_5_m4.py \
  --input /path/to/images \
  --output /path/to/results
```

### Test M2 Model
```bash
python scripts/test_m2orientation_model.py
```

### Analyze Results
```bash
python scripts/analysis/analyze_pipeline_results.py
```

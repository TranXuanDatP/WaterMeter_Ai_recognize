# Guide: Testing Pipeline on data_4digit2

## ⚠️ Important Finding

**Quick Test Result (M4 only on raw images):**
- Accuracy: **7%** (7/100) - **VERY LOW!**
- **Reason:** M4 model was trained on **M3.5 output** (black digits only), but data_4digit2 contains **RAW meter images**!

## 📊 What Happened

### ❌ Wrong Approach (What we just did):
```
data_4digit2 (RAW images)
     ↓
M4 OCR directly
     ↓
Result: 7% accuracy (WRONG!)
```

### ✅ Correct Approach (What we need to do):
```
data_4digit2 (RAW images)
     ↓
M1: Water Meter Detection (YOLO)
     ↓
M2: Orientation + Smart Rotate
     ↓
M3: ROI Detection (YOLO)
     ↓
M3.5: Black Digit Extraction (remove red digit)
     ↓
M4: OCR with Beam Search
     ↓
Expected: ~96% accuracy ✓
```

## 🔍 Evidence

**Test on same image with/without preprocessing:**

| Dataset | Image Type | Accuracy |
|---------|------------|----------|
| test_pipeline/m5_black_digits | Preprocessed (M3.5 output) | **96%** ✓ |
| data_4digit2 | Raw images (full meter) | **7%** ✗ |

**Example:**
- Raw image (data_4digit2): `320` → M4 predicts: `2` ✗
- Preprocessed image (m5_black_digits): `320` → M4 predicts: `320` ✓

## 🚀 Solutions

### Option 1: Quick Test with Existing Pipeline (RECOMMENDED)

Run full pipeline on **small sample** first:

```bash
# Test on first 50 images to verify it works
python scripts/pipeline_m1_m2_m3_m4.py \
  --input data/data_4digit2 \
  --output results/test_data_4digit2_sample \
  --single \
  --decoder beam --beam-width 10
```

**Expected time:** ~2-3 hours for 50 images
**If successful, then run on full dataset.**

### Option 2: Check if Preprocessed Data Exists

Maybe M1→M3.5 was already run before?

```bash
# Check for preprocessed data
find data/ -name "*4digit*processed*" -o -name "*4digit*m3*" -o -name "*4digit*m3.5*"
```

If found, you can run M4 directly on those preprocessed images.

### Option 3: Generate Preprocessed Data First

Run M1→M3.5 separately, then run M4:

```bash
# Step 1: Run M1→M3.5 (need to create this script)
# This generates black digit images from raw images

# Step 2: Run M4 on preprocessed images
python scripts/test_m4_ocr_beam_search.py \
  --input data/data_4digit2_preprocessed \
  --output results/m4_data_4digit2
```

## 📋 Current Status

### ✅ What Works:
- M4 OCR with Beam Search (**96% accuracy** on preprocessed images)
- Test scripts: `test_m4_ocr_beam_search.py`, `compare_m4_decoders.py`
- Pipeline integration: `pipeline_m1_m2_m3_m4.py` (has beam search)

### ❌ What Doesn't Work:
- Running M4 directly on raw images (7% accuracy)
- M2 model has architecture mismatch (needs fixing)

### ⏳ What Needs To Be Done:
1. **Fix M2 model** or use simple rotation fallback
2. **Run full pipeline** on data_4digit2 (M1→M2→M3→M3.5→M4)
3. **Monitor and log** results

## 💡 Recommendation

**For now, use the test dataset that already works:**
- `results/test_pipeline/m5_black_digits/` - Already preprocessed
- 50 images with known labels
- 96% accuracy with beam search

**For data_4digit2 (6527 images):**
- Need to fix M2 model first
- Then run full pipeline M1→M2→M3→M3.5→M4
- This will take several hours

## 🎯 Next Steps

1. **Fix M2 model architecture** (known issue from earlier)
2. **Test full pipeline on 50 images** from data_4digit2
3. **If successful, run on full dataset** (all 6527 images)
4. **Log and analyze results**

---

**Created:** 2026-03-08
**Status:** ⚠️  Waiting for M2 model fix
**Priority:** Medium (can use test dataset in meantime)

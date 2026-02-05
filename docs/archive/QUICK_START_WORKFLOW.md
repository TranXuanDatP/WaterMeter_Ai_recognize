# Quick Start: Water Meter Development Workflow

**Get started in 5 minutes**

---

## Overview

Your Water Meter Reading project now has a complete documentation and workflow system:

1. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Complete technical documentation
2. **[DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)** - Step-by-step development guide
3. **This file** - Quick reference and common tasks

---

## Quick Reference

### I Want To... 🔍

| Task | Go To | Command |
|------|-------|---------|
| **Understand the system** | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | - |
| **Improve the model** | [DEVELOPMENT_WORKFLOW.md Phase 1](DEVELOPMENT_WORKFLOW.md#phase-1-experiment-research--development) | `git checkout -b experiment/my-idea` |
| **Train a model** | [PROJECT_OVERVIEW.md Model Training](PROJECT_OVERVIEW.md#model-training-pipeline) | `python train_4digit_balanced.py --epochs 50` |
| **Test a model** | [PROJECT_OVERVIEW.md Inference](PROJECT_OVERVIEW.md#inference-pipeline) | `python test_4digit_integer.py --samples 100` |
| **Compare models** | [DEVELOPMENT_WORKFLOW.md Phase 3](DEVELOPMENT_WORKFLOW.md#phase-3-compare-benchmarking) | `python compare_models.py --model_a ... --model_b ...` |
| **Debug issues** | [PROJECT_OVERVIEW.md Troubleshooting](PROJECT_OVERVIEW.md#troubleshooting) | See troubleshooting table |
| **Deploy to production** | [DEVELOPMENT_WORKFLOW.md Phase 5](DEVELOPMENT_WORKFLOW.md#phase-5-deploy-production-update) | See deployment checklist |

---

## Common Workflows

### Workflow 1: Try a New Idea (Experiment)

**Time**: 2-4 hours | **Difficulty**: Medium

```bash
# 1. Create experiment branch
git checkout -b experiment/my-new-idea

# 2. Copy and modify training script
cp train_4digit_balanced.py train_4digit_experiment.py
# Edit train_4digit_experiment.py with your changes

# 3. Quick test (5 epochs)
python train_4digit_experiment.py --epochs 5 --experiment_name "my_idea_test"

# 4. If promising, full training (50 epochs)
python train_4digit_experiment.py --epochs 50 --experiment_name "my_idea_full"

# 5. Test and compare
python test_4digit_integer.py --model models/digit_classifier_4digit_experiment.pth --samples 100

# 6. Document results
# Create docs/experiments/my_new_idea_report.md
```

**See**: [DEVELOPMENT_WORKFLOW.md Phase 1-4](DEVELOPMENT_WORKFLOW.md#phase-1-experiment-research--development)

---

### Workflow 2: Train from Scratch

**Time**: 4-8 hours | **Difficulty**: Easy

```bash
# 1. Prepare data
# Ensure data/data_4digit.csv and data/images_4digit/ exist

# 2. Train model
python train_4digit_balanced.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --epochs 50 \
    --batch_size 32 \
    --patience 8

# 3. Monitor training
# Watch console output for validation accuracy and loss

# 4. Test model
python test_4digit_integer.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --samples 100

# 5. Use model
python -c "
from train_4digit_balanced import Integer4DigitReader
reader = Integer4DigitReader('models/digit_classifier_4digit_balanced.pth', num_digits=3)
result = reader.read_meter('data/images_4digit/sample.jpg')
print(result['predicted_value'])
"
```

**See**: [PROJECT_OVERVIEW.md Training Pipeline](PROJECT_OVERVIEW.md#model-training-pipeline)

---

### Workflow 3: Debug Low Accuracy

**Time**: 1-2 hours | **Difficulty**: Medium

```bash
# 1. Visualize preprocessing
python visualize_preprocessing.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --output debug_images \
    --samples 10

# 2. Check debug_images/ folder
# Look for issues in:
#   - Panel extraction (wrong corners?)
#   - Digit segmentation (bad splits?)
#   - Augmentation (too much blur?)

# 3. Test with Bayesian method
python test_4digit_integer.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --samples 100 \
    --bayesian \
    --prior_strategy "true_value"

# 4. Analyze failures
# Look at patterns in wrong predictions

# 5. Consult troubleshooting guide
# See PROJECT_OVERVIEW.md Troubleshooting section
```

**See**: [PROJECT_OVERVIEW.md Troubleshooting](PROJECT_OVERVIEW.md#troubleshooting)

---

### Workflow 4: Compare Two Models

**Time**: 30 minutes | **Difficulty**: Easy

```bash
# 1. Test both models
python test_4digit_integer.py \
    --model models/digit_classifier_4digit_balanced.pth \
    --csv data/data_4digit.csv \
    --samples 100 \
    --output results_baseline.json

python test_4digit_integer.py \
    --model models/digit_classifier_4digit_experiment.pth \
    --csv data/data_4digit.csv \
    --samples 100 \
    --output results_experiment.json

# 2. Compare metrics
# Check:
#   - Validation accuracy
#   - Bayesian accuracy
#   - Inference time
#   - Model size

# 3. Make decision
# Use decision matrix in DEVELOPMENT_WORKFLOW.md Phase 3
```

**See**: [DEVELOPMENT_WORKFLOW.md Phase 3](DEVELOPMENT_WORKFLOW.md#phase-3-compare-benchmarking)

---

### Workflow 5: Deploy New Model

**Time**: 1-2 hours | **Difficulty**: Hard

```bash
# 1. Pre-deployment checks
python test_4digit_integer.py --full_test
python verify_model.py --model models/new_model.pth

# 2. Archive old model
mv models/digit_classifier_4digit_balanced.pth \
   models/archive/digit_classifier_4digit_balanced_v1_$(date +%Y%m%d).pth

# 3. Deploy new model
cp models/new_model.pth models/digit_classifier_4digit_balanced.pth

# 4. Update documentation
# Update PROJECT_OVERVIEW.md with new metrics
# Update README.md if needed

# 5. Monitor
# Watch for issues in production
# Be ready to rollback
```

**See**: [DEVELOPMENT_WORKFLOW.md Phase 5](DEVELOPMENT_WORKFLOW.md#phase-5-deploy-production-update)

---

## Current Status

### Production Model
- **File**: `models/digit_classifier_4digit_balanced.pth`
- **Architecture**: CNN with balanced augmentation
- **Validation Accuracy**: TBD (training in progress)
- **Bayesian Accuracy**: TBD
- **Status**: 🔄 Training

### Next Steps
1. ⏳ Complete training (50+ epochs)
2. ⏳ Evaluate performance
3. ⏳ Compare with previous versions
4. ⏳ Document final results

---

## File Locations

### Training Scripts
```
Project/
├── train_4digit_balanced.py    # 4-digit training (current)
├── train_5digit_balanced.py    # 5-digit training (current)
├── test_4digit_integer.py      # 4-digit testing
└── test_5digit_integer.py      # 5-digit testing
```

### Documentation
```
docs/
├── PROJECT_OVERVIEW.md         # Technical documentation
├── DEVELOPMENT_WORKFLOW.md     # Development guide
└── QUICK_START_WORKFLOW.md     # This file
```

### Data & Models
```
data/
├── data_4digit.csv             # 4-digit dataset (6,485 samples)
├── images_4digit/              # 4-digit images
├── data.csv                    # 5-digit dataset (1,244 samples)
└── images/                     # 5-digit images

models/
├── digit_classifier_4digit_balanced.pth   # Current 4-digit model
├── digit_classifier_5digit_balanced.pth   # Current 5-digit model
└── archive/                   # Previous model versions
```

---

## Key Concepts

### Probabilistic Matching
The system uses 3 prediction methods:
1. **Argmax**: Highest probability per digit (baseline)
2. **Bayesian**: Incorporates prior knowledge (e.g., previous reading) - **80% accuracy**
3. **Expected Value**: Weighted average of predictions

**Recommendation**: Always use Bayesian method when prior knowledge is available.

### Balanced Augmentation
Current approach (v4.0):
- Moderate ColorJitter (brightness=0.3, contrast=0.3)
- Light Rotation (5°)
- Light Perspective (0.1)
- **NO GaussianBlur** (preserves sharp edges)
- Dropout 0.3 (reduced from 0.5)

**Key insight**: Too much augmentation (especially blur) harms digit recognition.

### Integer-Only Focus
System ignores decimal part, focusing only on integer digits:
- 4-digit meters: 3 integer digits (XXX format)
- 5-digit meters: 5 integer digits (XXXXX format)

**Benefit**: Simplifies problem, improves accuracy on integer part.

---

## Getting Help

### Documentation
- **Technical details**: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- **Development process**: [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)
- **Original README**: [../README.md](../README.md)

### Common Issues
| Issue | Solution |
|-------|----------|
| Low accuracy | Train longer (50+ epochs), use Bayesian method |
| Overfitting | Add dropout, increase augmentation |
| CUDA OOM | Reduce batch_size to 16 |
| Wrong predictions | Check num_digits matches dataset |

### Troubleshooting
See [PROJECT_OVERVIEW.md Troubleshooting](PROJECT_OVERVIEW.md#troubleshooting) for detailed solutions.

---

## Tips & Best Practices

### Training ✅
- Always use balanced augmentation (current best)
- Train for 50+ epochs with early stopping
- Use Bayesian method for inference
- Monitor validation loss closely

### Experimentation 🧪
- Change ONE thing at a time
- Document hypothesis before starting
- Keep baseline for comparison
- Archive all experiments (even failures)

### Deployment 🚀
- Test thoroughly before deploying
- Keep old models archived
- Monitor post-deployment
- Have rollback plan ready

---

## Workflow Diagram

```
┌─────────────┐
│   Have Idea │
└──────┬──────┘
       │
       ▼
┌─────────────┐       ┌──────────────┐
│  Experiment │──────>│  Validate    │
└──────┬──────┘       └──────┬───────┘
       │                     │
       │                     ▼
       │              ┌──────────────┐
       │              │   Compare    │
       │              └──────┬───────┘
       │                     │
       │         ┌───────────┴───────────┐
       │         │                       │
       │    ▼    ▼                   ▼  ▼
       │ ┌────────┐            ┌──────────┐
       │ │ Reject │            │  Adopt   │
       │ └────────┘            └─────┬────┘
       │                             │
       │                             ▼
       │                      ┌──────────┐
       │                      │ Document │
       │                      └─────┬────┘
       │                             │
       │                             ▼
       │                      ┌──────────┐
       └─────────────────────>│  Deploy  │
                              └──────────┘
```

---

## Next Actions

### For New Developers
1. Read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (15 minutes)
2. Run [Workflow 2: Train from Scratch](#workflow-2-train-from-scratch)
3. Explore test results and debug images

### For Improving Model
1. Read [DEVELOPMENT_WORKFLOW.md Phase 1](DEVELOPMENT_WORKFLOW.md#phase-1-experiment-research--development)
2. Run [Workflow 1: Try a New Idea](#workflow-1-try-a-new-idea-experiment)
3. Document results

### For Deploying
1. Complete training and validation
2. Run [Workflow 4: Compare Two Models](#workflow-4-compare-two-models)
3. Follow [Workflow 5: Deploy New Model](#workflow-5-deploy-new-model)

---

**Version**: 1.0
**Last Updated**: 2026-01-16
**For**: Water Meter Reading Project Team

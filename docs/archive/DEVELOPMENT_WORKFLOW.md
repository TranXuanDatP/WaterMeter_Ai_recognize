# Water Meter Reading - Custom Development Workflow

**Purpose**: Streamlined workflow for improving and maintaining the Water Meter Reading System
**Created**: 2026-01-16
**Based on**: BMAD methodology, adapted for this specific project

---

## Workflow Overview

This workflow guides you through the development lifecycle of the water meter reading system, from ideation to deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                 Water Meter Development Workflow            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. EXPERIMENT      → Try new ideas, techniques             │
│  2. VALIDATE        → Test on validation set                │
│  3. COMPARE         → Benchmark against current best        │
│  4. DOCUMENT        → Record findings and results           │
│  5. DEPLOY          → Update production if improved         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Experiment (Research & Development)

### Goal
Explore new techniques, architectures, or approaches to improve model accuracy.

### When to Use
- Current model accuracy is insufficient
- New research papers suggest better methods
- Need to handle new meter types or edge cases

### Process

#### Step 1.1: Define Experiment
```yaml
Experiment Template:
  name: "Descriptive name (e.g., ResNet50 backbone)"
  hypothesis: "What you expect to improve and why"
  baseline: "Current best metric to beat"
  metrics:
    - Validation accuracy
    - Bayesian accuracy
    - Training time
    - Inference time
  duration: "Estimated time (e.g., 2 hours)"
```

#### Step 1.2: Create Experiment Branch
```bash
# Create feature branch
git checkout -b experiment/resnet50-backbone

# Copy current best script as starting point
cp train_4digit_balanced.py train_4digit_experiment.py
```

#### Step 1.3: Implement Changes
- Modify architecture
- Adjust augmentation
- Change hyperparameters
- Add new features

#### Step 1.4: Test Implementation
```bash
# Quick sanity check (5 epochs)
python train_4digit_experiment.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --epochs 5 \
    --batch_size 32 \
    --experiment_name "resnet50_test"
```

### Checklist
- [ ] Experiment hypothesis clearly defined
- [ ] Baseline metrics documented
- [ ] Implementation tested (sanity check passes)
- [ ] Code committed with descriptive message

---

## Phase 2: Validate (Training & Testing)

### Goal
Train the experiment model and validate performance.

### Process

#### Step 2.1: Full Training Run
```bash
# Train for full duration (50+ epochs)
python train_4digit_experiment.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --epochs 50 \
    --batch_size 32 \
    --patience 8 \
    --grad_clip 1.0 \
    --experiment_name "resnet50_full"
```

#### Step 2.2: Monitor Training
- Watch validation loss curve
- Check for overfitting (train acc >> val acc)
- Ensure early stopping triggers appropriately
- Save best model checkpoint

#### Step 2.3: Test on Holdout Set
```bash
# Test with argmax method
python test_4digit_integer.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --model models/digit_classifier_4digit_experiment.pth \
    --samples 100

# Test with Bayesian method (if applicable)
python test_4digit_integer.py \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --model models/digit_classifier_4digit_experiment.pth \
    --samples 100 \
    --bayesian \
    --prior_strategy "true_value"
```

#### Step 2.4: Record Metrics
Create experiment report:
```bash
# Generate comparison report
python compare_experiments.py \
    --baseline models/digit_classifier_4digit_balanced.pth \
    --experiment models/digit_classifier_4digit_experiment.pth \
    --output docs/experiments/resnet50_report.md
```

### Checklist
- [ ] Training completed (50+ epochs or early stopped)
- [ ] Best model saved
- [ ] Validation metrics recorded
- [ ] Test metrics recorded
- [ ] Training curves saved (loss/accuracy plots)

---

## Phase 3: Compare (Benchmarking)

### Goal
Compare experiment results against baseline and decide on adoption.

### Process

#### Step 3.1: Collect Metrics
```yaml
Metrics to Compare:
  Accuracy:
    - Validation accuracy
    - Test accuracy (argmax)
    - Test accuracy (Bayesian)
  Performance:
    - Training time
    - Inference time (per image)
    - Model size
  Robustness:
    - Confidence distribution
    - Failure case analysis
    - Cross-meter type performance
```

#### Step 3.2: Statistical Significance
- Run multiple training runs (3-5 seeds)
- Calculate mean and std deviation
- Perform statistical test if needed

#### Step 3.3: Failure Analysis
```bash
# Visualize and categorize failures
python analyze_failures.py \
    --model models/digit_classifier_4digit_experiment.pth \
    --csv data/data_4digit.csv \
    --images data/images_4digit \
    --output docs/experiments/failure_analysis/
```

#### Step 3.4: Decision Matrix

| Criterion | Baseline | Experiment | Winner | Notes |
|-----------|----------|------------|--------|-------|
| Val Accuracy | 24.57% | ?% | - | Target: >30% |
| Bayesian Acc | 80% | ?% | - | Target: >85% |
| Inference Time | 50ms | ?ms | - | Must be <100ms |
| Model Size | 419KB | ?MB | - | Must be <10MB |
| Simplicity | High | Medium | - | Prefer simpler |

### Decision Rules
- **Adopt if**: ≥5% accuracy improvement AND no major drawbacks
- **Consider if**: 2-5% improvement OR other benefits (speed, size)
- **Reject if**: <2% improvement OR any major regression
- **Retry if**: Promising but needs tuning

### Checklist
- [ ] All metrics compared
- [ ] Statistical significance tested
- [ ] Failure cases analyzed
- [ ] Decision made (adopt/consider/reject/retry)

---

## Phase 4: Document (Knowledge Capture)

### Goal
Document experiment results for future reference.

### Process

#### Step 4.1: Create Experiment Report
Template: [docs/experiments/EXPERIMENT_REPORT_TEMPLATE.md](EXPERIMENT_REPORT_TEMPLATE.md)

Include:
- Abstract (1 paragraph)
- Hypothesis and motivation
- Method changes
- Results (tables/plots)
- Discussion and analysis
- Conclusion and recommendations
- Appendix (code snippets, configs)

#### Step 4.2: Update Project Documentation
- Update [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) if adopted
- Update [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) if new files
- Update README.md if user-facing changes

#### Step 4.3: Archive Old Models
```bash
# Move old model to archive
mv models/digit_classifier_4digit_balanced.pth \
   models/archive/digit_classifier_4digit_balanced_v1_$(date +%Y%m%d).pth

# Update model registry
echo "v1_$(date +%Y%m%d): Balanced augmentation, 24.57% val acc" \
    >> models/MODEL_REGISTRY.txt
```

### Checklist
- [ ] Experiment report created
- [ ] Project docs updated (if adopted)
- [ ] Old models archived
- [ ] Git commit with documentation

---

## Phase 5: Deploy (Production Update)

### Goal
Update production system with improved model.

### Process

#### Step 5.1: Pre-Deployment Checks
```bash
# Run full test suite
python test_4digit_integer.py --full_test

# Verify model compatibility
python verify_model.py \
    --model models/digit_classifier_4digit_experiment.pth

# Check backward compatibility
python test_api.py --regression_test
```

#### Step 5.2: Staged Rollout
1. **Canary release** (10% of traffic)
2. **Monitor** for errors and performance
3. **Gradual rollout** (50%, then 100%)

#### Step 5.3: Update Documentation
- Update model version in configs
- Update API documentation
- Communicate changes to stakeholders

#### Step 5.4: Monitor Post-Deployment
- Track accuracy in production
- Monitor inference latency
- Collect user feedback
- Watch for edge cases

### Checklist
- [ ] All tests pass
- [ ] Canary deployment successful
- [ ] Documentation updated
- [ ] Monitoring in place
- [ ] Rollback plan ready

---

## Common Experiment Templates

### Template 1: Architecture Change
```yaml
name: "New Architecture (e.g., ResNet50)"
type: architecture
hypothesis: "Deeper network will capture more features"
implementation:
  - Modify CNN architecture in DigitClassifier
  - Adjust input size if needed
  - Update model save format
training:
  epochs: 50
  batch_size: 16  # May need smaller for larger model
metrics:
  - Accuracy
  - Training time
  - Model size
```

### Template 2: Augmentation Tuning
```yaml
name: "Augmentation Strategy (e.g., MixUp)"
type: augmentation
hypothesis: "MixUp will improve generalization"
implementation:
  - Add MixUp to training loop
  - Adjust augmentation parameters
training:
  epochs: 50
  batch_size: 32
metrics:
  - Validation accuracy
  - Overfitting gap
```

### Template 3: Hyperparameter Optimization
```yaml
name: "Hyperparameter Search (e.g., Learning Rate)"
type: hyperparameter
hypothesis: "Optimal LR will improve convergence"
implementation:
  - Grid search or random search
  - Try LR values: [0.0001, 0.001, 0.01]
training:
  epochs: 30  # Shorter for search
metrics:
  - Best validation accuracy
  - Convergence speed
```

### Template 4: Post-Processing Improvement
```yaml
name: "New Post-Processing (e.g., CRF)"
type: postprocessing
hypothesis: "CRF will smooth digit predictions"
implementation:
  - Add post-processing step
  - Keep model unchanged
training:
  epochs: 0  # No retraining needed
metrics:
  - Test accuracy
  - Inference time
```

---

## Quick Reference Commands

### Training
```bash
# Quick test (5 epochs)
python train_4digit_balanced.py --epochs 5 --quick_test

# Full training (50 epochs)
python train_4digit_balanced.py --epochs 50 --full_train

# Resume from checkpoint
python train_4digit_balanced.py --resume models/checkpoint_epoch25.pth
```

### Testing
```bash
# Test on validation set
python test_4digit_integer.py --csv data/data_4digit.csv --samples 100

# Interactive testing
python test_4digit_integer.py --interactive

# Full evaluation
python test_4digit_integer.py --full_eval --save_results
```

### Comparison
```bash
# Compare two models
python compare_models.py \
    --model_a models/balanced.pth \
    --model_b models/experiment.pth \
    --csv data/data_4digit.csv
```

### Visualization
```bash
# Visualize predictions
python visualize_predictions.py \
    --model models/experiment.pth \
    --images data/images_4digit \
    --output debug_images/
```

---

## Git Workflow

### Branch Naming
```
experiment/<feature-name>
bugfix/<issue-description>
refactor/<component-name>
docs/<documentation-update>
```

### Commit Messages
```
feat: add ResNet50 backbone for digit classification
exp: test MixUp augmentation strategy
fix: correct digit segmentation for 5-digit meters
docs: update experiment report for ResNet50
refactor: extract data augmentation to separate module
perf: optimize inference speed by 20%
```

### Example Workflow
```bash
# 1. Start experiment
git checkout -b experiment/attention-mechanism

# 2. Implement and test
# ... make changes ...
git add .
git commit -m "exp: add self-attention layer to CNN"

# 3. Train and validate
python train_4digit_experiment.py --epochs 50
python test_4digit_integer.py --model experiment.pth

# 4. Document results
# ... write experiment report ...

# 5. Merge if successful
git checkout main
git merge experiment/attention-mechanism
git tag -a v4.1-attention -m "Add attention mechanism"
```

---

## Troubleshooting Workflow

### Problem: Training Loss Not Decreasing
**Workflow**:
1. Check learning rate (too high? try 0.0001)
2. Verify data loading (correct labels?)
3. Examine gradients (vanishing/exploding?)
4. Simplify model (reduce capacity)
5. Check for bugs (forward pass correct?)

### Problem: Overfitting
**Workflow**:
1. Increase dropout (0.3 → 0.5)
2. Add more augmentation
3. Reduce model capacity
4. Add weight decay
5. Implement early stopping

### Problem: Low Accuracy
**Workflow**:
1. Review data quality (labels correct?)
2. Visualize predictions (find patterns)
3. Analyze failures (categorize errors)
4. Try different architecture
5. Ensemble methods

---

## Metrics Dashboard Template

Track these metrics for each experiment:

```yaml
Experiment: _____
Date: _____
Baseline: _____

Results:
  Validation Accuracy: _____% (target: >30%)
  Bayesian Accuracy: _____% (target: >85%)
  Training Time: _____ hours
  Inference Time: _____ ms/image
  Model Size: _____ MB

Decision: [ ] Adopt  [ ] Consider  [ ] Reject  [ ] Retry

Next Steps:
  _____
```

---

## Best Practices

### DO ✅
- Always start with a hypothesis
- Document everything (even failures)
- Use version control for models
- Test on held-out set
- Monitor training closely
- Keep experiments reproducible
- Share findings with team

### DON'T ❌
- Don't skip documentation
- Don't deploy without testing
- Don't ignore negative results
- Don't overfit to validation set
- Don't forget to archive old models
- Don't make too many changes at once
- Don't skip statistical testing

---

**Workflow Version**: 1.0
**Last Updated**: 2026-01-16
**Maintainer**: Project Team

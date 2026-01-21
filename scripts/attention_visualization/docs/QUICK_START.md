# Quick Start Guide

## Getting Started with Attention Visualization

This guide provides step-by-step instructions to run the attention visualization experiment and understand the outputs.

---

## Prerequisites

### Required Software
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Required Data
- Trained PointerGeneratorTransformer model checkpoint
- Processed test dataset (pickle format)
- Model configuration file (YAML format)

### Python Dependencies
```bash
pip install torch numpy pandas matplotlib seaborn pyyaml tqdm scipy
```

---

## Quick Start

### Step 1: Run Single Dataset Experiment

```bash
cd /data/next_loc_clean_v2

# For DIY dataset
python scripts/attention_visualization/run_attention_experiment.py --dataset diy --seed 42

# For Geolife dataset
python scripts/attention_visualization/run_attention_experiment.py --dataset geolife --seed 42
```

**Expected Output**:
```
======================================================================
ATTENTION VISUALIZATION EXPERIMENT - DIY
======================================================================
Device: cuda
[1/6] Loading model and data...
Loaded model from: .../checkpoints/best.pt
  Parameters: 1,081,280
  Test samples: 12,368

[2/6] Extracting attention weights...
  Extracted attention for 12368 samples

[3/6] Computing aggregate statistics...

[4/6] Selecting best samples for visualization...
  Selected 10 samples

[5/6] Generating visualizations...
  - Aggregate pointer attention...
  - Gate analysis...
  - Self-attention aggregate...
  - Position bias analysis...
  - Sample overview...
  - Individual sample attention...

[6/6] Generating statistical tables...

======================================================================
EXPERIMENT SUMMARY
======================================================================
Dataset: DIY
Samples Analyzed: 12368
Prediction Accuracy: 56.58%
Mean Gate Value: 0.7872 ± 0.1366
Gate (Correct): 0.8168
Gate (Incorrect): 0.7486
Mean Pointer Entropy: 2.3358

Output saved to: scripts/attention_visualization/results/diy
======================================================================
```

### Step 2: Run Cross-Dataset Comparison

**Important**: Run both single-dataset experiments first.

```bash
python scripts/attention_visualization/cross_dataset_comparison.py
```

**Expected Output**:
```
============================================================
CROSS-DATASET ATTENTION COMPARISON
============================================================

Loading experiment data...
Generating comparison table...
Metric,DIY,Geolife
Dataset,DIY Check-in,GeoLife GPS
Test Samples,12368.0,3502.0
...

Generating comparison visualizations...
Generating summary statistics...

============================================================
KEY FINDINGS
============================================================
Finding,DIY,Geolife,Interpretation
Higher pointer reliance,Gate: 0.787,Gate: 0.627,DIY shows stronger copy mechanism preference
...

Results saved to: scripts/attention_visualization/results
============================================================
```

---

## Understanding the Output

### Directory Structure After Running

```
scripts/attention_visualization/results/
├── diy/                              # DIY dataset results
│   ├── aggregate_pointer_attention.png
│   ├── attention_statistics.csv
│   ├── experiment_metadata.json
│   ├── gate_analysis.png
│   ├── position_attention.csv
│   ├── position_bias_analysis.png
│   ├── sample_01_attention.png
│   ├── ... (sample_02 - sample_10)
│   ├── samples_overview.png
│   ├── selected_samples.csv
│   └── self_attention_aggregate.png
├── geolife/                          # Geolife dataset results
│   └── ... (same structure as diy/)
├── cross_dataset_attention_patterns.png
├── cross_dataset_comparison.csv
├── cross_dataset_gate_comparison.png
└── key_findings.csv
```

### Key Files to Review

| File | Description | Key Metrics |
|------|-------------|-------------|
| `attention_statistics.csv` | Main metrics | Accuracy, Gate, Entropy |
| `position_attention.csv` | Per-position attention | Position-wise breakdown |
| `selected_samples.csv` | Top 10 samples | Gate, confidence, length |
| `experiment_metadata.json` | Experiment config | Model config, timestamp |
| `key_findings.csv` | Comparison summary | DIY vs Geolife |

### Quick Metrics Reference

**Good to Know**:
- **Gate > 0.5**: Pointer mechanism preferred
- **Gate > 0.8**: Strong pointer preference
- **Entropy < 2.0**: Focused attention
- **Entropy > 2.5**: Spread attention

---

## Common Tasks

### Task 1: View Summary Statistics

```bash
cat scripts/attention_visualization/results/diy/attention_statistics.csv
```

### Task 2: View Position Attention

```bash
cat scripts/attention_visualization/results/diy/position_attention.csv
```

### Task 3: Compare Datasets

```bash
cat scripts/attention_visualization/results/cross_dataset_comparison.csv
```

### Task 4: View Experiment Config

```bash
cat scripts/attention_visualization/results/diy/experiment_metadata.json
```

---

## Programmatic Usage

### Extract Attention for Custom Analysis

```python
import torch
from scripts.attention_visualization.attention_extractor import (
    AttentionExtractor, 
    extract_batch_attention,
    compute_attention_statistics
)

# Load your model
model = load_your_model()
device = torch.device('cuda')
model = model.to(device)
model.eval()

# Create extractor
extractor = AttentionExtractor(model, device)

# Extract from dataloader
results = extract_batch_attention(extractor, test_loader, num_samples=100)

# Compute statistics
stats = compute_attention_statistics(results)

print(f"Mean gate: {stats['gate_mean']:.4f}")
print(f"Mean entropy: {stats['pointer_entropy_mean']:.4f}")
print(f"Accuracy: {stats['accuracy']:.4f}")
```

### Extract Single Sample

```python
# For a single batch
attention_data = extractor.extract_attention(x, x_dict)

# Access components
pointer_probs = attention_data['pointer_probs']  # [batch, seq_len]
gate_values = attention_data['gate_values']      # [batch, 1]
self_attention = attention_data['self_attention'] # List of [batch, heads, seq, seq]

print(f"Gate: {gate_values[0].item():.4f}")
print(f"Max pointer attention: {pointer_probs[0].max().item():.4f}")
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in the experiment config or process fewer samples:

```python
# In run_attention_experiment.py, modify:
test_loader = DataLoader(test_ds, batch_size=32, ...)  # Reduce from 64
```

### Issue: Model Checkpoint Not Found

**Solution**: Verify the experiment directory path in `EXPERIMENT_CONFIGS`:

```python
EXPERIMENT_CONFIGS = {
    'diy': {
        'experiment_dir': '/correct/path/to/diy_pointer_v45_...',
        ...
    }
}
```

### Issue: Test Data Not Found

**Solution**: Ensure the pickle files exist:

```bash
ls -la /data/next_loc_clean_v2/data/diy_eps50/processed/
# Should show: diy_eps50_prev7_test.pk
```

### Issue: Import Errors

**Solution**: Add project root to Python path:

```python
import sys
sys.path.insert(0, '/data/next_loc_clean_v2')
```

Or run from project root:

```bash
cd /data/next_loc_clean_v2
python scripts/attention_visualization/run_attention_experiment.py --dataset diy
```

---

## Next Steps

After running the experiment:

1. **Review the main README**: `docs/README.md` for comprehensive documentation
2. **Understand the theory**: `docs/THEORETICAL_BACKGROUND.md` for mathematical foundations
3. **Interpret results**: `docs/RESULTS_ANALYSIS.md` for detailed analysis
4. **Explore visualizations**: `docs/VISUALIZATION_GALLERY.md` for plot guide
5. **Check code details**: `docs/CODE_DOCUMENTATION.md` for API reference

---

## Support

For issues or questions:
1. Check the documentation in `docs/`
2. Review the code comments in source files
3. Examine the experiment logs in results directory

---

*Quick Start Guide - Version 1.0*
*Last Updated: January 2026*

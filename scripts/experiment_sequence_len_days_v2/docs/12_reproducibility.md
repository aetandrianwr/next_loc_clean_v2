# 12. Reproducibility Guide

## How to Reproduce This Experiment

---

## Document Overview

| Item | Details |
|------|---------|
| **Document Type** | Reproducibility Documentation |
| **Audience** | Developers, Researchers wanting to reproduce |
| **Reading Time** | 10-12 minutes |
| **Prerequisites** | Python, Linux command line familiarity |

---

## 1. Environment Requirements

### 1.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU | 4 GB VRAM | 8+ GB VRAM |
| Storage | 10 GB free | 20+ GB free |

**GPU Options**:
- NVIDIA RTX 2060+ (Consumer)
- NVIDIA T4+ (Cloud)
- CPU-only possible but slower

### 1.2 Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Runtime |
| PyTorch | 2.0+ | Deep learning |
| CUDA | 11.7+ | GPU acceleration |
| Linux | Ubuntu 20.04+ | Operating system |

### 1.3 Python Dependencies

```txt
# Core
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0

# Configuration
pyyaml>=6.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Metrics
scikit-learn>=1.0.0

# Utilities
tqdm>=4.62.0
```

---

## 2. Setup Instructions

### 2.1 Environment Setup

**Option A: Conda (Recommended)**
```bash
# Create environment
conda create -n mlenv python=3.9 -y
conda activate mlenv

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy pandas pyyaml matplotlib seaborn scikit-learn tqdm
```

**Option B: Virtual Environment**
```bash
# Create virtual environment
python3 -m venv mlenv
source mlenv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas pyyaml matplotlib seaborn scikit-learn tqdm
```

### 2.2 Verify Installation

```bash
# Verify Python
python --version
# Expected: Python 3.9.x or higher

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Expected: PyTorch: 2.x.x

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Expected: CUDA: True (or False for CPU-only)

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## 3. Data Requirements

### 3.1 Required Data Files

```
data/
├── diy_eps50/processed/
│   ├── diy_eps50_prev7_train.pk     # DIY training data
│   └── diy_eps50_prev7_test.pk      # DIY test data
└── geolife_eps20/processed/
    ├── geolife_eps20_prev7_train.pk # GeoLife training data
    └── geolife_eps20_prev7_test.pk  # GeoLife test data
```

### 3.2 Verify Data Integrity

```bash
# Check file existence
ls -la data/diy_eps50/processed/
ls -la data/geolife_eps20/processed/

# Check file sizes (approximate)
# diy_eps50_prev7_test.pk: ~50 MB
# geolife_eps20_prev7_test.pk: ~15 MB
```

```python
# Verify data loading
import pickle

with open('data/diy_eps50/processed/diy_eps50_prev7_test.pk', 'rb') as f:
    diy_data = pickle.load(f)
print(f"DIY samples: {len(diy_data)}")  # Expected: 12,368

with open('data/geolife_eps20/processed/geolife_eps20_prev7_test.pk', 'rb') as f:
    geolife_data = pickle.load(f)
print(f"GeoLife samples: {len(geolife_data)}")  # Expected: 3,502
```

---

## 4. Model Checkpoints

### 4.1 Required Checkpoint Files

```
experiments/
├── diy_pointer_v45_20260101_155348/
│   └── checkpoints/
│       └── best.pt                   # DIY model checkpoint
└── geolife_pointer_v45_20260101_151038/
    └── checkpoints/
        └── best.pt                   # GeoLife model checkpoint
```

### 4.2 Verify Checkpoints

```python
import torch

# Check DIY checkpoint
diy_ckpt = torch.load(
    'experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt',
    map_location='cpu'
)
print(f"DIY checkpoint keys: {diy_ckpt.keys()}")
# Expected: dict_keys(['epoch', 'model_state_dict', ...])

# Check GeoLife checkpoint
geo_ckpt = torch.load(
    'experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt',
    map_location='cpu'
)
print(f"GeoLife checkpoint keys: {geo_ckpt.keys()}")
```

---

## 5. Running the Experiment

### 5.1 Navigate to Experiment Directory

```bash
cd /data/next_loc_clean_v2/scripts/experiment_sequence_len_days_v2
```

### 5.2 Run Complete Experiment

**Method 1: Using Shell Script (Recommended)**
```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

**Method 2: Step by Step**
```bash
# Step 1: Run evaluation
python evaluate_sequence_length.py --dataset all --batch_size 64

# Step 2: Generate visualizations
python visualize_results.py
```

**Method 3: Single Dataset**
```bash
# DIY only
python evaluate_sequence_length.py --dataset diy --batch_size 64

# GeoLife only
python evaluate_sequence_length.py --dataset geolife --batch_size 64
```

### 5.3 Command Line Options

```bash
python evaluate_sequence_length.py --help

Options:
  --dataset {diy,geolife,all}  Dataset to evaluate (default: all)
  --batch_size INT             Batch size for evaluation (default: 64)
  --output_dir PATH            Output directory (default: results/)
```

---

## 6. Expected Output

### 6.1 Console Output

```
==========================================
Sequence Length Days Experiment V2
==========================================

[Step 1/2] Running evaluation...

==================================================
Evaluating DIY with prev_days=1
==================================================
Filtering sequences for prev_days=1...
Samples after filtering: 11532
Avg sequence length: 5.62 (±4.13)
Max sequence length: 29
Evaluating: 100%|████████████████| 181/181 [00:15<00:00, 11.54it/s]

Results:
  Acc@1:  50.00%
  Acc@5:  72.55%
  Acc@10: 74.65%
  MRR:    59.97%
  NDCG:   63.47%
  F1:     46.73%
  Loss:   3.7628

[... continues for all configurations ...]

[Step 2/2] Generating visualizations...
Generating performance_comparison.pdf...
Generating accuracy_heatmap.pdf...
Generating loss_curve.pdf...
Generating radar_comparison.pdf...
Generating improvement_comparison.pdf...
Generating sequence_length_distribution.pdf...
Generating samples_vs_performance.pdf...
Generating combined_figure.pdf...

==========================================
Experiment completed successfully!
Results saved to: results/
==========================================
```

### 6.2 Output Files

After successful execution:

```
results/
├── diy_sequence_length_results.json      # Raw DIY results
├── geolife_sequence_length_results.json  # Raw GeoLife results
├── full_results.csv                       # Combined CSV
├── summary_statistics.csv                 # Statistical summary
├── improvement_analysis.csv               # Improvement calculations
├── results_table.tex                      # LaTeX table
├── statistics_table.tex                   # LaTeX stats table
├── performance_comparison.pdf             # 6-panel figure
├── performance_comparison.png
├── performance_comparison.svg
├── accuracy_heatmap.pdf
├── accuracy_heatmap.png
├── loss_curve.pdf
├── loss_curve.png
├── radar_comparison.pdf
├── radar_comparison.png
├── improvement_comparison.pdf
├── improvement_comparison.png
├── sequence_length_distribution.pdf
├── sequence_length_distribution.png
├── samples_vs_performance.pdf
├── samples_vs_performance.png
├── combined_figure.pdf                    # Publication figure
├── combined_figure.png
└── combined_figure.svg
```

---

## 7. Validation Checks

### 7.1 Verify Results

Run this Python script to validate your results:

```python
import json
import numpy as np

# Load results
with open('results/diy_sequence_length_results.json') as f:
    diy = json.load(f)
with open('results/geolife_sequence_length_results.json') as f:
    geolife = json.load(f)

# Expected values (with tolerance)
expected = {
    'diy_prev7_acc1': 56.58,
    'diy_prev7_loss': 2.874,
    'geolife_prev7_acc1': 51.40,
    'geolife_prev7_loss': 2.630,
}

# Validate
tolerance = 0.1  # percentage points

for key, expected_val in expected.items():
    if 'diy' in key:
        actual = diy['results']['7']['metrics']['acc@1'] if 'acc1' in key else diy['results']['7']['metrics']['loss']
    else:
        actual = geolife['results']['7']['metrics']['acc@1'] if 'acc1' in key else geolife['results']['7']['metrics']['loss']
    
    if abs(actual - expected_val) > tolerance:
        print(f"❌ FAILED: {key} = {actual:.2f} (expected {expected_val:.2f})")
    else:
        print(f"✅ PASSED: {key} = {actual:.2f}")
```

**Expected Output**:
```
✅ PASSED: diy_prev7_acc1 = 56.58
✅ PASSED: diy_prev7_loss = 2.87
✅ PASSED: geolife_prev7_acc1 = 51.40
✅ PASSED: geolife_prev7_loss = 2.63
```

### 7.2 Visual Inspection

Check that generated figures:
1. Have correct axis labels
2. Show two lines (DIY blue, GeoLife red)
3. Display upward trends for accuracy metrics
4. Display downward trend for loss

---

## 8. Troubleshooting

### 8.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Batch size too large | Reduce `--batch_size` to 32 or 16 |
| `FileNotFoundError: data/...` | Missing data files | Verify data paths exist |
| `FileNotFoundError: experiments/...` | Missing checkpoints | Verify checkpoint paths |
| `ModuleNotFoundError` | Missing dependency | `pip install <module>` |
| `Slow evaluation` | CPU-only mode | Check CUDA availability |

### 8.2 GPU Memory Issues

```bash
# Check GPU memory
nvidia-smi

# If memory is low, reduce batch size
python evaluate_sequence_length.py --batch_size 32
# or
python evaluate_sequence_length.py --batch_size 16
```

### 8.3 Path Issues

If paths don't match, update `EXPERIMENT_CONFIG` in `evaluate_sequence_length.py`:

```python
EXPERIMENT_CONFIG = {
    'diy': {
        'checkpoint_path': '/your/path/to/diy/best.pt',
        'config_path': '/your/path/to/config.yaml',
        'test_data_path': '/your/path/to/test.pk',
        'train_data_path': '/your/path/to/train.pk',
        'dataset_name': 'DIY',
    },
    # ... similar for geolife
}
```

---

## 9. Expected Runtime

### 9.1 Runtime Breakdown

| Component | GPU (RTX 3080) | GPU (T4) | CPU Only |
|-----------|----------------|----------|----------|
| DIY evaluation (7 configs) | ~3 min | ~5 min | ~20 min |
| GeoLife evaluation (7 configs) | ~1 min | ~2 min | ~8 min |
| Visualization generation | ~30 sec | ~30 sec | ~30 sec |
| **Total** | **~5 min** | **~8 min** | **~30 min** |

### 9.2 Memory Usage

| Phase | GPU Memory | System RAM |
|-------|------------|------------|
| Model loading | ~1 GB | ~2 GB |
| DIY evaluation | ~2 GB | ~4 GB |
| GeoLife evaluation | ~1.5 GB | ~2 GB |
| Visualization | ~100 MB | ~1 GB |

---

## 10. Reproducibility Guarantees

### 10.1 Deterministic Settings

The experiment uses fixed seeds for reproducibility:

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 10.2 What Should Be Identical

| Aspect | Reproducibility |
|--------|-----------------|
| Accuracy metrics | Identical (±0.01%) |
| Loss values | Identical (±0.001) |
| Sample counts | Identical |
| Sequence statistics | Identical |
| Figure contents | Identical |

### 10.3 What May Vary Slightly

| Aspect | Variation Source |
|--------|-----------------|
| Runtime | Hardware differences |
| Memory usage | OS/driver differences |
| Figure rendering | Font/display settings |

---

## 11. Archival Information

### 11.1 Experiment Metadata

| Property | Value |
|----------|-------|
| Experiment date | January 2, 2026 |
| Repository | next_loc_clean_v2 |
| Branch | main |
| Python version | 3.9.x |
| PyTorch version | 2.0.x |
| Random seed | 42 |

### 11.2 File Checksums (for verification)

```bash
# Generate checksums
md5sum results/*.json
md5sum results/full_results.csv

# Expected (may vary with floating point):
# diy_sequence_length_results.json: [checksum]
# geolife_sequence_length_results.json: [checksum]
# full_results.csv: [checksum]
```

---

## 12. Contact and Support

For issues reproducing this experiment:

1. **Check this document** for common troubleshooting
2. **Verify all prerequisites** are met
3. **Check file paths** match your environment
4. **Review console output** for error messages

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Created** | 2026-01-02 |
| **Word Count** | ~1,800 |
| **Status** | Final |

---

**Navigation**: [← Interpretation & Insights](./11_interpretation_and_insights.md) | [Index](./INDEX.md)

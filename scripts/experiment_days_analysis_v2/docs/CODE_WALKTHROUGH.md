# Code Walkthrough and Implementation Details

## Line-by-Line Explanation of Experiment Scripts

**Document Version:** 1.0  
**Date:** January 2, 2026

---

## Table of Contents

1. [run_days_analysis.py Walkthrough](#1-run_days_analysispy-walkthrough)
2. [generate_visualizations.py Walkthrough](#2-generate_visualizationspy-walkthrough)
3. [Key Algorithms Explained](#3-key-algorithms-explained)
4. [Error Handling and Edge Cases](#4-error-handling-and-edge-cases)
5. [Performance Considerations](#5-performance-considerations)
6. [Customization Guide](#6-customization-guide)

---

## 1. run_days_analysis.py Walkthrough

### 1.1 Imports and Setup (Lines 1-65)

```python
#!/usr/bin/env python
"""
Day-of-Week Analysis Experiment for Next Location Prediction.
...
"""
```

**Purpose**: Shebang line enables direct script execution. Docstring provides comprehensive documentation including scientific rationale, experiment design, and usage instructions.

```python
import os
import sys
import json
import pickle
import argparse
import random
from pathlib import Path
from datetime import datetime
```

**Purpose**: Standard library imports for:
- `os`: File system operations
- `sys`: System-level operations (path manipulation)
- `json`: Results serialization
- `pickle`: Loading preprocessed data
- `argparse`: Command-line argument parsing
- `random`: Seed control
- `Path`: Cross-platform path handling
- `datetime`: Timestamp generation (unused in current version)

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from tqdm import tqdm
from scipy import stats
```

**Purpose**: External library imports:
- `numpy`: Numerical operations
- `pandas`: DataFrame creation for results
- `torch`: Deep learning framework
- `Dataset, DataLoader`: PyTorch data handling
- `pad_sequence`: Variable-length sequence handling
- `f1_score`: F1 metric calculation
- `tqdm`: Progress bars
- `stats`: Statistical testing (t-test)

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

**Purpose**: Add project root to Python path for importing custom modules. This allows `from src.models...` imports regardless of where the script is run from.

```python
from src.models.proposed.pointer_v45 import PointerNetworkV45
from src.evaluation.metrics import (
    calculate_correct_total_prediction,
    get_performance_dict,
)
```

**Purpose**: Import custom model and evaluation functions from project source.

### 1.2 Configuration (Lines 64-99)

```python
SEED = 42
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
WEEKDAY_INDICES = [0, 1, 2, 3, 4]  # Monday to Friday
WEEKEND_INDICES = [5, 6]  # Saturday, Sunday
```

**Purpose**: Global constants for:
- Reproducibility seed
- Human-readable day names
- Index sets for weekday/weekend grouping

```python
CONFIGS = {
    'diy': {
        'name': 'DIY',
        'test_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_test.pk',
        'train_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_train.pk',
        'checkpoint': '/data/next_loc_clean_v2/experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt',
        'model_config': {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.2,
        }
    },
    # ... geolife config similar
}
```

**Purpose**: Dataset-specific configuration dictionary containing:
- File paths for data and checkpoints
- Model hyperparameters
- This centralized config makes it easy to add new datasets

### 1.3 Seed Function (Lines 101-109)

```python
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Line-by-line explanation**:
1. `random.seed(seed)`: Set Python's built-in random number generator
2. `np.random.seed(seed)`: Set NumPy's random generator
3. `torch.manual_seed(seed)`: Set PyTorch CPU random generator
4. `torch.cuda.manual_seed(seed)`: Set current GPU's generator
5. `torch.cuda.manual_seed_all(seed)`: Set all GPUs' generators
6. `cudnn.deterministic = True`: Force deterministic algorithms in cuDNN
7. `cudnn.benchmark = False`: Disable auto-tuning (can cause non-determinism)

### 1.4 Target Day Computation (Lines 112-128)

```python
def compute_y_weekday(sample):
    """
    Compute the weekday of the target Y from sample data.
    """
    last_weekday = sample['weekday_X'][-1]
    last_diff = sample['diff'][-1]
    return (last_weekday + last_diff) % 7
```

**Algorithm**:
1. `last_weekday`: Get the weekday of the most recent historical visit
2. `last_diff`: Get how many days until the target from that visit
3. Return: Add them and wrap around using modulo 7

**Example**:
- If last visit was Friday (4) and diff is 2:
- (4 + 2) % 7 = 6 = Sunday

### 1.5 DayFilteredDataset Class (Lines 130-199)

```python
class DayFilteredDataset(Dataset):
    def __init__(self, data_path: str, day_filter: int = None):
        # Load all data
        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)
        
        # Compute Y weekday for each sample
        self.y_weekdays = [compute_y_weekday(sample) for sample in self.all_data]
        
        # Filter data if day_filter is specified
        if day_filter is not None:
            self.data = [
                sample for sample, wd in zip(self.all_data, self.y_weekdays)
                if wd == day_filter
            ]
        else:
            self.data = self.all_data
```

**Key operations**:
1. Load pickle file containing list of sample dictionaries
2. Pre-compute target weekday for ALL samples
3. If filtering, keep only samples where target falls on specified day
4. `_compute_statistics()` calculates vocabulary sizes

```python
def __getitem__(self, idx):
    sample = self.data[idx]
    
    return_dict = {
        'user': torch.tensor(sample['user_X'][0], dtype=torch.long),
        'weekday': torch.tensor(sample['weekday_X'], dtype=torch.long),
        'time': torch.tensor(sample['start_min_X'] // 15, dtype=torch.long),
        'duration': torch.tensor(sample['dur_X'] // 30, dtype=torch.long),
        'diff': torch.tensor(sample['diff'], dtype=torch.long),
    }
    
    x = torch.tensor(sample['X'], dtype=torch.long)
    y = torch.tensor(sample['Y'], dtype=torch.long)
    
    return x, y, return_dict
```

**Feature engineering**:
- `time`: Converted from minutes to 15-minute buckets (0-95)
- `duration`: Converted from minutes to 30-minute buckets
- `user`: Scalar user ID
- `weekday`, `diff`: Direct integer encoding

### 1.6 Collate Function (Lines 202-225)

```python
def collate_fn(batch):
    """Collate function for variable length sequences."""
    x_batch, y_batch = [], []
    x_dict_batch = {'len': []}
    
    for key in batch[0][-1]:
        x_dict_batch[key] = []
    
    for x, y, return_dict in batch:
        x_batch.append(x)
        y_batch.append(y)
        x_dict_batch['len'].append(len(x))
        for key in return_dict:
            x_dict_batch[key].append(return_dict[key])
    
    x_batch = pad_sequence(x_batch, batch_first=False, padding_value=0)
    y_batch = torch.stack(y_batch)
    # ... pad other sequences
```

**Purpose**: Handle variable-length sequences in a batch:
1. Collect all samples' data
2. Pad sequences to the maximum length in the batch
3. Use `padding_value=0` (which is the padding token in vocabulary)
4. Stack scalar values

### 1.7 Model Loading (Lines 228-284)

```python
def load_model(dataset_key: str, device: torch.device):
    config = CONFIGS[dataset_key]
    
    # Load checkpoint first to get the correct max_seq_len
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    
    # Get max_seq_len from position_bias shape in checkpoint
    max_seq_len = checkpoint['model_state_dict']['position_bias'].shape[0]
```

**Critical step**: The model's `max_seq_len` must match the checkpoint. We infer it from the saved `position_bias` tensor's shape.

```python
    # Get number of locations and users from training data
    with open(config['train_path'], 'rb') as f:
        train_data = pickle.load(f)
    
    all_locs = set()
    all_users = set()
    for sample in train_data:
        all_locs.update(sample['X'].tolist())
        all_locs.add(sample['Y'])
        all_users.add(sample['user_X'][0])
    
    num_locations = max(all_locs) + 1
    num_users = max(all_users) + 1
```

**Purpose**: Determine vocabulary sizes by scanning training data. The +1 accounts for 0-indexing and the padding token.

```python
    model = PointerNetworkV45(
        num_locations=num_locations,
        num_users=num_users,
        d_model=model_cfg['d_model'],
        # ... other params
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
```

**Steps**:
1. Instantiate model with correct architecture
2. Load saved weights
3. Move to GPU
4. Set to evaluation mode (disables dropout)

### 1.8 Evaluation Function (Lines 287-360)

```python
@torch.no_grad()
def evaluate_on_day(model, dataset, device, batch_size=64):
    if len(dataset) == 0:
        return None
```

**Decorator**: `@torch.no_grad()` disables gradient computation, saving memory and computation.

```python
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
```

**DataLoader settings**:
- `shuffle=False`: Deterministic order for reproducibility
- `num_workers=0`: No multiprocessing (avoids issues)
- `pin_memory=True`: Faster GPU transfer

```python
    for x, y, x_dict in tqdm(dataloader, desc="Evaluating", leave=False):
        x = x.to(device)
        y = y.to(device)
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        
        with torch.cuda.amp.autocast():
            logits = model(x, x_dict)
            loss = criterion(logits, y)
```

**Key points**:
- Move all data to GPU
- Use mixed precision (`autocast`) for speed
- Get model predictions (logits)
- Compute cross-entropy loss

```python
        results, true_y, pred_y = calculate_correct_total_prediction(logits, y)
        all_results.append(results)
```

**Metrics**: Call the imported function to compute Acc@K, MRR, NDCG.

### 1.9 Main Analysis Function (Lines 363-506)

```python
def run_day_analysis(dataset_key: str, output_dir: str):
    # ... setup ...
    
    # Evaluate each day
    for day_idx in range(7):
        day_name = DAY_NAMES[day_idx]
        dataset = DayFilteredDataset(config['test_path'], day_filter=day_idx)
        metrics = evaluate_on_day(model, dataset, device)
        results[day_name] = { ... }
```

**Main loop**: Iterate through 7 days, filter dataset, evaluate, store results.

```python
    # Compute weekday vs weekend aggregates
    def weighted_avg(metrics_list, key):
        total_samples = sum(m['samples'] for m in metrics_list)
        return sum(m[key] * m['samples'] for m in metrics_list) / total_samples
```

**Weighted averaging**: Metrics are averaged weighted by sample count per day. This prevents days with fewer samples from disproportionately affecting the average.

```python
    # Statistical tests
    if len(weekday_acc1) > 1 and len(weekend_acc1) > 1:
        t_stat, p_value = stats.ttest_ind(weekday_acc1, weekend_acc1)
```

**T-test**: Independent samples t-test comparing weekday and weekend accuracy values.

---

## 2. generate_visualizations.py Walkthrough

### 2.1 Style Configuration (Lines 41-94)

```python
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 12,
    # ... many more settings
})
```

**Purpose**: Set matplotlib's global style to match scientific publication standards (Nature-style).

**Key settings**:
- `axes.spines.top/right/bottom/left = True`: Show all four axis borders
- `xtick.direction = 'in'`: Tick marks point inward
- `axes.grid = False`: No grid lines
- `font.family = 'serif'`: Use serif fonts (Times)

### 2.2 Visualization Functions

Each plotting function follows a similar pattern:

```python
def plot_accuracy_by_day(results: dict, dataset_name: str, output_dir: str):
    # 1. Create figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # 2. Extract data
    values = [results[day][metric] for day in days]
    
    # 3. Create bars/lines/etc.
    bars = ax.bar(x, values, width, color='white', edgecolor=colors, ...)
    
    # 4. Add annotations
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.1f}', xy=(...), ...)
    
    # 5. Style adjustments
    setup_classic_axes(ax)
    
    # 6. Save in multiple formats
    for fmt in ['pdf', 'png', 'svg']:
        plt.savefig(output_path, format=fmt, dpi=300, ...)
    
    plt.close()
```

### 2.3 Combined Figure (Lines 501-723)

```python
def create_combined_figure(diy_results, geolife_results, output_dir):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
```

**GridSpec**: Creates a 3×3 grid of subplots with specified spacing.

```python
    # Panel (a)
    ax1 = fig.add_subplot(gs[0, 0])
    # ... plotting code ...
    
    # Panel labels
    for label, ax in zip(panel_labels, axes):
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.02, bbox.y1 + 0.01, label, ...)
```

**Panel labeling**: Labels like "(a)" are added outside the axis area using figure-level text with position calculated from axis bounds.

---

## 3. Key Algorithms Explained

### 3.1 Top-K Accuracy Calculation

```python
# In src/evaluation/metrics.py
for k in [1, 3, 5, 10]:
    prediction = torch.topk(logits, k=k, dim=-1).indices
    top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum()
```

**Algorithm**:
1. `torch.topk`: Get indices of K highest logit values
2. `torch.eq`: Compare true labels with all K predictions
3. `.any(dim=1)`: True if any of the K matches
4. `.sum()`: Count correct samples

### 3.2 MRR Calculation

```python
def get_mrr(prediction, targets):
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float()
    rranks = torch.reciprocal(ranks)
    return torch.sum(rranks)
```

**Algorithm**:
1. Sort predictions descending → get rank order
2. Find where target appears in sorted list
3. Convert to 1-indexed rank
4. Take reciprocal (1/rank)
5. Sum across all samples

### 3.3 Weighted Average

```python
def weighted_avg(metrics_list, key):
    total_samples = sum(m['samples'] for m in metrics_list)
    return sum(m[key] * m['samples'] for m in metrics_list) / total_samples
```

**Formula**:
$$\bar{x}_w = \frac{\sum_{i} n_i \cdot x_i}{\sum_{i} n_i}$$

This gives more weight to days with more samples.

---

## 4. Error Handling and Edge Cases

### 4.1 Empty Dataset Handling

```python
def evaluate_on_day(model, dataset, device, batch_size=64):
    if len(dataset) == 0:
        return None
```

**Purpose**: If a day has no samples (rare but possible), return None instead of crashing.

### 4.2 Missing Results Handling

```python
for results, name in [(diy_results, 'DIY'), (geolife_results, 'GeoLife')]:
    if results is None:
        continue
```

**Purpose**: Skip datasets that failed to load or process.

### 4.3 Statistical Test Validation

```python
if len(weekday_acc1) > 1 and len(weekend_acc1) > 1:
    t_stat, p_value = stats.ttest_ind(...)
```

**Purpose**: T-test requires at least 2 samples per group. This check prevents errors.

---

## 5. Performance Considerations

### 5.1 Memory Efficiency

- **Mixed precision**: `torch.cuda.amp.autocast()` uses FP16 where safe
- **No gradient storage**: `@torch.no_grad()` decorator
- **Batch processing**: Process data in chunks of 64
- **Close figures**: `plt.close()` frees memory after saving

### 5.2 Speed Optimizations

- **Pin memory**: `pin_memory=True` in DataLoader for faster GPU transfer
- **Vectorized operations**: NumPy and PyTorch vectorization
- **Efficient metrics**: Custom functions optimized for batch processing

### 5.3 Typical Runtime

| Operation | DIY (~12K samples) | GeoLife (~3.5K samples) |
|-----------|-------------------|-------------------------|
| Model loading | ~5 seconds | ~5 seconds |
| Per-day evaluation | ~15 seconds | ~5 seconds |
| Total evaluation | ~2 minutes | ~40 seconds |
| Visualization | ~20 seconds | ~20 seconds |

---

## 6. Customization Guide

### 6.1 Adding a New Dataset

```python
CONFIGS['new_dataset'] = {
    'name': 'NewDataset',
    'test_path': '/path/to/test.pk',
    'train_path': '/path/to/train.pk',
    'checkpoint': '/path/to/checkpoint.pt',
    'model_config': {
        'd_model': 128,
        'nhead': 8,
        # ... other hyperparameters
    }
}
```

### 6.2 Changing Evaluation Metrics

To add new metrics, modify `evaluate_on_day()`:

```python
# Add after existing metric calculations
from sklearn.metrics import precision_score, recall_score

all_precision.append(precision_score(true_y.cpu(), pred_y.cpu(), average='macro'))
# ... aggregate and include in results
```

### 6.3 Customizing Visualizations

To change colors:

```python
COLORS = {
    'weekday': '#2ecc71',  # Green
    'weekend': '#e74c3c',  # Red
    # ... add more
}
```

To change figure sizes:

```python
fig, ax = plt.subplots(figsize=(12, 8))  # width, height in inches
```

### 6.4 Running Subset of Days

```python
# Modify the loop in run_day_analysis
for day_idx in [0, 4, 5, 6]:  # Only Mon, Fri, Sat, Sun
    ...
```

---

*End of Code Walkthrough*

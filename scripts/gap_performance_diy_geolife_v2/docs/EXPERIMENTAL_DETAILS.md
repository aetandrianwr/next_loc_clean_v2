# Experimental Details: Complete Protocol

This document provides exhaustive details about the experimental setup, data processing, and analysis protocols.

---

## 1. Data Preparation

### 1.1 Source Data Files

**DIY Dataset**:
```
Path: data/diy_eps50/processed/diy_eps50_prev7_test.pk
Format: Python pickle (protocol 4)
Size: ~5.2 MB
Samples: 12,368 test samples
```

**GeoLife Dataset**:
```
Path: data/geolife_eps20/processed/geolife_eps20_prev7_test.pk
Format: Python pickle (protocol 4)
Size: ~1.4 MB
Samples: 3,502 test samples
```

### 1.2 Data Schema

Each sample is a Python dictionary with the following structure:

```python
sample = {
    'X': np.array,           # Input location sequence
    'Y': int,                # Target next location
    'user_X': np.array,      # User ID for each position
    'weekday_X': np.array,   # Day of week (0=Monday, 6=Sunday)
    'start_min_X': np.array, # Start time in minutes from midnight
    'dur_X': np.array,       # Visit duration in minutes
    'diff': np.array,        # Days ago from reference date
}
```

**Example sample**:
```python
{
    'X': array([42, 17, 42, 8, 42, 17, 42]),
    'Y': 42,
    'user_X': array([5, 5, 5, 5, 5, 5, 5]),
    'weekday_X': array([0, 0, 1, 1, 2, 2, 3]),
    'start_min_X': array([480, 720, 480, 720, 480, 720, 480]),
    'dur_X': array([120, 60, 120, 60, 120, 60, 120]),
    'diff': array([6, 6, 5, 5, 4, 4, 3]),
}
```

**Interpretation of example**:
- User visited locations: 42→17→42→8→42→17→42
- Next location to predict: 42
- User ID: 5 (same throughout sequence)
- Days: Monday, Monday, Tuesday, Tuesday, Wednesday, Wednesday, Thursday
- Times: 8:00 AM, 12:00 PM pattern
- Durations: 2 hours, 1 hour alternating
- Recency: 6 days ago to 3 days ago

### 1.3 Preprocessing Applied

**Spatial Clustering**:
- DIY: DBSCAN with epsilon=50 meters
- GeoLife: DBSCAN with epsilon=20 meters
- Locations are cluster IDs, not raw coordinates

**Temporal Segmentation**:
- prev7: Uses 7 days of history as input
- Each sample represents one prediction task
- Multiple samples per user (sliding window)

**Filtering**:
- Minimum sequence length: 2
- Maximum sequence length: 48 (DIY), 40 (GeoLife)
- Users with insufficient data excluded

---

## 2. Model Checkpoints

### 2.1 DIY Model

```
Path: experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt
Config: scripts/sci_hyperparam_tuning/configs/pointer_v45_diy_trial09.yaml
```

**Training details**:
- Epochs: ~100 with early stopping
- Batch size: 64
- Learning rate: 1e-4 with scheduler
- Best validation accuracy: ~57%

**Architecture (from checkpoint)**:
- d_model: 128
- num_layers: 3
- nhead: 4
- dim_feedforward: 256
- dropout: 0.15
- max_seq_len: 48

### 2.2 GeoLife Model

```
Path: experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt
Config: scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml
```

**Training details**:
- Epochs: ~80 with early stopping
- Batch size: 32
- Learning rate: 1e-4 with scheduler
- Best validation accuracy: ~54%

**Architecture (from checkpoint)**:
- d_model: 64
- num_layers: 2
- nhead: 4
- dim_feedforward: 192
- dropout: 0.15
- max_seq_len: 40

---

## 3. Analysis Protocols

### 3.1 Mobility Pattern Analysis Protocol

**Script**: `analyze_mobility_patterns.py`

**Execution steps**:

1. **Load datasets**
   ```python
   with open(data_path, 'rb') as f:
       data = pickle.load(f)
   ```

2. **For each sample, compute**:
   - Is target in history? (`y in x`)
   - Position of target from end
   - Frequency of target in history
   - Unique location ratio
   - Sequence entropy
   - Consecutive repeat rate
   - Most frequent location statistics

3. **Aggregate statistics**:
   - Mean, std, median, min, max for each metric
   - Per-user aggregation where applicable

4. **Statistical tests**:
   - Chi-square for categorical comparisons
   - Mann-Whitney U for distribution comparisons
   - Cohen's d for effect sizes

5. **Generate visualizations**:
   - Bar charts for rate comparisons
   - Histograms for distributions
   - Box plots for spread visualization

6. **Save results**:
   - JSON for numerical results
   - CSV/LaTeX for tables
   - PNG/PDF for figures

### 3.2 Model Behavior Analysis Protocol

**Script**: `analyze_model_pointer.py`

**Execution steps**:

1. **Load model**:
   ```python
   checkpoint = torch.load(checkpoint_path)
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   ```

2. **Create dataloader**:
   ```python
   dataset = NextLocationDataset(data_path)
   dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
   ```

3. **Inference loop**:
   ```python
   with torch.no_grad():
       for x, y, x_dict, raw in dataloader:
           log_probs, analysis = model.forward_with_analysis(x, x_dict)
           # Extract: gate, ptr_probs, gen_probs, predictions
   ```

4. **Compute metrics**:
   - Gate value statistics
   - Accuracy breakdown by condition
   - Probability mass analysis
   - Conditional comparisons

5. **Generate visualizations**:
   - Gate distribution histograms
   - Probability scatter plots
   - Accuracy bar charts

### 3.3 Recency Pattern Analysis Protocol

**Script**: `analyze_recency_patterns.py`

**Execution steps**:

1. **For each sample**:
   ```python
   positions = np.where(x == y)[0]
   if len(positions) > 0:
       pos_from_end = seq_len - positions
       most_recent = min(pos_from_end)
   ```

2. **Compute recency metrics**:
   - Target position distribution
   - Target = last rate
   - Target in top-K recent rates
   - Return patterns (A→B→A)

3. **Compute predictability scores**:
   ```python
   recency_score = 1 / position_from_end
   frequency_score = count / seq_len
   predictability = recency * frequency
   ```

4. **Generate visualizations**:
   - Position distribution histograms
   - Cumulative distribution plots
   - Correlation scatter plots

---

## 4. Computation Details

### 4.1 Target-in-History Computation

```python
def compute_target_in_history(sample):
    x = sample['X']  # Input sequence
    y = sample['Y']  # Target
    
    # Check presence
    is_in_history = y in x  # O(n) operation
    
    if is_in_history:
        # Find all positions
        positions = np.where(x == y)[0]  # Returns indices
        
        # Most recent position (position from end)
        seq_len = len(x)
        pos_from_end = seq_len - positions[-1]  # positions[-1] is rightmost
        
        # Frequency
        frequency = len(positions)
        
        return True, pos_from_end, frequency
    else:
        return False, None, 0
```

**Time complexity**: O(n) per sample, O(N*n) total

### 4.2 Unique Ratio Computation

```python
def compute_unique_ratio(sample):
    x = sample['X']
    
    unique_locs = np.unique(x)
    n_unique = len(unique_locs)
    seq_len = len(x)
    
    unique_ratio = n_unique / seq_len
    repetition_rate = 1 - unique_ratio
    
    return unique_ratio, repetition_rate
```

**Time complexity**: O(n log n) per sample (sorting in unique)

### 4.3 Entropy Computation

```python
def compute_entropy(sample):
    x = sample['X']
    
    # Count occurrences
    counter = Counter(x)  # O(n)
    total = len(x)
    
    # Compute entropy
    entropy = 0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    # Normalized entropy
    n_unique = len(counter)
    if n_unique > 1:
        max_entropy = np.log2(n_unique)
        normalized = entropy / max_entropy
    else:
        normalized = 0
    
    return entropy, normalized
```

**Time complexity**: O(n) per sample

### 4.4 Consecutive Repeat Computation

```python
def compute_consecutive_repeats(sample):
    x = sample['X']
    y = sample['Y']
    
    if len(x) < 2:
        return 0, False
    
    # Count A→A patterns
    n_consecutive = sum(1 for i in range(len(x)-1) if x[i] == x[i+1])
    rate = n_consecutive / (len(x) - 1)
    
    # Target equals last
    target_equals_last = (y == x[-1])
    
    return rate, target_equals_last
```

**Time complexity**: O(n) per sample

---

## 5. Statistical Test Details

### 5.1 Chi-Square Test Implementation

```python
from scipy import stats

def chi_square_test(diy_data, geolife_data):
    # Count target in history for each dataset
    diy_in = sum(1 for s in diy_data if s['Y'] in s['X'])
    diy_not = len(diy_data) - diy_in
    
    geo_in = sum(1 for s in geolife_data if s['Y'] in s['X'])
    geo_not = len(geolife_data) - geo_in
    
    # Contingency table
    table = [[diy_in, diy_not], [geo_in, geo_not]]
    
    # Perform test
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    
    return chi2, p_value
```

**Our results**:
- χ² = 0.174
- p-value = 0.676
- Conclusion: Not significant (p > 0.05)

### 5.2 Mann-Whitney U Test Implementation

```python
def mann_whitney_test(diy_ratios, geolife_ratios):
    # Perform two-sided test
    u_stat, p_value = stats.mannwhitneyu(
        diy_ratios, 
        geolife_ratios,
        alternative='two-sided'
    )
    
    return u_stat, p_value
```

**Our results**:
- U = 19,139,076
- p-value = 7.03 × 10⁻²⁶
- Conclusion: Highly significant (p < 0.001)

### 5.3 Effect Size Calculation

```python
def cohens_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1), np.std(group2)
    
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    d = (mean1 - mean2) / pooled_std
    
    return d
```

**Our results**:
- Cohen's d = -0.160
- Interpretation: Small effect size

---

## 6. Visualization Parameters

### 6.1 Figure Sizes

| Figure Type | Size (inches) |
|-------------|---------------|
| Single panel | 8 × 6 |
| Two panel | 12 × 5 |
| Four panel | 12 × 10 |
| Six panel | 14 × 8 |

### 6.2 Color Palette

```python
COLORS = {
    'diy': '#1f77b4',      # Blue
    'geolife': '#d62728',  # Red
    'black': '#000000',
    'gray': '#7f7f7f',
}
```

### 6.3 Hatching Patterns

```python
HATCHES = {
    'diy': '///',     # Diagonal lines
    'geolife': '...'  # Dots
}
```

### 6.4 Output Formats

- PNG: 300 DPI, white background, tight bounding box
- PDF: Vector format, same styling

---

## 7. Reproducibility Checklist

### 7.1 Environment

- [ ] Python 3.8+
- [ ] PyTorch 1.9+
- [ ] NumPy 1.19+
- [ ] Pandas 1.3+
- [ ] Matplotlib 3.4+
- [ ] SciPy 1.7+
- [ ] PyYAML 5.4+

### 7.2 Files Required

- [ ] `data/diy_eps50/processed/diy_eps50_prev7_test.pk`
- [ ] `data/geolife_eps20/processed/geolife_eps20_prev7_test.pk`
- [ ] `experiments/diy_pointer_v45_*/checkpoints/best.pt`
- [ ] `experiments/geolife_pointer_v45_*/checkpoints/best.pt`
- [ ] Config YAML files

### 7.3 Expected Runtime

| Script | Approximate Time |
|--------|------------------|
| analyze_mobility_patterns.py | ~30 seconds |
| analyze_model_pointer.py | ~2 minutes (GPU), ~10 minutes (CPU) |
| analyze_recency_patterns.py | ~20 seconds |
| Total (run_all_experiments.py) | ~3-12 minutes |

### 7.4 Expected Outputs

- [ ] `results/analysis_results.json`
- [ ] `results/model_analysis_results.json`
- [ ] `results/recency_analysis_results.json`
- [ ] `results/tables/*.csv` (3 files)
- [ ] `results/tables/*.tex` (2 files)
- [ ] `results/figures/*.png` (10 files)
- [ ] `results/figures/*.pdf` (10 files)

---

*Experimental Details Version: 1.0*

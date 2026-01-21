# Technical Reference: Gap Performance Analysis Scripts

This document provides detailed technical documentation for all scripts in the `gap_performance_diy_geolife_v2` analysis framework.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Script: analyze_mobility_patterns.py](#2-script-analyze_mobility_patternspy)
3. [Script: analyze_model_pointer.py](#3-script-analyze_model_pointerpy)
4. [Script: analyze_recency_patterns.py](#4-script-analyze_recency_patternspy)
5. [Script: run_all_experiments.py](#5-script-run_all_experimentspy)
6. [Data Structures](#6-data-structures)
7. [Visualization Style Guide](#7-visualization-style-guide)
8. [Dependencies](#8-dependencies)

---

## 1. Project Structure

```
gap_performance_diy_geolife_v2/
├── analyze_mobility_patterns.py    # Mobility characteristic analysis
├── analyze_model_pointer.py        # Trained model behavior analysis
├── analyze_recency_patterns.py     # Recency effect analysis
├── run_all_experiments.py          # Master orchestration script
├── docs/                           # Documentation
│   ├── README.md                   # Main comprehensive documentation
│   ├── TECHNICAL_REFERENCE.md      # This file
│   ├── FIGURES_GALLERY.md          # Figure interpretations
│   └── QUICK_START.md              # Quick start guide
└── results/                        # Generated outputs
    ├── analysis_results.json       # Mobility analysis JSON
    ├── model_analysis_results.json # Model analysis JSON
    ├── recency_analysis_results.json # Recency analysis JSON
    ├── figures/                    # Generated plots (PNG and PDF)
    └── tables/                     # Generated tables (CSV and LaTeX)
```

---

## 2. Script: `analyze_mobility_patterns.py`

### 2.1 Purpose

Analyzes fundamental mobility characteristics to explain why pointer mechanism has different impacts on DIY (8.3% drop) vs GeoLife (46.7% drop).

### 2.2 Main Class: `MobilityPatternAnalyzer`

```python
class MobilityPatternAnalyzer:
    """Analyzes mobility patterns to explain pointer mechanism performance gap."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Path to save results (creates figures/ and tables/ subdirs)
        """
```

### 2.3 Core Methods

#### `load_dataset(data_path: str, name: str) -> list`

```python
def load_dataset(self, data_path: str, name: str) -> list:
    """
    Load dataset from pickle file.
    
    Args:
        data_path: Path to .pk file
        name: Human-readable dataset name (for logging)
    
    Returns:
        list: List of sample dictionaries
    """
```

**Example usage**:
```python
diy_data = analyzer.load_dataset('data/diy_eps50/processed/diy_eps50_prev7_test.pk', 'DIY')
# Returns: [{'X': array([1,2,3]), 'Y': 2, 'user_X': array([1,1,1]), ...}, ...]
```

---

#### `analyze_target_in_history(data: list, name: str) -> tuple`

```python
def analyze_target_in_history(self, data: list, name: str) -> dict:
    """
    Experiment 1: Analyze how often the target location appears in input history.
    
    This is the most direct measure of whether the copy mechanism can help.
    If target is in history, pointer can directly copy it.
    
    Args:
        data: List of samples
        name: Dataset name
        
    Returns:
        tuple: (results_dict, target_in_history_list, target_positions_list)
    """
```

**Metrics computed**:
- `total_samples`: Number of samples analyzed
- `target_in_history_count`: Absolute count where target in history
- `target_in_history_rate`: Percentage (0-100)
- `avg_position_from_end`: Average position of target from sequence end
- `std_position_from_end`: Standard deviation of position
- `avg_target_frequency`: Average number of times target appears in history
- `std_target_frequency`: Standard deviation of frequency

**Algorithm**:
```python
for sample in data:
    x = sample['X']  # Input sequence
    y = sample['Y']  # Target
    
    is_in_history = y in x  # Check if target appears in input
    
    if is_in_history:
        positions = np.where(x == y)[0]  # Find all occurrences
        pos_from_end = len(x) - positions[-1]  # Most recent occurrence
        target_position_from_end.append(pos_from_end)
        target_frequency_in_history.append(len(positions))
```

---

#### `analyze_unique_location_ratio(data: list, name: str) -> tuple`

```python
def analyze_unique_location_ratio(self, data: list, name: str) -> dict:
    """
    Experiment 2: Analyze the ratio of unique locations to total sequence length.
    
    Lower ratio = more repetitive patterns
    Higher ratio = more diverse patterns
    
    Args:
        data: List of samples
        name: Dataset name
    
    Returns:
        tuple: (results_dict, unique_ratios_list)
    """
```

**Metrics computed**:
- `avg_unique_ratio`: Mean unique ratio (0-1)
- `std_unique_ratio`: Standard deviation
- `median_unique_ratio`: Median value
- `min_unique_ratio`: Minimum (most repetitive)
- `max_unique_ratio`: Maximum (most diverse)
- `avg_seq_length`: Average sequence length
- `avg_unique_count`: Average number of unique locations
- `repetition_rate`: 1 - avg_unique_ratio (proportion of repeated visits)

**Formula**:
```
unique_ratio = |unique(X)| / |X|
repetition_rate = 1 - unique_ratio
```

---

#### `analyze_location_entropy(data: list, name: str) -> tuple`

```python
def analyze_location_entropy(self, data: list, name: str) -> dict:
    """
    Experiment 3: Analyze entropy of location distributions.
    
    Lower entropy = more predictable/repetitive
    Higher entropy = more random/diverse
    
    Args:
        data: List of samples
        name: Dataset name
    
    Returns:
        tuple: (results_dict, sequence_entropies, user_entropies, normalized_entropies)
    """
```

**Entropy calculation**:
```python
def calculate_entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0
    probs = np.array([c / total for c in counts if c > 0])
    return -np.sum(probs * np.log2(probs))  # Shannon entropy
```

**Normalized entropy**:
```python
normalized_entropy = entropy / log2(n_unique)  # Range: [0, 1]
```

**Metrics computed**:
- `avg_sequence_entropy`: Mean entropy per sequence (bits)
- `std_sequence_entropy`: Standard deviation
- `avg_user_entropy`: Mean entropy per user (aggregated)
- `std_user_entropy`: Standard deviation
- `avg_normalized_entropy`: Entropy / max possible entropy
- `num_users`: Number of unique users
- `avg_user_unique_locations`: Average unique locations per user

---

#### `analyze_consecutive_repeats(data: list, name: str) -> tuple`

```python
def analyze_consecutive_repeats(self, data: list, name: str) -> dict:
    """
    Experiment 4: Analyze consecutive location repeats (A->A patterns).
    
    Higher consecutive repeat rate indicates stronger repetitive patterns.
    
    Returns:
        tuple: (results_dict, consecutive_repeat_rates, target_equals_last_list)
    """
```

**Algorithm**:
```python
for sample in data:
    x = sample['X']
    # Count A->A patterns
    n_consecutive = sum(1 for i in range(len(x)-1) if x[i] == x[i+1])
    rate = n_consecutive / (len(x) - 1)
    
    # Check if target equals last position
    target_equals_last.append(sample['Y'] == x[-1])
```

**Metrics computed**:
- `avg_consecutive_repeat_rate`: Mean A→A rate
- `std_consecutive_repeat_rate`: Standard deviation
- `pct_with_any_consecutive`: % sequences with at least one A→A
- `target_equals_last_rate`: % where target = last position (critical metric!)

---

#### `analyze_most_frequent_location(data: list, name: str) -> tuple`

```python
def analyze_most_frequent_location(self, data: list, name: str) -> dict:
    """
    Experiment 5: Analyze the dominance of most frequent locations.
    
    Higher concentration = more predictable patterns.
    
    Returns:
        tuple: (results_dict, most_freq_ratios, target_is_most_freq, target_is_top3)
    """
```

**Metrics computed**:
- `avg_most_freq_ratio`: Avg visits to #1 location / total visits
- `avg_top3_ratio`: Avg visits to top-3 locations / total visits
- `target_is_most_freq_rate`: % where target is the most frequent location
- `target_is_top3_rate`: % where target is among top-3 frequent

---

#### `run_statistical_tests(diy_data: list, geolife_data: list) -> dict`

```python
def run_statistical_tests(self, diy_data: list, geolife_data: list) -> dict:
    """
    Run statistical tests to verify significance of differences.
    
    Tests performed:
    1. Chi-square test for target-in-history (categorical)
    2. Mann-Whitney U test for unique ratios (non-parametric)
    3. Cohen's d effect size for unique ratios
    
    Returns:
        dict: Test statistics and p-values
    """
```

**Statistical tests**:
```python
# Chi-square for target-in-history
contingency_table = [[diy_in, diy_not_in], [geolife_in, geolife_not_in]]
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)

# Mann-Whitney U for unique ratios
u_stat, p_mannwhitney = stats.mannwhitneyu(diy_unique_ratios, geolife_unique_ratios)

# Cohen's d effect size
pooled_std = np.sqrt((np.std(diy)**2 + np.std(geolife)**2) / 2)
cohens_d = (np.mean(diy) - np.mean(geolife)) / pooled_std
```

---

### 2.4 Visualization Methods

#### `plot_target_in_history_comparison(diy_rate, geolife_rate)`

Creates a bar chart comparing target-in-history rates.

**Output**: `target_in_history_comparison.png/pdf`

---

#### `plot_unique_ratio_distribution(diy_ratios, geolife_ratios)`

Creates two-panel figure: histogram + box plot of unique ratios.

**Output**: `unique_ratio_distribution.png/pdf`

---

#### `plot_entropy_comparison(...)`

Creates two-panel box plot comparing sequence and normalized entropy.

**Output**: `entropy_comparison.png/pdf`

---

#### `plot_comprehensive_comparison(diy_results, geolife_results)`

Creates grouped bar chart with 6 key metrics side-by-side.

**Output**: `comprehensive_comparison.png/pdf`

---

#### `plot_pointer_benefit_analysis(diy_results, geolife_results)`

Creates four-panel analysis showing why pointer benefits GeoLife more.

**Output**: `pointer_benefit_analysis.png/pdf`

---

### 2.5 Output Files

**JSON**: `analysis_results.json`
```json
{
  "diy": {
    "target_in_history": {...},
    "unique_ratio": {...},
    "entropy": {...},
    "consecutive": {...},
    "most_freq": {...}
  },
  "geolife": {...},
  "statistical_tests": {...}
}
```

**CSV**: `metric_comparison.csv`
**LaTeX**: `metric_comparison.tex`

---

## 3. Script: `analyze_model_pointer.py`

### 3.1 Purpose

Analyzes trained PointerGeneratorTransformer models to understand how they use the pointer mechanism on each dataset.

### 3.2 Custom Dataset Class

```python
class NextLocationDataset(Dataset):
    """Dataset for next location prediction with analysis support."""
    
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self._compute_statistics()
    
    def __getitem__(self, idx):
        # Returns: x, y, x_dict, raw_sample
        # x: Location sequence tensor
        # y: Target tensor
        # x_dict: Temporal features dictionary
        # raw_sample: Original sample for analysis
```

### 3.3 Extended Model Class

```python
class PointerGeneratorTransformerWithAnalysis(PointerGeneratorTransformer):
    """Extended model that returns analysis information."""
    
    def forward_with_analysis(self, x, x_dict):
        """
        Forward pass with intermediate values for analysis.
        
        Returns:
            tuple: (log_probs, analysis_dict)
            
        analysis_dict contains:
        - gate: Gate values [batch_size]
        - ptr_probs: Pointer attention [batch_size, seq_len]
        - ptr_dist: Pointer distribution [batch_size, num_locations]
        - gen_probs: Generation distribution [batch_size, num_locations]
        - final_probs: Combined distribution [batch_size, num_locations]
        - input_locs: Input locations [batch_size, seq_len]
        - mask: Padding mask [batch_size, seq_len]
        - seq_lens: Sequence lengths [batch_size]
        """
```

### 3.4 Main Analysis Class

```python
class ModelPointerAnalyzer:
    """Analyzes trained models' pointer behavior."""
    
    def __init__(self, output_dir: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

#### `analyze_model_behavior(model, dataloader, name) -> dict`

```python
def analyze_model_behavior(self, model, dataloader, name):
    """
    Analyze model's pointer behavior on dataset.
    
    Returns comprehensive statistics about:
    - Gate values (pointer vs generation preference)
    - Pointer attention patterns
    - Prediction accuracy breakdown by gate value
    
    Returns:
        dict: Analysis results with metrics and raw arrays
    """
```

**Metrics computed**:
- `avg_gate`: Average gate value (1=pointer, 0=generation)
- `std_gate`: Standard deviation of gate
- `median_gate`, `min_gate`, `max_gate`: Distribution stats
- `overall_accuracy`: Total accuracy (%)
- `acc_target_in_history`: Accuracy when target in history (%)
- `acc_target_not_in_history`: Accuracy when target NOT in history (%)
- `pct_target_in_history`: % of samples with target in history
- `avg_ptr_prob_on_target`: Mean pointer probability on target
- `avg_gen_prob_on_target`: Mean generation probability on target
- `avg_ptr_prob_when_target_in_hist`: Pointer prob (filtered)
- `avg_gen_prob_when_target_in_hist`: Generation prob (filtered)
- `avg_gate_when_correct`: Gate value when prediction is correct
- `avg_gate_when_wrong`: Gate value when prediction is wrong
- `avg_gate_target_in_hist`: Gate when target in history
- `avg_gate_target_not_in_hist`: Gate when target not in history

**Raw arrays** (for visualization):
- `raw_gates`: All gate values
- `raw_correct`: Boolean correctness
- `raw_target_in_history`: Boolean target in history
- `raw_ptr_on_target`: Pointer probabilities
- `raw_gen_on_target`: Generation probabilities

---

### 3.5 Model Loading

```python
def load_model(config_path, checkpoint_path, dataset_info, device, max_seq_len_override=None):
    """
    Load trained model from checkpoint.
    
    Args:
        config_path: Path to YAML config file
        checkpoint_path: Path to .pt checkpoint
        dataset_info: Dict with num_locations, num_users, max_seq_len
        device: torch.device
    
    Returns:
        PointerGeneratorTransformerWithAnalysis: Loaded model in eval mode
    """
```

**Checkpoint structure**:
```python
checkpoint = {
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'epoch': int,
    'loss': float,
    # etc.
}
```

---

### 3.6 Visualization Methods

#### `plot_gate_comparison(diy_results, geolife_results)`

Three-panel figure:
1. Gate distribution histogram
2. Gate by correctness (bar chart)
3. Gate by target location status (bar chart)

**Output**: `gate_comparison.png/pdf`

---

#### `plot_probability_analysis(diy_results, geolife_results)`

Two-panel figure:
1. Pointer vs Generation probability on target (when target in history)
2. Accuracy breakdown (overall, in history, not in history)

**Output**: `probability_analysis.png/pdf`

---

#### `plot_pointer_contribution_breakdown(diy_results, geolife_results)`

Four-panel comprehensive figure:
1. Gate histogram with density and means
2. Pointer vs Generation scatter plot
3. Pointer contribution metrics bar chart
4. Summary text box with key findings

**Output**: `pointer_contribution_breakdown.png/pdf`

---

### 3.7 Output Files

**JSON**: `model_analysis_results.json`
```json
{
  "diy": {
    "avg_gate": 0.787,
    "overall_accuracy": 56.58,
    ...
  },
  "geolife": {...}
}
```

**CSV**: `model_behavior_comparison.csv`

---

## 4. Script: `analyze_recency_patterns.py`

### 4.1 Purpose

Specifically analyzes recency patterns - how recently visited locations relate to next location prediction. This is critical because the pointer mechanism uses position bias to favor recent locations.

### 4.2 Main Class: `RecencyPatternAnalyzer`

```python
class RecencyPatternAnalyzer:
    """Analyzes recency patterns that explain pointer mechanism benefit."""
```

### 4.3 Core Methods

#### `analyze_target_recency(data: list, name: str) -> tuple`

```python
def analyze_target_recency(self, data: list, name: str) -> dict:
    """
    Analyze how recent the target location was visited.
    
    Key insight: If target is often the most recent location,
    the pointer mechanism with position bias will excel.
    
    Returns:
        tuple: (results_dict, target_positions_array, target_is_last_list)
    """
```

**Position calculation**:
```python
# Position from end: 1 = most recent, 2 = second most recent, etc.
for sample in data:
    positions = np.where(x == y)[0]  # All occurrences
    if len(positions) > 0:
        pos_from_end = seq_len - positions  # Convert to from-end
        most_recent_pos = min(pos_from_end)  # Smallest = most recent
```

**Metrics computed**:
- `target_in_history_pct`: % with target in history
- `target_is_last_pct`: % where target = position 1 (most recent)
- `target_in_top3_recent_pct`: % where target in positions 1-3
- `target_in_top5_recent_pct`: % where target in positions 1-5
- `avg_target_position`: Mean position from end
- `median_target_position`: Median position
- `std_target_position`: Standard deviation
- `position_distribution`: Dict of position counts (top 20)

---

#### `analyze_return_patterns(data: list, name: str) -> dict`

```python
def analyze_return_patterns(self, data: list, name: str) -> dict:
    """
    Analyze return-to-previous-location patterns.
    
    A→B→A pattern: Target equals location 2 positions back.
    These patterns indicate strong sequential dependencies.
    
    Returns:
        dict: Return pattern statistics
    """
```

**Pattern detection**:
```python
# A→B→A pattern: current location was visited 2 steps ago
return_to_prev_prev = [y == x[-2] for sample in data if len(x) >= 2]

# Return to any recent: target in last 5 locations
return_to_any_recent = [y in x[-5:] for sample in data]
```

**Metrics computed**:
- `return_to_prev_prev_pct`: A→B→A pattern rate
- `return_to_any_recent5_pct`: Return to last 5 rate

---

#### `analyze_location_predictability(data: list, name: str) -> tuple`

```python
def analyze_location_predictability(self, data: list, name: str) -> dict:
    """
    Analyze how predictable locations are based on recency.
    
    Predictability = recency_score × frequency_score
    High predictability = pointer-friendly patterns.
    
    Returns:
        tuple: (results_dict, recency_scores, frequency_scores, predictability_scores)
    """
```

**Score calculations**:
```python
# Recency score: 1/position_from_end (higher = more recent)
recency = 1 / most_recent_pos

# Frequency score: count/seq_len
frequency = len(positions) / seq_len

# Combined predictability
predictability = recency * frequency
```

**Metrics computed**:
- `avg_recency_score`: Mean 1/position (0-1 scale)
- `avg_frequency_score`: Mean count/length (0-1 scale)
- `avg_predictability_score`: Mean combined score
- `high_predictability_pct`: % with score > 0.1

---

### 4.4 Visualization Methods

#### `plot_recency_comparison(results: dict)`

Four-panel figure:
1. Target position distribution histogram
2. Key recency metrics bar chart
3. Cumulative target distribution by position
4. Correlation: recency score vs ablation impact

**Output**: `recency_pattern_analysis.png/pdf`

---

#### `plot_predictability_analysis(results: dict)`

Two-panel figure:
1. Predictability score distribution histogram
2. Predictability metrics comparison bar chart

**Output**: `predictability_analysis.png/pdf`

---

### 4.5 Output Files

**JSON**: `recency_analysis_results.json`
**CSV**: `recency_metrics.csv`
**LaTeX**: `recency_metrics.tex`

---

## 5. Script: `run_all_experiments.py`

### 5.1 Purpose

Master script that orchestrates all three analysis scripts sequentially.

### 5.2 Implementation

```python
def run_script(script_name: str) -> bool:
    """
    Run a Python script and return success status.
    
    Args:
        script_name: Name of script to run (e.g., 'analyze_mobility_patterns.py')
    
    Returns:
        bool: True if successful (returncode == 0)
    """
    script_path = PROJECT_ROOT / 'scripts' / 'gap_performance_diy_geolife' / script_name
    result = subprocess.run(
        ['python', str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )
    return result.returncode == 0

def main():
    scripts = [
        'analyze_mobility_patterns.py',
        'analyze_model_pointer.py',
        'analyze_recency_patterns.py',
    ]
    
    for script in scripts:
        success = run_script(script)
        if not success:
            print(f"WARNING: {script} failed!")
```

### 5.3 Output

Prints summary of:
- Success/failure status for each script
- List of generated tables
- List of generated figures
- List of JSON results

---

## 6. Data Structures

### 6.1 Sample Dictionary

Each sample in the pickle file has:

```python
sample = {
    'X': np.array([1, 2, 3, 2, 1]),       # Input location sequence
    'Y': 2,                                # Target next location
    'user_X': np.array([5, 5, 5, 5, 5]),  # User ID for each position
    'weekday_X': np.array([1, 1, 2, 2, 3]),# Day of week (0-6)
    'start_min_X': np.array([480, 720, 1080, ...]),  # Start time in minutes
    'dur_X': np.array([30, 60, 45, ...]), # Duration in minutes
    'diff': np.array([0, 1, 1, 2, 3]),    # Days ago from present
}
```

### 6.2 Model x_dict

Input dictionary for model forward pass:

```python
x_dict = {
    'user': torch.tensor([user_id], dtype=torch.long),
    'weekday': torch.tensor([[1, 1, 2, 2, 3]], dtype=torch.long),  # [seq_len, batch]
    'time': torch.tensor([[32, 48, 72, ...]], dtype=torch.long),   # 15-min intervals
    'duration': torch.tensor([[1, 2, 1, ...]], dtype=torch.long),  # 30-min buckets
    'diff': torch.tensor([[0, 1, 1, 2, 3]], dtype=torch.long),     # Recency
    'len': torch.tensor([5], dtype=torch.long),                     # Sequence length
}
```

---

## 7. Visualization Style Guide

### 7.1 matplotlib Configuration

All scripts use consistent classic scientific publication style:

```python
plt.rcParams.update({
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    
    # Axes settings - box style (all 4 sides visible)
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'axes.spines.top': True,
    'axes.spines.right': True,
    
    # Tick settings - inside ticks
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    
    # Legend settings
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
})
```

### 7.2 Color Palette

```python
COLORS = {
    'diy': 'blue',
    'geolife': 'red',
    'black': 'black',
    'green': 'green',
}
```

### 7.3 Markers

```python
MARKERS = {
    'diy': 'o',        # Circle
    'geolife': 's',    # Square
}
```

### 7.4 Hatching Patterns

- DIY: `'///'` (diagonal lines)
- GeoLife: `'...'` (dots)

### 7.5 Output Formats

All figures saved in both:
- PNG (300 DPI, for display)
- PDF (vector, for publication)

---

## 8. Dependencies

### 8.1 Python Standard Library

- `os`, `sys`: File system operations
- `json`: JSON serialization
- `pickle`: Data loading
- `collections`: Counter, defaultdict
- `warnings`: Warning suppression

### 8.2 Scientific Computing

```python
import numpy as np          # Array operations
import pandas as pd         # DataFrames
from scipy import stats     # Statistical tests
```

### 8.3 Deep Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
```

### 8.4 Visualization

```python
import matplotlib
matplotlib.use('Agg')       # Non-interactive backend
import matplotlib.pyplot as plt
```

### 8.5 Configuration

```python
import yaml                 # Config file parsing
```

### 8.6 Project-Specific

```python
from src.models.proposed.pgt import PointerGeneratorTransformer
```

---

## Appendix: Code Snippets

### A.1 Loading and Analyzing a Single Sample

```python
import pickle
import numpy as np

# Load data
with open('data/diy_eps50/processed/diy_eps50_prev7_test.pk', 'rb') as f:
    data = pickle.load(f)

# Analyze first sample
sample = data[0]
x = sample['X']
y = sample['Y']

# Check if target in history
target_in_history = y in x
print(f"Target {y} in history: {target_in_history}")

# Find position from end
if target_in_history:
    positions = np.where(x == y)[0]
    pos_from_end = len(x) - positions[-1]
    print(f"Position from end: {pos_from_end}")
```

### A.2 Computing Entropy

```python
from collections import Counter
import numpy as np

def compute_entropy(sequence):
    """Compute Shannon entropy of a sequence."""
    counter = Counter(sequence)
    total = len(sequence)
    probs = np.array([c / total for c in counter.values()])
    return -np.sum(probs * np.log2(probs))

# Example
x = [1, 2, 1, 3, 1, 2]
entropy = compute_entropy(x)
print(f"Entropy: {entropy:.4f} bits")
```

### A.3 Model Forward Pass with Analysis

```python
import torch
from src.models.proposed.pgt import PointerGeneratorTransformer

# Load model (simplified)
model = PointerGeneratorTransformer(num_locations=1000, num_users=100)
model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])
model.eval()

# Prepare input
x = torch.tensor([[1, 2, 3, 2, 1]]).T  # [seq_len, batch_size]
x_dict = {
    'user': torch.tensor([5]),
    'len': torch.tensor([5]),
    'time': torch.tensor([[32, 48, 72, 96, 120]]).T,
    'weekday': torch.tensor([[1, 1, 2, 2, 3]]).T,
    'duration': torch.tensor([[1, 2, 1, 2, 1]]).T,
    'diff': torch.tensor([[0, 1, 1, 2, 3]]).T,
}

# Forward pass
with torch.no_grad():
    log_probs = model(x, x_dict)
    prediction = log_probs.argmax(dim=-1)
    print(f"Prediction: {prediction.item()}")
```

---

*Technical Reference Version: 1.0*
*Generated: January 2, 2026*

# Technical API Reference

## Day-of-Week Analysis Experiment V2

This document provides detailed technical reference for all functions, classes, and modules in the experiment.

---

## Table of Contents

1. [run_days_analysis.py](#run_days_analysispy)
2. [generate_visualizations.py](#generate_visualizationspy)
3. [Data Structures](#data-structures)
4. [Configuration](#configuration)

---

## run_days_analysis.py

### Module Overview

Main script for executing the day-of-week analysis experiment. Evaluates pre-trained models on test data filtered by the day of week of the target prediction.

### Constants

```python
SEED = 42
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
WEEKDAY_INDICES = [0, 1, 2, 3, 4]  # Monday to Friday
WEEKEND_INDICES = [5, 6]  # Saturday, Sunday
```

### Configuration Dictionary

```python
CONFIGS = {
    'diy': {
        'name': 'DIY',
        'test_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_test.pk',
        'train_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_train.pk',
        'checkpoint': '/data/next_loc_clean_v2/experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt',
        'config': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pointer_v45_diy_trial09.yaml',
        'model_config': {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.2,
        }
    },
    'geolife': {
        'name': 'Geolife',
        'test_path': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_test.pk',
        'train_path': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_train.pk',
        'checkpoint': '/data/next_loc_clean_v2/experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt',
        'config': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml',
        'model_config': {
            'd_model': 96,
            'nhead': 2,
            'num_layers': 2,
            'dim_feedforward': 192,
            'dropout': 0.25,
        }
    }
}
```

---

### Functions

#### `set_seed(seed: int = 42) -> None`

Set random seeds for reproducibility across all random number generators.

**Parameters:**
- `seed` (int): Random seed value. Default: 42

**Affects:**
- Python random module
- NumPy random generator
- PyTorch CPU random generator
- PyTorch CUDA random generator
- cuDNN deterministic behavior

**Example:**
```python
set_seed(42)  # All random operations will now be reproducible
```

---

#### `compute_y_weekday(sample: dict) -> int`

Compute the weekday of the target location Y from a data sample.

**Parameters:**
- `sample` (dict): Sample dictionary containing:
  - `weekday_X`: Array of weekday indices for input sequence
  - `diff`: Array of day differences

**Returns:**
- `int`: Weekday index of target Y (0=Monday, 6=Sunday)

**Algorithm:**
```
y_weekday = (weekday_X[-1] + diff[-1]) % 7
```

**Example:**
```python
sample = {
    'weekday_X': np.array([0, 1, 2]),  # Mon, Tue, Wed
    'diff': np.array([1, 1, 2])        # 2 days after Wednesday
}
weekday = compute_y_weekday(sample)  # Returns 4 (Friday)
```

---

#### `load_model(dataset_key: str, device: torch.device) -> nn.Module`

Load a pre-trained PointerGeneratorTransformer model from checkpoint.

**Parameters:**
- `dataset_key` (str): Dataset identifier ('diy' or 'geolife')
- `device` (torch.device): Target device for model

**Returns:**
- `nn.Module`: Loaded model in evaluation mode

**Process:**
1. Load checkpoint file
2. Extract max_seq_len from position_bias tensor shape
3. Load training data to get vocabulary sizes
4. Initialize model with correct architecture
5. Load state dictionary
6. Move to device and set to eval mode

**Example:**
```python
device = torch.device('cuda')
model = load_model('diy', device)
print(f"Parameters: {model.count_parameters():,}")
```

---

#### `evaluate_on_day(model, dataset: DayFilteredDataset, device: torch.device, batch_size: int = 64) -> dict`

Evaluate model on a day-filtered dataset.

**Parameters:**
- `model`: Pre-trained model
- `dataset` (DayFilteredDataset): Dataset filtered to specific day
- `device` (torch.device): Evaluation device
- `batch_size` (int): Batch size for evaluation. Default: 64

**Returns:**
- `dict` or `None`: Performance metrics dictionary, or None if dataset is empty

**Metrics Returned:**
```python
{
    'acc@1': float,      # Top-1 accuracy (%)
    'acc@5': float,      # Top-5 accuracy (%)
    'acc@10': float,     # Top-10 accuracy (%)
    'mrr': float,        # Mean Reciprocal Rank (%)
    'ndcg': float,       # NDCG@10 (%)
    'f1': float,         # Weighted F1 score (0-1)
    'loss': float,       # Cross-entropy loss
}
```

**Example:**
```python
dataset = DayFilteredDataset(test_path, day_filter=0)  # Monday
metrics = evaluate_on_day(model, dataset, device)
print(f"Monday Acc@1: {metrics['acc@1']:.2f}%")
```

---

#### `run_day_analysis(dataset_key: str, output_dir: str) -> dict`

Run complete day-of-week analysis for a dataset.

**Parameters:**
- `dataset_key` (str): Dataset identifier ('diy' or 'geolife')
- `output_dir` (str): Directory for output files

**Returns:**
- `dict`: Complete results dictionary with:
  - Individual day results (Monday-Sunday)
  - Weekday_Avg: Weighted average of weekday metrics
  - Weekend_Avg: Weighted average of weekend metrics
  - Overall: Full test set evaluation
  - Statistical_Test: t-test results

**Side Effects:**
- Creates output directory if not exists
- Saves results JSON file

**Example:**
```python
results = run_day_analysis('diy', './results')
print(f"Weekend Acc@1: {results['Weekend_Avg']['acc@1']:.2f}%")
```

---

#### `create_results_table(results: dict, dataset_name: str) -> pd.DataFrame`

Create a formatted pandas DataFrame from results.

**Parameters:**
- `results` (dict): Results dictionary from run_day_analysis
- `dataset_name` (str): Name for labeling

**Returns:**
- `pd.DataFrame`: Formatted table with columns:
  - Day, Type, Samples, Acc@1, Acc@5, Acc@10, MRR, NDCG, F1, Loss

**Example:**
```python
df = create_results_table(results, 'DIY')
print(df.to_string())
```

---

#### `print_summary(diy_results: dict = None, geolife_results: dict = None) -> None`

Print formatted experiment summary to console.

**Parameters:**
- `diy_results` (dict, optional): DIY results dictionary
- `geolife_results` (dict, optional): GeoLife results dictionary

**Output:**
- Formatted tables for each dataset
- Statistical analysis summary

---

### Classes

#### `DayFilteredDataset`

PyTorch Dataset that filters test data by target day of week.

```python
class DayFilteredDataset(Dataset):
    """
    Dataset filtered by target day of week.
    
    Attributes:
        data: List of filtered samples
        all_data: Original unfiltered samples
        y_weekdays: Target weekday for each original sample
        num_locations: Number of unique locations
        num_users: Number of unique users
        max_seq_len: Maximum sequence length
    """
```

**Constructor:**

```python
def __init__(self, data_path: str, day_filter: int = None):
    """
    Args:
        data_path: Path to pickle file with test data
        day_filter: Day to filter (0-6) or None for all
    """
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `__len__()` | int | Number of filtered samples |
| `__getitem__(idx)` | tuple | (x, y, x_dict) for sample |
| `_compute_statistics()` | None | Calculate dataset statistics |

**Example:**
```python
# All samples
full_dataset = DayFilteredDataset(test_path, day_filter=None)

# Monday samples only
monday_dataset = DayFilteredDataset(test_path, day_filter=0)
print(f"Monday samples: {len(monday_dataset)}")
```

---

#### `collate_fn`

Custom collate function for variable-length sequences.

```python
def collate_fn(batch) -> tuple:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (x, y, x_dict) tuples
        
    Returns:
        tuple: (x_batch, y_batch, x_dict_batch)
            - x_batch: [seq_len, batch_size] padded locations
            - y_batch: [batch_size] targets
            - x_dict_batch: Dictionary with padded tensors
    """
```

---

## generate_visualizations.py

### Module Overview

Generates publication-quality visualizations following Nature Journal style guidelines.

### Style Configuration

```python
# Classic scientific publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    # ... more settings
})
```

### Color Palette

```python
COLORS = {
    'weekday': 'green',
    'weekend': 'orange',
    'diy': 'blue',
    'geolife': 'red',
    'black': 'black',
    'green': 'green',
}

MARKERS = {
    'diy': 'o',      # Circle
    'geolife': 's',  # Square
}
```

---

### Functions

#### `setup_classic_axes(ax) -> None`

Configure axes to match classic scientific publication style.

**Parameters:**
- `ax`: Matplotlib Axes object

**Effects:**
- All spines visible and black
- Inside tick marks on all sides
- Major ticks: length=5, width=1
- Minor ticks: length=3, width=0.5

---

#### `load_results(results_dir: str, dataset: str) -> dict`

Load results from JSON file.

**Parameters:**
- `results_dir` (str): Directory containing results
- `dataset` (str): 'diy' or 'geolife'

**Returns:**
- `dict`: Loaded results dictionary

**Raises:**
- `FileNotFoundError`: If results file doesn't exist

---

#### `plot_accuracy_by_day(results: dict, dataset_name: str, output_dir: str) -> None`

Create 3-panel bar chart showing Acc@1, Acc@5, Acc@10 by day.

**Parameters:**
- `results` (dict): Results dictionary
- `dataset_name` (str): Dataset name for labeling
- `output_dir` (str): Output directory

**Output Files:**
- `{dataset}_accuracy_by_day.pdf`
- `{dataset}_accuracy_by_day.png`
- `{dataset}_accuracy_by_day.svg`

**Visual Elements:**
- Green bars with backslash hatch: Weekdays
- Orange bars with forward slash hatch: Weekends
- Value labels above bars

---

#### `plot_weekday_weekend_comparison(results: dict, dataset_name: str, output_dir: str) -> None`

Create grouped bar chart comparing weekday vs weekend.

**Parameters:**
- `results` (dict): Results dictionary
- `dataset_name` (str): Dataset name for labeling
- `output_dir` (str): Output directory

**Output Files:**
- `{dataset}_weekday_weekend_comparison.pdf`
- `{dataset}_weekday_weekend_comparison.png`
- `{dataset}_weekday_weekend_comparison.svg`

**Visual Elements:**
- Side-by-side bars for weekday/weekend
- Delta annotations showing differences
- Significance annotation (p-value)

---

#### `plot_performance_heatmap(results: dict, dataset_name: str, output_dir: str) -> None`

Create grayscale heatmap of all metrics by day.

**Parameters:**
- `results` (dict): Results dictionary
- `dataset_name` (str): Dataset name for labeling
- `output_dir` (str): Output directory

**Output Files:**
- `{dataset}_metrics_heatmap.pdf`
- `{dataset}_metrics_heatmap.png`

**Visual Elements:**
- Grayscale colormap
- Annotated cell values
- Red border around weekend rows

---

#### `plot_performance_trend(results: dict, dataset_name: str, output_dir: str) -> None`

Create line plot showing metric trends across days.

**Parameters:**
- `results` (dict): Results dictionary
- `dataset_name` (str): Dataset name for labeling
- `output_dir` (str): Output directory

**Output Files:**
- `{dataset}_performance_trend.pdf`
- `{dataset}_performance_trend.png`
- `{dataset}_performance_trend.svg`

**Visual Elements:**
- Multi-line plot (ACC@1, ACC@5, ACC@10, MRR)
- Open markers (circles, squares, triangles, diamonds)
- Weekend region shading

---

#### `plot_sample_distribution(results: dict, dataset_name: str, output_dir: str) -> None`

Create bar chart of sample counts by day.

**Parameters:**
- `results` (dict): Results dictionary
- `dataset_name` (str): Dataset name for labeling
- `output_dir` (str): Output directory

**Output Files:**
- `{dataset}_sample_distribution.pdf`
- `{dataset}_sample_distribution.png`

---

#### `plot_combined_comparison(diy_results: dict, geolife_results: dict, output_dir: str) -> None`

Create side-by-side dataset comparison.

**Parameters:**
- `diy_results` (dict): DIY results
- `geolife_results` (dict): GeoLife results
- `output_dir` (str): Output directory

**Output Files:**
- `combined_comparison.pdf`
- `combined_comparison.png`
- `combined_comparison.svg`

---

#### `create_combined_figure(diy_results: dict, geolife_results: dict, output_dir: str) -> None`

Create comprehensive 9-panel publication figure.

**Parameters:**
- `diy_results` (dict): DIY results
- `geolife_results` (dict): GeoLife results
- `output_dir` (str): Output directory

**Output Files:**
- `combined_figure.pdf`
- `combined_figure.png`
- `combined_figure.svg`

**Panel Layout:**
- (a-b): Daily Acc@1 for each dataset
- (c): Comparison line plot
- (d-e): Weekday vs weekend for each dataset
- (f): Weekend drop comparison
- (g-h): Sample distribution for each dataset
- (i): Statistical summary

---

#### `generate_latex_table(results: dict, dataset_name: str, output_dir: str) -> None`

Generate LaTeX table for publication.

**Parameters:**
- `results` (dict): Results dictionary
- `dataset_name` (str): Dataset name for labeling
- `output_dir` (str): Output directory

**Output Files:**
- `{dataset}_table.tex`

**LaTeX Features:**
- booktabs package formatting
- Gray highlighting for weekend rows
- Bold formatting for aggregate rows

---

#### `create_summary_csv(diy_results: dict, geolife_results: dict, output_dir: str) -> pd.DataFrame`

Create comprehensive CSV summary.

**Parameters:**
- `diy_results` (dict): DIY results
- `geolife_results` (dict): GeoLife results
- `output_dir` (str): Output directory

**Returns:**
- `pd.DataFrame`: Combined summary table

**Output Files:**
- `days_analysis_summary.csv`

---

## Data Structures

### Sample Dictionary

Each data sample in the pickle files has the following structure:

```python
sample = {
    'X': np.array([...]),           # Location sequence (int array)
    'Y': int,                       # Target location
    'user_X': np.array([...]),      # User IDs (same for all elements)
    'weekday_X': np.array([...]),   # Weekday indices (0-6)
    'start_min_X': np.array([...]), # Start time in minutes from midnight
    'dur_X': np.array([...]),       # Duration in minutes
    'diff': np.array([...]),        # Days since this visit
}
```

### Results Dictionary

```python
results = {
    'Monday': {
        'day_index': 0,
        'samples': 2020,
        'is_weekend': False,
        'correct@1': 1157.0,
        'correct@3': 1570.0,
        'correct@5': 1665.0,
        'correct@10': 1721.0,
        'rr': 1376.05,
        'ndcg': 72.23,
        'f1': 0.53,
        'total': 2020.0,
        'acc@1': 57.28,
        'acc@5': 82.43,
        'acc@10': 85.20,
        'mrr': 68.12,
        'loss': 2.44,
    },
    # ... Tuesday through Sunday ...
    'Weekday_Avg': {
        'day_index': -2,
        'samples': 8578,
        'is_weekend': False,
        'acc@1': 57.24,
        # ... other averaged metrics
    },
    'Weekend_Avg': {
        'day_index': -3,
        'samples': 3790,
        'is_weekend': True,
        'acc@1': 55.09,
        # ... other averaged metrics
    },
    'Overall': {
        'day_index': -1,
        'samples': 12368,
        'is_weekend': None,
        # ... overall metrics
    },
    'Statistical_Test': {
        'test': 'Independent t-test',
        'comparison': 'Weekday vs Weekend Acc@1',
        't_statistic': 1.32,
        'p_value': 0.24,
        'significant_at_005': False,
        'significant_at_001': False,
        'weekday_mean': 57.51,
        'weekend_mean': 55.10,
        'difference': 2.42,
    }
}
```

---

## Configuration

### Command Line Arguments

**run_days_analysis.py:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | 'both' | 'diy', 'geolife', or 'both' |
| `--output_dir` | str | './results' | Output directory |
| `--seed` | int | 42 | Random seed |

**generate_visualizations.py:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--results_dir` | str | './results' | Results directory |
| `--output_dir` | str | './figures' | Output directory |

### Environment Variables

None required. All paths are hardcoded in CONFIGS.

### Dependencies

```
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.7.0
tqdm>=4.60.0
```

---

*End of Technical API Reference*

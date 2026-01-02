# 05. Technical Implementation

## Code, Scripts, and Execution Details

---

## Document Overview

| Item | Details |
|------|---------|
| **Document Type** | Technical Documentation |
| **Audience** | Developers, Researchers who want to reproduce |
| **Reading Time** | 20-25 minutes |
| **Prerequisites** | Python, PyTorch familiarity |

---

## 1. Directory Structure

### 1.1 Experiment Directory Layout

```
experiment_sequence_len_days_v2/
│
├── evaluate_sequence_length.py    # Main evaluation script
├── visualize_results.py           # Visualization generation
├── run_experiment.sh              # Master execution script
├── README.md                      # Quick start guide
│
├── docs/                          # Documentation (this folder)
│   ├── INDEX.md
│   ├── 01_executive_summary.md
│   ├── 02_introduction_and_motivation.md
│   ├── ... (more docs)
│   └── COMPREHENSIVE_DOCUMENTATION.md
│
└── results/                       # Output files
    ├── diy_sequence_length_results.json
    ├── geolife_sequence_length_results.json
    ├── full_results.csv
    ├── summary_statistics.csv
    ├── improvement_analysis.csv
    ├── results_table.tex
    ├── statistics_table.tex
    └── *.{pdf,png,svg}           # Visualizations
```

### 1.2 Related Project Files

```
next_loc_clean_v2/
│
├── src/
│   ├── models/
│   │   └── proposed/
│   │       └── pointer_v45.py          # Model architecture
│   ├── evaluation/
│   │   └── metrics.py                   # Evaluation metrics
│   └── data/
│       └── dataset.py                   # Dataset classes
│
├── data/
│   ├── diy_eps50/processed/
│   │   ├── diy_eps50_prev7_train.pk
│   │   └── diy_eps50_prev7_test.pk
│   └── geolife_eps20/processed/
│       ├── geolife_eps20_prev7_train.pk
│       └── geolife_eps20_prev7_test.pk
│
├── experiments/
│   ├── diy_pointer_v45_20260101_155348/
│   │   └── checkpoints/best.pt
│   └── geolife_pointer_v45_20260101_151038/
│       └── checkpoints/best.pt
│
└── scripts/
    └── sci_hyperparam_tuning/
        └── configs/
            ├── pointer_v45_diy_trial09.yaml
            └── pointer_v45_geolife_trial01.yaml
```

---

## 2. Configuration System

### 2.1 EXPERIMENT_CONFIG Dictionary

The main configuration is defined in `evaluate_sequence_length.py`:

```python
EXPERIMENT_CONFIG = {
    'diy': {
        'checkpoint_path': '/data/next_loc_clean_v2/experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt',
        'config_path': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pointer_v45_diy_trial09.yaml',
        'test_data_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_test.pk',
        'train_data_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_train.pk',
        'dataset_name': 'DIY',
    },
    'geolife': {
        'checkpoint_path': '/data/next_loc_clean_v2/experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt',
        'config_path': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml',
        'test_data_path': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_test.pk',
        'train_data_path': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_train.pk',
        'dataset_name': 'GeoLife',
    },
}
```

### 2.2 YAML Configuration Files

Example: `pointer_v45_diy_trial09.yaml`

```yaml
# Hyperparameter Tuning Config: pointer_v45_diy_trial09
# Model: pointer_v45, Dataset: diy, Trial: 9

seed: 42

data:
  data_dir: data/diy_eps50/processed
  dataset_prefix: diy_eps50_prev7
  dataset: diy
  experiment_root: experiments
  num_workers: 0

model:
  d_model: 64
  nhead: 4
  num_layers: 2
  dim_feedforward: 256
  dropout: 0.2

training:
  batch_size: 64
  num_epochs: 50
  learning_rate: 0.0005
  weight_decay: 1.0e-05
  label_smoothing: 0.05
  grad_clip: 0.8
  patience: 5
  min_epochs: 8
  warmup_epochs: 7
  use_amp: true
  min_lr: 1.0e-06
```

### 2.3 Configuration Loading

```python
import yaml

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
```

---

## 3. Data Handling

### 3.1 Data Format (Pickle Files)

Test data is stored as pickle files containing lists of dictionaries:

```python
# Structure of each sample
sample = {
    'X': np.array([...]),           # Location IDs, shape [seq_len]
    'user_X': np.array([...]),      # User IDs (repeated), shape [seq_len]
    'weekday_X': np.array([...]),   # Day of week (0-6), shape [seq_len]
    'start_min_X': np.array([...]), # Time of day (0-95), shape [seq_len]
    'dur_X': np.array([...]),       # Duration bucket, shape [seq_len]
    'diff': np.array([...]),        # Days ago (0-7+), shape [seq_len]
    'Y': int,                       # Target location ID
}
```

### 3.2 SequenceLengthDataset Class

```python
class SequenceLengthDataset(Dataset):
    """
    Dataset that filters sequences based on previous_days parameter.
    
    Filters location visits to include only those within the specified
    temporal window (measured in days).
    """
    
    def __init__(self, data_path, previous_days, min_seq_length=2):
        """
        Initialize dataset with temporal filtering.
        
        Args:
            data_path: Path to pickle file containing test data
            previous_days: Number of days to include (1-7)
            min_seq_length: Minimum sequence length after filtering
        """
        self.previous_days = previous_days
        self.min_seq_length = min_seq_length
        
        # Load original data
        with open(data_path, 'rb') as f:
            original_data = pickle.load(f)
        
        # Filter sequences
        self.data = self._filter_sequences(original_data, previous_days)
        
        # Compute statistics
        self._compute_statistics()
    
    def _filter_sequences(self, original_data, previous_days):
        """Filter sequences to match temporal window."""
        filtered_data = []
        
        for sample in original_data:
            diff = sample['diff']
            
            # Create mask for visits within window
            # diff <= previous_days means "at most previous_days days ago"
            mask = diff <= previous_days
            
            # Skip if too few visits remain
            if mask.sum() < self.min_seq_length:
                continue
            
            # Apply mask to all sequence features
            filtered_sample = {
                'X': sample['X'][mask],
                'user_X': sample['user_X'][mask],
                'weekday_X': sample['weekday_X'][mask],
                'start_min_X': sample['start_min_X'][mask],
                'dur_X': sample['dur_X'][mask],
                'diff': sample['diff'][mask],
                'Y': sample['Y'],  # Target unchanged
            }
            filtered_data.append(filtered_sample)
        
        return filtered_data
    
    def _compute_statistics(self):
        """Compute sequence length statistics."""
        lengths = [len(s['X']) for s in self.data]
        self.avg_seq_len = np.mean(lengths)
        self.std_seq_len = np.std(lengths)
        self.max_seq_len = np.max(lengths) if lengths else 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

### 3.3 Custom Collate Function

```python
def collate_fn(batch):
    """
    Collate function for variable-length sequences.
    
    Pads sequences to batch maximum length and creates attention masks.
    """
    # Find max length in batch
    max_len = max(len(sample['X']) for sample in batch)
    batch_size = len(batch)
    
    # Initialize tensors
    X = torch.zeros(max_len, batch_size, dtype=torch.long)
    user_X = torch.zeros(max_len, batch_size, dtype=torch.long)
    weekday_X = torch.zeros(max_len, batch_size, dtype=torch.long)
    time_X = torch.zeros(max_len, batch_size, dtype=torch.long)
    dur_X = torch.zeros(max_len, batch_size, dtype=torch.long)
    diff_X = torch.zeros(max_len, batch_size, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    Y = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, sample in enumerate(batch):
        seq_len = len(sample['X'])
        X[:seq_len, i] = torch.from_numpy(sample['X'])
        user_X[:seq_len, i] = torch.from_numpy(sample['user_X'])
        weekday_X[:seq_len, i] = torch.from_numpy(sample['weekday_X'])
        time_X[:seq_len, i] = torch.from_numpy(sample['start_min_X'])
        dur_X[:seq_len, i] = torch.from_numpy(sample['dur_X'])
        diff_X[:seq_len, i] = torch.from_numpy(sample['diff'])
        lengths[i] = seq_len
        Y[i] = sample['Y']
    
    # Create input dictionary for model
    x_dict = {
        'user': user_X[0],  # User ID (constant per sequence)
        'time': time_X,
        'weekday': weekday_X,
        'diff': diff_X,
        'duration': dur_X,
        'len': lengths,
    }
    
    return X, x_dict, Y
```

---

## 4. Model Loading

### 4.1 Loading Checkpoint

```python
def load_model(config, checkpoint_path, device):
    """
    Load pre-trained model from checkpoint.
    
    Args:
        config: YAML configuration dictionary
        checkpoint_path: Path to checkpoint file
        device: torch.device to load model on
    
    Returns:
        Loaded model in evaluation mode
    """
    # Get dataset statistics for model initialization
    with open(config['train_data_path'], 'rb') as f:
        train_data = pickle.load(f)
    
    # Count unique locations and users
    all_locations = set()
    all_users = set()
    for sample in train_data:
        all_locations.update(sample['X'].tolist())
        all_locations.add(sample['Y'])
        all_users.update(sample['user_X'].tolist())
    
    num_locations = max(all_locations) + 1
    num_users = max(all_users) + 1
    
    # Initialize model
    model = PointerNetworkV45(
        num_locations=num_locations,
        num_users=num_users,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        max_seq_len=150,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    return model, num_locations
```

### 4.2 Checkpoint Structure

```python
# Checkpoint contents
checkpoint = {
    'epoch': int,                    # Training epoch
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # LR scheduler state
    'best_acc': float,               # Best validation accuracy
    'config': dict,                  # Training configuration
}
```

---

## 5. Evaluation Loop

### 5.1 Main Evaluation Function

```python
def evaluate_sequence_length(dataset_key, batch_size=64):
    """
    Evaluate model across different sequence lengths.
    
    Args:
        dataset_key: 'diy' or 'geolife'
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with results for each previous_days value
    """
    config = EXPERIMENT_CONFIG[dataset_key]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration
    yaml_config = load_config(config['config_path'])
    
    # Load model
    model, num_locations = load_model(yaml_config, config['checkpoint_path'], device)
    
    # Results container
    results = {}
    
    # Evaluate for each previous_days value (1-7)
    for prev_days in range(1, 8):
        print(f"\n{'='*50}")
        print(f"Evaluating {config['dataset_name']} with prev_days={prev_days}")
        print(f"{'='*50}")
        
        # Create filtered dataset
        dataset = SequenceLengthDataset(
            config['test_data_path'],
            previous_days=prev_days,
            min_seq_length=2
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        # Run evaluation
        metrics = evaluate_model_detailed(model, dataloader, device, num_locations)
        
        # Store results
        results[str(prev_days)] = {
            'metrics': metrics,
            'num_samples': len(dataset),
            'avg_seq_len': dataset.avg_seq_len,
            'std_seq_len': dataset.std_seq_len,
            'max_seq_len': dataset.max_seq_len,
        }
        
        # Print summary
        print(f"Samples: {len(dataset)}")
        print(f"Avg Seq Len: {dataset.avg_seq_len:.2f} (±{dataset.std_seq_len:.2f})")
        print(f"Acc@1: {metrics['acc@1']:.2f}%")
        print(f"Acc@5: {metrics['acc@5']:.2f}%")
        print(f"MRR: {metrics['mrr']:.2f}%")
        print(f"Loss: {metrics['loss']:.4f}")
    
    return results
```

### 5.2 Detailed Model Evaluation

```python
def evaluate_model_detailed(model, dataloader, device, num_locations):
    """
    Evaluate model and compute all metrics including loss.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with test data
        device: torch.device
        num_locations: Number of location classes
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    # Accumulators
    all_logits = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for X, x_dict, Y in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            X = X.to(device)
            Y = Y.to(device)
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            
            # Forward pass
            log_probs = model(X, x_dict)  # [batch_size, num_locations]
            
            # Compute loss
            loss = criterion(log_probs, Y)
            total_loss += loss.item()
            num_batches += 1
            
            # Store for metrics
            all_logits.append(log_probs.cpu())
            all_targets.append(Y.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics using standard metrics module
    metrics = calculate_metrics(all_logits, all_targets)
    
    # Add average loss
    metrics['loss'] = total_loss / num_batches
    
    return metrics
```

---

## 6. Results Saving

### 6.1 JSON Results Format

```python
def save_results(results, dataset_name, output_dir):
    """Save results to JSON file."""
    output = {
        'dataset': dataset_name,
        'experiment_date': datetime.now().isoformat(),
        'checkpoint': EXPERIMENT_CONFIG[dataset_name.lower()]['checkpoint_path'],
        'results': results,
    }
    
    output_path = os.path.join(output_dir, f'{dataset_name.lower()}_sequence_length_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {output_path}")
```

### 6.2 CSV Export

```python
def results_to_csv(diy_results, geolife_results, output_dir):
    """Export results to CSV for analysis."""
    rows = []
    
    for dataset, results in [('DIY', diy_results), ('GeoLife', geolife_results)]:
        for prev_days, data in results.items():
            row = {
                'dataset': dataset,
                'prev_days': int(prev_days),
                'num_samples': data['num_samples'],
                'avg_seq_len': data['avg_seq_len'],
                'std_seq_len': data['std_seq_len'],
                'max_seq_len': data['max_seq_len'],
            }
            row.update(data['metrics'])
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'full_results.csv'), index=False)
```

### 6.3 LaTeX Table Generation

```python
def generate_latex_table(df, output_path):
    """Generate publication-ready LaTeX table."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Impact of Sequence Length on Next Location Prediction Performance}
\label{tab:sequence_length_results}
\begin{tabular}{llcccccc}
\toprule
Dataset & Prev Days & Acc@1 & Acc@5 & Acc@10 & MRR & NDCG & F1 \\
\midrule
"""
    
    for dataset in ['DIY', 'GeoLife']:
        subset = df[df['dataset'] == dataset]
        for _, row in subset.iterrows():
            latex += f"{dataset} & {row['prev_days']} & "
            latex += f"{row['acc@1']:.2f} & {row['acc@5']:.2f} & {row['acc@10']:.2f} & "
            latex += f"{row['mrr']:.2f} & {row['ndcg']:.2f} & {row['f1']:.2f} \\\\\n"
        latex += r"\midrule" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
```

---

## 7. Visualization Script

### 7.1 Style Configuration

```python
# Publication-quality style settings
STYLE_CONFIG = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'font.family': 'serif',
    'font.size': 10,
    'legend.frameon': False,
}

# Color scheme
COLORS = {
    'diy': '#1f77b4',      # Blue
    'geolife': '#d62728',  # Red
}

# Marker styles
MARKERS = {
    'diy': 'o',      # Circle
    'geolife': 's',  # Square
}
```

### 7.2 Performance Comparison Plot

```python
def plot_performance_comparison(df, output_dir):
    """Generate 6-panel performance comparison plot."""
    plt.rcParams.update(STYLE_CONFIG)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
    titles = ['Accuracy@1 (%)', 'Accuracy@5 (%)', 'Accuracy@10 (%)',
              'MRR (%)', 'NDCG@10 (%)', 'F1 Score (%)']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for dataset in ['DIY', 'GeoLife']:
            subset = df[df['dataset'] == dataset]
            ax.plot(
                subset['prev_days'],
                subset[metric],
                marker=MARKERS[dataset.lower()],
                color=COLORS[dataset.lower()],
                markerfacecolor='white',
                markeredgewidth=1.5,
                markersize=8,
                linewidth=1.5,
                label=dataset
            )
        
        ax.set_xlabel('t (days)')
        ax.set_ylabel(title)
        ax.set_xticks(range(1, 8))
        ax.legend()
    
    plt.tight_layout()
    
    # Save in multiple formats
    for ext in ['pdf', 'png', 'svg']:
        plt.savefig(os.path.join(output_dir, f'performance_comparison.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
```

### 7.3 Combined Figure

```python
def plot_combined_figure(df, output_dir):
    """Generate publication-ready combined figure."""
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid: 3 rows x 3 columns
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Acc@1, Acc@5, MRR
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Row 2: NDCG, Loss, Seq Length
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Row 3: Improvement bar chart (spans 2 cols), Sample count
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Plot each panel...
    # [Implementation details for each subplot]
    
    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
                         ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'combined_figure.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 8. Execution Instructions

### 8.1 Command Line Interface

```bash
# Run full experiment (both datasets)
python evaluate_sequence_length.py --dataset all --batch_size 64

# Run single dataset
python evaluate_sequence_length.py --dataset diy --batch_size 64
python evaluate_sequence_length.py --dataset geolife --batch_size 64

# Generate visualizations
python visualize_results.py
```

### 8.2 Master Script (run_experiment.sh)

```bash
#!/bin/bash

# Sequence Length Days Experiment V2
# This script runs the complete experiment

set -e  # Exit on error

echo "=========================================="
echo "Sequence Length Days Experiment V2"
echo "=========================================="

# Step 1: Run evaluation
echo -e "\n[Step 1/2] Running evaluation..."
python evaluate_sequence_length.py --dataset all --batch_size 64

# Step 2: Generate visualizations
echo -e "\n[Step 2/2] Generating visualizations..."
python visualize_results.py

echo -e "\n=========================================="
echo "Experiment completed successfully!"
echo "Results saved to: results/"
echo "=========================================="
```

### 8.3 Expected Output

```
==========================================
Sequence Length Days Experiment V2
==========================================

[Step 1/2] Running evaluation...

==================================================
Evaluating DIY with prev_days=1
==================================================
Samples: 11532
Avg Seq Len: 5.62 (±4.13)
Acc@1: 50.00%
Acc@5: 72.55%
MRR: 59.97%
Loss: 3.7628

[... continues for all configurations ...]

[Step 2/2] Generating visualizations...
Generating performance comparison plot...
Generating accuracy heatmap...
Generating loss curve...
Generating radar comparison...
Generating improvement comparison...
Generating sequence length distribution...
Generating samples vs performance...
Generating combined figure...

==========================================
Experiment completed successfully!
Results saved to: results/
==========================================
```

---

## 9. Dependencies

### 9.1 Python Packages

```python
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

# Progress
tqdm>=4.62.0

# Data
pickle5>=0.0.11  # For older Python versions
```

### 9.2 Environment Setup

```bash
# Create conda environment
conda create -n mlenv python=3.9
conda activate mlenv

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy pandas pyyaml matplotlib seaborn scikit-learn tqdm
```

---

## 10. Troubleshooting

### 10.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Batch size too large | Reduce batch_size |
| FileNotFoundError | Wrong paths in config | Verify paths exist |
| KeyError in sample | Missing data field | Check data format |
| Import error | Missing package | Install dependencies |

### 10.2 Debugging Commands

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# Check data loading
import pickle
with open('path/to/test.pk', 'rb') as f:
    data = pickle.load(f)
print(f"Samples: {len(data)}")
print(f"Sample keys: {data[0].keys()}")

# Check model loading
checkpoint = torch.load('path/to/best.pt')
print(f"Checkpoint keys: {checkpoint.keys()}")
```

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Created** | 2026-01-02 |
| **Word Count** | ~2,800 |
| **Status** | Final |

---

**Navigation**: [← Experimental Methodology](./04_experimental_methodology.md) | [Index](./INDEX.md) | [Next: Model Architecture →](./06_model_architecture.md)

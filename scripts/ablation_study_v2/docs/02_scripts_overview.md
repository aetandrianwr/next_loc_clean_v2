# 2. Scripts Overview

## Complete Guide to Ablation Study Scripts

---

## 2.1 Directory Structure

```
scripts/ablation_study_v2/
├── docs/                           # This documentation
│   ├── 00_table_of_contents.md
│   ├── 01_introduction.md
│   ├── 02_scripts_overview.md      # ← You are here
│   └── ...
├── configs/                        # Configuration files (if any)
├── logs/                           # Training logs
│   ├── geolife_full_baseline.log
│   ├── geolife_no_pointer.log
│   ├── diy_full_baseline.log
│   └── ...
├── results/                        # Experiment outputs
│   ├── geolife/
│   │   ├── ablation_results.csv
│   │   ├── ablation_table.tex
│   │   └── ablation_geolife_{variant}_{timestamp}/
│   └── diy/
│       ├── ablation_results.csv
│       ├── ablation_table.tex
│       └── ablation_diy_{variant}_{timestamp}/
├── pointer_v45_ablation.py         # Ablation model variants
├── train_ablation.py               # Training script
├── run_ablation_study.py           # Main orchestrator
└── collect_results.py              # Results analyzer
```

---

## 2.2 Script 1: pointer_v45_ablation.py

### Purpose
This script defines the `PointerNetworkV45Ablation` class, which is a modified version of the original model that can selectively disable specific components.

### Key Features

```python
class PointerNetworkV45Ablation(nn.Module):
    """
    Ablation variant of PointerNetworkV45.
    
    Supports selective component disabling via ablation_type parameter.
    """
    
    VALID_ABLATIONS = [
        'full',              # Complete model (baseline)
        'no_pointer',        # Remove pointer mechanism
        'no_generation',     # Remove generation head
        'no_position_bias',  # Remove position bias
        'no_temporal',       # Remove temporal embeddings
        'no_user',           # Remove user embeddings
        'no_pos_from_end',   # Remove position-from-end
        'single_layer',      # Use single transformer layer
        'no_gate',           # Fixed 0.5 gate
    ]
```

### How It Works

The ablation model uses conditional initialization to include or exclude components:

```python
def __init__(self, ..., ablation_type='full'):
    # Example: User embedding is conditionally created
    self.use_user = ablation_type != 'no_user'
    if self.use_user:
        self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
    
    # Example: Transformer layers adjusted for single_layer ablation
    actual_num_layers = 1 if ablation_type == 'single_layer' else num_layers
```

### Forward Pass Logic

The forward method adapts based on which components are enabled:

```python
def forward(self, x, x_dict):
    # Build embeddings based on ablation configuration
    embeddings = [self.loc_emb(x)]  # Location always included
    
    if self.use_user:
        embeddings.append(user_emb)
    
    if self.use_temporal:
        embeddings.append(temporal)
    
    if self.use_pos_from_end:
        embeddings.append(pos_emb)
    
    # Combine and process through transformer
    combined = torch.cat(embeddings, dim=-1)
    ...
    
    # Distribution computation depends on ablation
    if self.use_pointer and self.use_generation:
        # Both pointer and generation
        final_probs = gate * ptr_dist + (1 - gate) * gen_probs
    elif self.use_pointer:
        # Pointer only
        final_probs = ptr_dist
    else:
        # Generation only
        final_probs = gen_probs
```

### Usage Example

```python
from pointer_v45_ablation import PointerNetworkV45Ablation

# Create full model (baseline)
model_full = PointerNetworkV45Ablation(
    num_locations=1000,
    num_users=100,
    ablation_type='full'  # All components enabled
)

# Create model without pointer mechanism
model_no_pointer = PointerNetworkV45Ablation(
    num_locations=1000,
    num_users=100,
    ablation_type='no_pointer'  # Pointer disabled
)
```

---

## 2.3 Script 2: train_ablation.py

### Purpose
This script handles the complete training pipeline for a single ablation experiment.

### Key Components

#### 1. Dataset Class
```python
class NextLocationDataset(Dataset):
    """
    Loads preprocessed data from pickle files.
    
    Provides:
    - Location sequence (X)
    - Target location (Y)
    - Temporal features: user, time, weekday, duration, diff
    """
```

#### 2. Collate Function
```python
def collate_fn(batch):
    """
    Handles variable-length sequences.
    
    - Pads sequences to same length
    - Stacks tensors for batch processing
    """
```

#### 3. Trainer Class
```python
class AblationTrainer:
    """
    Handles the complete training loop.
    
    Features:
    - Mixed precision training (AMP)
    - Warmup + Cosine LR schedule
    - Early stopping on validation loss
    - Gradient clipping
    - Comprehensive logging
    """
```

### Training Protocol

```python
def train(self):
    for epoch in range(self.num_epochs):
        # 1. Set learning rate (warmup + cosine decay)
        lr = self._get_lr(epoch)
        self._set_lr(lr)
        
        # 2. Train epoch
        train_loss = self.train_epoch()
        
        # 3. Validate
        val_metrics = self.evaluate(self.val_loader, "val")
        
        # 4. Early stopping check
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.patience_counter = 0
            self._save_checkpoint("best.pt")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                break  # Early stop
    
    # 5. Load best model and evaluate on test set
    self._load_checkpoint("best.pt")
    test_metrics = self.evaluate(self.test_loader, "test")
    
    return val_metrics, test_metrics
```

### Command Line Usage

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Run single ablation experiment
python scripts/ablation_study_v2/train_ablation.py \
    --config scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml \
    --ablation full \
    --output_dir scripts/ablation_study_v2/results/geolife \
    --seed 42
```

### Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--config` | Path to YAML config file | Yes | - |
| `--ablation` | Ablation type | Yes | - |
| `--output_dir` | Output directory | No | `scripts/ablation_study_v2/results` |
| `--seed` | Random seed | No | 42 |

### Output Files

Each experiment creates a directory with:
```
ablation_{dataset}_{variant}_{timestamp}/
├── checkpoints/
│   └── best.pt              # Best model checkpoint
├── training.log             # Detailed training log
├── results.json             # Final metrics
└── config.yaml              # Experiment configuration
```

---

## 2.4 Script 3: run_ablation_study.py

### Purpose
This is the main orchestrator that runs all ablation experiments in parallel.

### Key Configuration

```python
# Configuration
SEED = 42
PATIENCE = 5
MAX_PARALLEL_JOBS = 3  # Run 3 experiments simultaneously
JOB_DELAY_SECONDS = 5  # 5 second delay between starts

# Ablation types to evaluate
ABLATION_TYPES = [
    'full',              # Baseline
    'no_pointer',        # Remove pointer mechanism
    'no_generation',     # Remove generation head
    'no_position_bias',  # Remove position bias
    'no_temporal',       # Remove temporal embeddings
    'no_user',           # Remove user embeddings
    'no_pos_from_end',   # Remove position-from-end
    'single_layer',      # Single transformer layer
    'no_gate',           # Fixed 0.5 gate
]

# Dataset configurations
DATASET_CONFIGS = {
    'geolife': {
        'config_path': 'scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml',
        'expected_acc1': 51.39,
    },
    'diy': {
        'config_path': 'scripts/sci_hyperparam_tuning/configs/pointer_v45_diy_trial09.yaml',
        'expected_acc1': 56.58,
    },
}
```

### Parallel Execution Strategy

```python
def run_ablation_study(datasets=None):
    for dataset_name in datasets:
        # Run ablations in batches of MAX_PARALLEL_JOBS
        for batch_start in range(0, len(ABLATION_TYPES), MAX_PARALLEL_JOBS):
            batch = ABLATION_TYPES[batch_start:batch_start + MAX_PARALLEL_JOBS]
            
            processes = []
            for i, ablation_type in enumerate(batch):
                if i > 0:
                    time.sleep(JOB_DELAY_SECONDS)  # 5 second delay
                
                # Start subprocess
                proc = subprocess.Popen(cmd, ...)
                processes.append((ablation_type, proc))
            
            # Wait for all processes in batch to complete
            for ablation_type, proc in processes:
                proc.communicate()
```

### Visual Representation of Parallel Execution

```
Time →
────────────────────────────────────────────────────────────────────
Batch 1:
  Job 1 (full):        [████████████████████████]
  Job 2 (no_pointer):       [████████████████████████]
  Job 3 (no_generation):         [████████████████████████]
                       ↑    ↑    ↑
                       0s   5s   10s (start delays)

Batch 2 (starts after Batch 1 completes):
  Job 4 (no_position_bias): [████████████████████████]
  Job 5 (no_temporal):           [████████████████████████]
  Job 6 (no_user):                    [████████████████████████]

Batch 3:
  Job 7 (no_pos_from_end): [████████████████████████]
  Job 8 (single_layer):         [████████████████████████]
  Job 9 (no_gate):                   [████████████████████████]
────────────────────────────────────────────────────────────────────
```

### Usage

```bash
# Run full ablation study on both datasets
python scripts/ablation_study_v2/run_ablation_study.py

# Run on specific dataset
python scripts/ablation_study_v2/run_ablation_study.py --dataset geolife
python scripts/ablation_study_v2/run_ablation_study.py --dataset diy

# Custom output directory
python scripts/ablation_study_v2/run_ablation_study.py --output_dir /path/to/results
```

---

## 2.5 Script 4: collect_results.py

### Purpose
This script collects results from all experiments, generates summary statistics, LaTeX tables, and comprehensive reports.

### Key Functions

#### 1. Collect Results
```python
def collect_dataset_results(dataset_dir):
    """
    Walks through experiment directories and collects results.json files.
    
    Returns list of dictionaries with all metrics.
    """
    results = []
    for exp_dir in dataset_dir.iterdir():
        if (exp_dir / 'results.json').exists():
            with open(exp_dir / 'results.json') as f:
                data = json.load(f)
            results.append(parse_metrics(data))
    return results
```

#### 2. Generate LaTeX Table
```python
def generate_latex_table(df, dataset_name, baseline_acc1):
    """
    Generates publication-quality LaTeX table.
    
    Output format suitable for Nature Journal submission.
    """
```

#### 3. Generate Summary Statistics
```python
def generate_summary_statistics(df, dataset_name, baseline_acc1):
    """
    Creates human-readable summary with:
    - Component impact ranking
    - Key insights
    - Interpretation
    """
```

### Output Files Generated

| File | Description |
|------|-------------|
| `ablation_results.csv` | All metrics in CSV format |
| `ablation_table.tex` | LaTeX table for papers |
| `ablation_summary_report.txt` | Human-readable summary |

### Usage

```bash
python scripts/ablation_study_v2/collect_results.py
```

### Sample Output

```
======================================================================
ABLATION STUDY SUMMARY - GEOLIFE DATASET
======================================================================
Baseline (Full Model) Test Acc@1: 51.43%

Component Impact Ranking (by Acc@1 drop):
--------------------------------------------------
1. w/o Pointer Mechanism: -24.01% (46.7% relative drop)
2. w/o Temporal Embeddings: -4.03% (7.8% relative drop)
3. w/o User Embedding: -2.31% (4.5% relative drop)
...
```

---

## 2.6 How to Run Complete Ablation Study

### Step 1: Prepare Environment

```bash
# Navigate to project directory
cd /data/next_loc_clean_v2

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Verify setup
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Run Individual Experiment (for testing)

```bash
# Test with one ablation first
python scripts/ablation_study_v2/train_ablation.py \
    --config scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml \
    --ablation full \
    --output_dir scripts/ablation_study_v2/results/geolife \
    --seed 42
```

### Step 3: Run Full Study (all ablations)

```bash
# Option A: Use the orchestrator script
python scripts/ablation_study_v2/run_ablation_study.py

# Option B: Manual parallel execution (what was actually done)
# Batch 1 (GeoLife)
python train_ablation.py --ablation no_pointer ... &
sleep 5
python train_ablation.py --ablation no_generation ... &
sleep 5
python train_ablation.py --ablation no_position_bias ... &
wait
# Repeat for other batches...
```

### Step 4: Collect Results

```bash
python scripts/ablation_study_v2/collect_results.py
```

### Step 5: Review Results

```bash
# View summary report
cat scripts/ablation_study_v2/results/ablation_summary_report.txt

# View CSV results
cat scripts/ablation_study_v2/results/geolife/ablation_results.csv
```

---

## 2.7 Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size in config
# Or run fewer parallel jobs
MAX_PARALLEL_JOBS = 2  # Instead of 3
```

#### Issue 2: File Not Found
```bash
# Verify config paths
ls scripts/sci_hyperparam_tuning/configs/

# Verify data paths
ls data/geolife_eps20/processed/
```

#### Issue 3: Import Errors
```bash
# Ensure you're in the project root
cd /data/next_loc_clean_v2

# Verify imports
python -c "from src.evaluation.metrics import calculate_correct_total_prediction"
```

---

## 2.8 Code Quality Notes

### Design Principles

1. **Modularity**: Each script has a single responsibility
2. **Reproducibility**: Fixed seeds, documented parameters
3. **Logging**: Comprehensive logging for debugging
4. **Error Handling**: Graceful failure with informative messages

### Testing

```bash
# Test ablation model creation
python -c "
from scripts.ablation_study_v2.pointer_v45_ablation import PointerNetworkV45Ablation
for abl in PointerNetworkV45Ablation.VALID_ABLATIONS:
    model = PointerNetworkV45Ablation(100, 10, ablation_type=abl)
    print(f'{abl}: {model.count_parameters():,} params')
"
```

---

*Next: [03_model_architecture.md](03_model_architecture.md) - Detailed model architecture explanation*

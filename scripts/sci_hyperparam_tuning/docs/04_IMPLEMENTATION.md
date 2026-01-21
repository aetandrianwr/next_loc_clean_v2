# Code Implementation Details

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Configuration Generation](#configuration-generation)
3. [Parallel Training Manager](#parallel-training-manager)
4. [Results Logging](#results-logging)
5. [Final Evaluation Pipeline](#final-evaluation-pipeline)
6. [Code Walkthrough](#code-walkthrough)

---

## System Architecture

### Overall System Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         HYPERPARAMETER TUNING SYSTEM                       │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────┐                                                  │
│  │ hyperparam_search_  │  Defines search spaces for all models            │
│  │   space.py          │  PGT_SEARCH_SPACE, MHSA_SEARCH_SPACE,   │
│  └──────────┬──────────┘  LSTM_SEARCH_SPACE                              │
│             │                                                             │
│             ▼                                                             │
│  ┌─────────────────────┐                                                  │
│  │ generate_configs.py │  Generates 120 YAML configuration files         │
│  │                     │  using random sampling with fixed seed           │
│  └──────────┬──────────┘                                                  │
│             │                                                             │
│             ▼                                                             │
│  ┌─────────────────────────────────────────────────────┐                 │
│  │              configs/ directory                      │                 │
│  │  pointer_v45_geolife_trial00.yaml                   │                 │
│  │  pointer_v45_geolife_trial01.yaml                   │                 │
│  │  ...                                                │                 │
│  │  lstm_diy_trial19.yaml                              │                 │
│  │  all_configs_summary.yaml                           │                 │
│  └──────────┬──────────────────────────────────────────┘                 │
│             │                                                             │
│             ▼                                                             │
│  ┌─────────────────────┐                                                  │
│  │ run_hyperparam_     │  Parallel execution manager:                    │
│  │   tuning.py         │  - 5 concurrent jobs                            │
│  │                     │  - Auto-resume on interrupt                     │
│  │                     │  - CSV logging                                  │
│  └──────────┬──────────┘                                                  │
│             │                                                             │
│             ▼                                                             │
│  ┌─────────────────────────────────────────────────────┐                 │
│  │              results/ directory                      │                 │
│  │  pointer_v45_geolife_val_results.csv                │                 │
│  │  pointer_v45_geolife_test_results.csv               │                 │
│  │  ...                                                │                 │
│  └──────────┬──────────────────────────────────────────┘                 │
│             │                                                             │
│             ▼                                                             │
│  ┌─────────────────────┐                                                  │
│  │ run_final_          │  Final evaluation with multiple seeds           │
│  │   evaluation.py     │  for statistical significance                   │
│  └─────────────────────┘                                                  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Generation

### File: `generate_configs.py`

This script creates all 120 YAML configuration files for the hyperparameter search.

### Key Components

#### 1. Dataset Configurations

```python
DATASET_CONFIGS = {
    'geolife': {
        'data_dir': 'data/geolife_eps20/processed',
        'dataset_prefix': 'geolife_eps20_prev7',
        'total_loc_num': 1187,   # Number of unique locations
        'total_user_num': 46,    # Number of users
    },
    'diy': {
        'data_dir': 'data/diy_eps50/processed',
        'dataset_prefix': 'diy_eps50_prev7',
        'total_loc_num': 7038,   # Larger vocabulary
        'total_user_num': 693,   # More users
    }
}
```

#### 2. Config Generation Functions

Each model has a dedicated function to translate hyperparameters into the appropriate YAML structure:

**Pointer Generator Transformer Configuration:**
```python
def generate_pointer_v45_config(hp_config: dict, dataset: str) -> dict:
    """Generate Pointer Generator Transformer YAML config from hyperparameters."""
    ds_cfg = DATASET_CONFIGS[dataset]
    
    return {
        'seed': 42,
        'data': {
            'data_dir': ds_cfg['data_dir'],
            'dataset_prefix': ds_cfg['dataset_prefix'],
            'dataset': dataset,
            'experiment_root': 'experiments',
            'num_workers': 0,
        },
        'model': {
            'd_model': hp_config['d_model'],
            'nhead': hp_config['nhead'],
            'num_layers': hp_config['num_layers'],
            'dim_feedforward': hp_config['dim_feedforward'],
            'dropout': hp_config['dropout'],
        },
        'training': {
            'batch_size': hp_config['batch_size'],
            'num_epochs': 50,
            'learning_rate': hp_config['learning_rate'],
            'weight_decay': hp_config['weight_decay'],
            'label_smoothing': hp_config['label_smoothing'],
            'grad_clip': 0.8,
            'patience': 5,
            'min_epochs': 8,
            'warmup_epochs': hp_config['warmup_epochs'],
            'use_amp': True,
            'min_lr': 1e-6,
        },
    }
```

**MHSA Configuration:**
```python
def generate_mhsa_config(hp_config: dict, dataset: str) -> dict:
    """Generate MHSA YAML config from hyperparameters."""
    return {
        'seed': 42,
        'data': { ... },
        'training': {
            'if_embed_user': True,
            'if_embed_poi': False,
            'if_embed_time': True,
            'if_embed_duration': True,
            'previous_day': 7,
            'batch_size': hp_config['batch_size'],
            ...
        },
        'embedding': {
            'base_emb_size': hp_config['base_emb_size'],
        },
        'model': {
            'networkName': 'transformer',
            'num_encoder_layers': hp_config['num_encoder_layers'],
            'nhead': hp_config['nhead'],
            'dim_feedforward': hp_config['dim_feedforward'],
            'fc_dropout': hp_config['fc_dropout'],
        },
        'optimiser': {
            'optimizer': 'Adam',
            'lr': hp_config['lr'],
            'weight_decay': hp_config['weight_decay'],
            'num_warmup_epochs': hp_config['num_warmup_epochs'],
            'num_training_epochs': 50,
            'patience': 5,
        },
    }
```

#### 3. Main Generation Loop

```python
def main():
    set_seed(RANDOM_SEED)  # 42
    
    config_dir = Path(__file__).parent / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    generators = {
        'pgt': generate_pointer_v45_config,
        'mhsa': generate_mhsa_config,
        'lstm': generate_lstm_config,
    }
    
    all_configs = []
    for model_name in ['pgt', 'mhsa', 'lstm']:
        for dataset in ['geolife', 'diy']:
            hp_configs = generate_all_configs(model_name, dataset, num_trials=20)
            
            for hp_config in hp_configs:
                config_name = get_config_name(model_name, dataset, hp_config['trial_idx'])
                yaml_config = generators[model_name](hp_config, dataset)
                
                # Save with header comment
                config_path = config_dir / f'{config_name}.yaml'
                with open(config_path, 'w') as f:
                    f.write(f"# Hyperparameter Tuning Config: {config_name}\n")
                    f.write(f"# Model: {model_name}, Dataset: {dataset}\n")
                    f.write(f"# Hyperparameters: {hp_config}\n\n")
                    yaml.dump(yaml_config, f, default_flow_style=False)
```

---

## Parallel Training Manager

### File: `run_hyperparam_tuning.py`

This script orchestrates parallel execution of all training experiments.

### Configuration Constants

```python
MAX_PARALLEL_JOBS = 5      # Run 5 experiments simultaneously
DELAY_BETWEEN_JOBS = 1.5   # Seconds between job starts
BASE_DIR = Path(__file__).parent.parent.parent  # Repository root
```

### Training Script Mapping

```python
TRAINING_SCRIPTS = {
    'pgt': 'src/training/train_pgt.py',
    'mhsa': 'src/training/train_MHSA.py',
    'lstm': 'src/training/train_LSTM.py',
}
```

### Core Components

#### 1. ExperimentResult Data Class

```python
@dataclass
class ExperimentResult:
    """Data class for storing experiment results."""
    config_name: str
    model_name: str
    dataset: str
    trial_idx: int
    num_params: int
    correct_at_1: float
    correct_at_3: float
    correct_at_5: float
    correct_at_10: float
    total: float
    rr: float           # Sum of reciprocal ranks
    ndcg: float         # Sum of NDCG scores
    f1: float
    acc_at_1: float     # Accuracy@1 percentage
    acc_at_5: float
    acc_at_10: float
    mrr: float          # Mean Reciprocal Rank
    loss: float
    experiment_dir: str
    config_path: str
    hyperparameters: str
    status: str         # 'SUCCESS', 'FAILED', 'TIMEOUT'
    timestamp: str
```

#### 2. Thread-Safe Results Logger

```python
class ResultsLogger:
    """Thread-safe CSV logger for experiment results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.lock = threading.Lock()  # Prevent race conditions
        self._init_csv_files()
    
    def log_result(self, result: ExperimentResult, split: str):
        """Log a result to the appropriate CSV file."""
        with self.lock:  # Thread-safe writing
            csv_path = self.results_dir / f'{result.model_name}_{result.dataset}_{split}_results.csv'
            with open(csv_path, 'a', newline='') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # File lock
                writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
                writer.writerow(asdict(result))
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

#### 3. Single Experiment Runner

```python
def run_single_experiment(config_name, config_path, model_name, 
                          dataset, trial_idx, hyperparameters, logger):
    """Run a single experiment and log results."""
    
    training_script = TRAINING_SCRIPTS[model_name]
    start_time = time.time()
    
    # Build and execute training command
    cmd = [
        'bash', '-c',
        f'source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv && '
        f'cd {BASE_DIR} && python {training_script} --config {config_path}'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 2  # 2 hour timeout per experiment
        )
        
        if result.returncode != 0:
            # Log failure and return
            logger.log_result(fail_result, 'val')
            logger.log_result(fail_result, 'test')
            return False
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {config_name}")
        return False
    
    # Find experiment directory (created by training script)
    experiment_dir = find_experiment_dir(...)
    
    # Parse results from JSON files
    val_results = parse_results_from_json(f'{experiment_dir}/val_results.json')
    test_results = parse_results_from_json(f'{experiment_dir}/test_results.json')
    num_params = parse_num_params_from_log(f'{experiment_dir}/training.log')
    
    # Log results for both val and test
    for split, results in [('val', val_results), ('test', test_results)]:
        exp_result = ExperimentResult(
            config_name=config_name,
            model_name=model_name,
            dataset=dataset,
            acc_at_1=results.get('acc@1', 0),
            # ... all other fields
        )
        logger.log_result(exp_result, split)
    
    return True
```

#### 4. Worker Thread Function

```python
def worker(task_queue: queue.Queue, logger: ResultsLogger, worker_id: int):
    """Worker thread for running experiments."""
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:  # Poison pill - shut down
                break
            
            config_name, config_path, model_name, dataset, trial_idx, hp = task
            run_single_experiment(config_name, config_path, model_name, 
                                 dataset, trial_idx, hp, logger)
            
            task_queue.task_done()
            time.sleep(DELAY_BETWEEN_JOBS)  # Prevent resource contention
            
        except queue.Empty:
            continue
```

#### 5. Resume Capability

```python
def load_completed_configs(results_dir: Path) -> set:
    """Load set of already completed config names."""
    completed = set()
    
    for csv_file in results_dir.glob('*_val_results.csv'):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'SUCCESS':
                    completed.add(row['config_name'])
    
    return completed

# In main():
completed = load_completed_configs(RESULTS_DIR)
remaining_configs = [c for c in all_configs if c['config_name'] not in completed]
```

This allows the script to be interrupted and resumed without re-running completed experiments.

---

## Results Logging

### CSV File Structure

Each results CSV file contains these columns:

| Column | Type | Description |
|--------|------|-------------|
| config_name | str | e.g., "pointer_v45_geolife_trial01" |
| model_name | str | "pgt", "mhsa", or "lstm" |
| dataset | str | "geolife" or "diy" |
| trial_idx | int | 0-19 |
| num_params | int | Model parameter count |
| correct_at_1 | float | Number of top-1 correct predictions |
| correct_at_3 | float | Number of top-3 correct predictions |
| correct_at_5 | float | Number of top-5 correct predictions |
| correct_at_10 | float | Number of top-10 correct predictions |
| total | float | Total number of predictions |
| rr | float | Sum of reciprocal ranks |
| ndcg | float | Sum of NDCG@10 scores |
| f1 | float | Weighted F1 score |
| acc_at_1 | float | Accuracy@1 percentage |
| acc_at_5 | float | Accuracy@5 percentage |
| acc_at_10 | float | Accuracy@10 percentage |
| mrr | float | Mean Reciprocal Rank percentage |
| loss | float | Validation/test loss |
| experiment_dir | str | Path to experiment directory |
| config_path | str | Path to YAML config file |
| hyperparameters | str | JSON string of hyperparameters |
| status | str | "SUCCESS", "FAILED", or "TIMEOUT" |
| timestamp | str | ISO format timestamp |

---

## Final Evaluation Pipeline

### File: `run_final_evaluation.py`

After hyperparameter tuning, this script runs the best configuration multiple times with different random seeds to establish statistical significance.

### Key Functions

```python
NUM_FINAL_RUNS = 5
SEEDS = [42, 123, 456, 789, 1011]

def find_best_configs(results_df):
    """Find best configuration for each model-dataset by Val Acc@1."""
    best_configs = {}
    
    for model in ['pgt', 'mhsa', 'lstm']:
        for dataset in ['geolife', 'diy']:
            subset = results_df[
                (results_df['model_name'] == model) & 
                (results_df['dataset'] == dataset) &
                (results_df['status'] == 'SUCCESS')
            ]
            
            if len(subset) > 0:
                best_idx = subset['acc_at_1'].idxmax()
                best_configs[f"{model}_{dataset}"] = {
                    'config_name': subset.loc[best_idx, 'config_name'],
                    'config_path': subset.loc[best_idx, 'config_path'],
                    'val_acc_at_1': subset.loc[best_idx, 'acc_at_1'],
                }
    
    return best_configs

def run_final_evaluations(best_configs):
    """Run 5 evaluations for each best config with different seeds."""
    final_results = {}
    
    for key, config_info in best_configs.items():
        results_list = []
        
        for run_idx, seed in enumerate(SEEDS):
            # Modify config to use new seed
            config['seed'] = seed
            
            # Run training
            val_res, test_res = run_single_evaluation(config, seed)
            results_list.append(test_res)
        
        # Compute mean ± std
        final_results[key] = {
            'acc@1': {
                'mean': np.mean([r['acc@1'] for r in results_list]),
                'std': np.std([r['acc@1'] for r in results_list]),
            },
            # ... other metrics
        }
    
    return final_results
```

---

## Code Walkthrough

### Complete Execution Flow

```bash
# Step 1: Generate all configurations
cd /data/next_loc_clean_v2
python scripts/sci_hyperparam_tuning/generate_configs.py
# Creates 120 YAML files in configs/

# Step 2: Run hyperparameter tuning
python scripts/sci_hyperparam_tuning/run_hyperparam_tuning.py
# Runs all experiments, logs to results/

# Step 3: (Optional) Run final evaluation
python scripts/sci_hyperparam_tuning/run_final_evaluation.py
# Runs best configs 5x with different seeds
```

### Example Log Output

```
============================================================
SCIENTIFIC HYPERPARAMETER TUNING
============================================================
Base directory: /data/next_loc_clean_v2
Max parallel jobs: 5
Delay between jobs: 1.5s
============================================================
Total configs: 120
Already completed: 0
Remaining: 120
============================================================

[08:10:34] Starting: pointer_v45_geolife_trial00
[08:10:36] Starting: pointer_v45_geolife_trial01
[08:10:37] Starting: pointer_v45_geolife_trial02
[08:10:39] Starting: pointer_v45_geolife_trial03
[08:10:40] Starting: pointer_v45_geolife_trial04
[08:11:11] DONE: pointer_v45_geolife_trial02 (Val Acc@1: 48.02%, Time: 0.6min)
[08:11:32] DONE: pointer_v45_geolife_trial01 (Val Acc@1: 49.25%, Time: 0.9min)
...
```

---

## Next: [05_MODELS.md](05_MODELS.md) - Model Architecture Details

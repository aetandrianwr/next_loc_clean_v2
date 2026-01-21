"""
Parallel Hyperparameter Tuning Manager.

This script manages the parallel execution of hyperparameter tuning experiments.
It maintains 5 concurrent training sessions, logs results to CSV files, and
tracks all experiment metrics.

Scientific Methodology:
1. Random Search (Bergstra & Bengio, 2012) for hyperparameter selection
2. Fixed seed (42) for reproducibility
3. Early stopping with patience=5 for efficiency
4. Val Acc@1 as primary optimization objective
5. All results logged with full metrics for analysis

Usage:
    python scripts/sci_hyperparam_tuning/run_hyperparam_tuning.py
"""

import os
import sys
import json
import yaml
import time
import csv
import re
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import fcntl

# Configuration
MAX_PARALLEL_JOBS = 5
DELAY_BETWEEN_JOBS = 1.5  # seconds
BASE_DIR = Path(__file__).parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPTS_DIR / 'results'
CONFIGS_DIR = SCRIPTS_DIR / 'configs'

# Training scripts
TRAINING_SCRIPTS = {
    "pgt": 'src/training/train_pgt.py',
    'mhsa': 'src/training/train_MHSA.py',
    'lstm': 'src/training/train_LSTM.py',
}

# CSV columns for results
RESULTS_COLUMNS = [
    'config_name', 'model_name', 'dataset', 'trial_idx', 'num_params',
    'correct_at_1', 'correct_at_3', 'correct_at_5', 'correct_at_10', 'total',
    'rr', 'ndcg', 'f1', 'acc_at_1', 'acc_at_5', 'acc_at_10', 'mrr', 'loss',
    'experiment_dir', 'config_path', 'hyperparameters', 'status', 'timestamp'
]


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
    rr: float
    ndcg: float
    f1: float
    acc_at_1: float
    acc_at_5: float
    acc_at_10: float
    mrr: float
    loss: float
    experiment_dir: str
    config_path: str
    hyperparameters: str
    status: str
    timestamp: str


class ResultsLogger:
    """Thread-safe CSV logger for experiment results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        
        # Initialize CSV files
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist."""
        for split in ['val', 'test']:
            for model in ["pgt", 'mhsa', 'lstm']:
                for dataset in ['geolife', 'diy']:
                    csv_path = self.results_dir / f'{model}_{dataset}_{split}_results.csv'
                    if not csv_path.exists():
                        with open(csv_path, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
                            writer.writeheader()
    
    def log_result(self, result: ExperimentResult, split: str):
        """Log a result to the appropriate CSV file."""
        with self.lock:
            csv_path = self.results_dir / f'{result.model_name}_{result.dataset}_{split}_results.csv'
            
            # Use file locking for safety
            with open(csv_path, 'a', newline='') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
                writer.writerow(asdict(result))
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def parse_num_params_from_log(log_path: str, model_name: str) -> int:
    """Parse number of parameters from training log."""
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        # Different models print params differently
        if model_name == "pgt":
            # Look for "Model parameters: XXX" or "Parameters: XXX"
            match = re.search(r'(?:Model\s+)?[Pp]arameters:\s*([\d,]+)', log_content)
        else:
            # MHSA and LSTM use "Total trainable parameters: XXX"
            match = re.search(r'Total trainable parameters:\s*([\d,]+)', log_content)
        
        if match:
            return int(match.group(1).replace(',', ''))
    except Exception as e:
        print(f"Warning: Could not parse params from {log_path}: {e}")
    
    return 0


def parse_results_from_json(json_path: str) -> Dict:
    """Parse results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not parse results from {json_path}: {e}")
        return {}


def find_experiment_dir(experiments_root: Path, config_name: str, model_name: str, 
                        dataset: str, start_time: float) -> Optional[str]:
    """Find the experiment directory created after start_time."""
    model_suffix = {
        "pgt": "pgt",
        'mhsa': 'MHSA',
        'lstm': 'LSTM',
    }[model_name]
    
    dataset_prefix = dataset
    pattern = f"{dataset_prefix}_{model_suffix}_"
    
    # Look for directories matching the pattern created after start_time
    candidates = []
    for d in experiments_root.iterdir():
        if d.is_dir() and d.name.startswith(pattern):
            try:
                # Get directory creation time
                dir_time = d.stat().st_mtime
                if dir_time >= start_time - 10:  # 10 second buffer
                    candidates.append((d, dir_time))
            except:
                pass
    
    if candidates:
        # Return the most recent one
        candidates.sort(key=lambda x: x[1], reverse=True)
        return str(candidates[0][0])
    
    return None


def run_single_experiment(config_name: str, config_path: str, model_name: str, 
                          dataset: str, trial_idx: int, hyperparameters: Dict,
                          logger: ResultsLogger) -> bool:
    """Run a single experiment and log results."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting: {config_name}")
    
    training_script = TRAINING_SCRIPTS[model_name]
    start_time = time.time()
    
    # Run training
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
            timeout=3600 * 2  # 2 hour timeout
        )
        
        success = result.returncode == 0
        
        if not success:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] FAILED: {config_name}")
            print(f"  stderr: {result.stderr[:500] if result.stderr else 'None'}")
            
            # Log failure
            fail_result = ExperimentResult(
                config_name=config_name,
                model_name=model_name,
                dataset=dataset,
                trial_idx=trial_idx,
                num_params=0,
                correct_at_1=0, correct_at_3=0, correct_at_5=0, correct_at_10=0,
                total=0, rr=0, ndcg=0, f1=0,
                acc_at_1=0, acc_at_5=0, acc_at_10=0, mrr=0, loss=0,
                experiment_dir='',
                config_path=config_path,
                hyperparameters=str(hyperparameters),
                status='FAILED',
                timestamp=datetime.now().isoformat()
            )
            logger.log_result(fail_result, 'val')
            logger.log_result(fail_result, 'test')
            return False
        
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] TIMEOUT: {config_name}")
        return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {config_name}: {e}")
        return False
    
    # Find experiment directory
    experiments_root = BASE_DIR / 'experiments'
    experiment_dir = find_experiment_dir(experiments_root, config_name, model_name, dataset, start_time)
    
    if not experiment_dir:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Could not find experiment dir for {config_name}")
        return False
    
    # Parse results
    val_json_path = os.path.join(experiment_dir, 'val_results.json')
    test_json_path = os.path.join(experiment_dir, 'test_results.json')
    log_path = os.path.join(experiment_dir, 'training.log')
    
    val_results = parse_results_from_json(val_json_path)
    test_results = parse_results_from_json(test_json_path)
    num_params = parse_num_params_from_log(log_path, model_name)
    
    # Create result objects
    for split, results in [('val', val_results), ('test', test_results)]:
        if results:
            exp_result = ExperimentResult(
                config_name=config_name,
                model_name=model_name,
                dataset=dataset,
                trial_idx=trial_idx,
                num_params=num_params,
                correct_at_1=results.get('correct@1', 0),
                correct_at_3=results.get('correct@3', 0),
                correct_at_5=results.get('correct@5', 0),
                correct_at_10=results.get('correct@10', 0),
                total=results.get('total', 0),
                rr=results.get('rr', 0),
                ndcg=results.get('ndcg', results.get('ndcg@10', 0)),
                f1=results.get('f1', 0),
                acc_at_1=results.get('acc@1', 0),
                acc_at_5=results.get('acc@5', 0),
                acc_at_10=results.get('acc@10', 0),
                mrr=results.get('mrr', 0),
                loss=results.get('loss', 0),
                experiment_dir=experiment_dir,
                config_path=config_path,
                hyperparameters=str(hyperparameters),
                status='SUCCESS',
                timestamp=datetime.now().isoformat()
            )
            logger.log_result(exp_result, split)
    
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE: {config_name} "
          f"(Val Acc@1: {val_results.get('acc@1', 0):.2f}%, Time: {elapsed/60:.1f}min)")
    
    return True


def worker(task_queue: queue.Queue, logger: ResultsLogger, worker_id: int):
    """Worker thread for running experiments."""
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:  # Poison pill
                break
            
            config_name, config_path, model_name, dataset, trial_idx, hyperparameters = task
            run_single_experiment(config_name, config_path, model_name, dataset, 
                                 trial_idx, hyperparameters, logger)
            
            task_queue.task_done()
            time.sleep(DELAY_BETWEEN_JOBS)
            
        except queue.Empty:
            continue


def load_completed_configs(results_dir: Path) -> set:
    """Load set of already completed config names."""
    completed = set()
    
    for csv_file in results_dir.glob('*_val_results.csv'):
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('status') == 'SUCCESS':
                        completed.add(row['config_name'])
        except:
            pass
    
    return completed


def main():
    """Main entry point for hyperparameter tuning."""
    print("=" * 60)
    print("SCIENTIFIC HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Max parallel jobs: {MAX_PARALLEL_JOBS}")
    print(f"Delay between jobs: {DELAY_BETWEEN_JOBS}s")
    print("=" * 60)
    
    # Initialize logger
    logger = ResultsLogger(RESULTS_DIR)
    
    # Load configs summary
    summary_path = CONFIGS_DIR / 'all_configs_summary.yaml'
    if not summary_path.exists():
        print("ERROR: Config summary not found. Run generate_configs.py first.")
        sys.exit(1)
    
    with open(summary_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    # Filter out already completed configs
    completed = load_completed_configs(RESULTS_DIR)
    remaining_configs = [c for c in all_configs if c['config_name'] not in completed]
    
    print(f"Total configs: {len(all_configs)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining_configs)}")
    print("=" * 60)
    
    if not remaining_configs:
        print("All experiments completed!")
        return
    
    # Create task queue
    task_queue = queue.Queue()
    
    # Add tasks to queue
    for config in remaining_configs:
        task = (
            config['config_name'],
            config['config_path'],
            config['model_name'],
            config['dataset'],
            config['trial_idx'],
            config['hyperparameters'],
        )
        task_queue.put(task)
    
    # Start worker threads
    workers = []
    for i in range(MAX_PARALLEL_JOBS):
        t = threading.Thread(target=worker, args=(task_queue, logger, i))
        t.daemon = True
        t.start()
        workers.append(t)
        time.sleep(DELAY_BETWEEN_JOBS)  # Stagger start times
    
    # Wait for all tasks to complete
    task_queue.join()
    
    # Send poison pills to stop workers
    for _ in workers:
        task_queue.put(None)
    
    for t in workers:
        t.join(timeout=5)
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

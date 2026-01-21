"""
Best Hyperparameter Finder and Final Evaluation Runner.

This script:
1. Analyzes hyperparameter tuning results to find the best configuration
2. Runs final evaluation with best hyperparameters (5 runs per model/dataset)
3. Computes mean ± std for all metrics
4. Generates final comparison tables

Usage:
    python scripts/sci_hyperparam_tuning/run_final_evaluation.py
"""

import os
import sys
import json
import yaml
import csv
import time
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

# Configuration
NUM_FINAL_RUNS = 5
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

# Metrics to evaluate
METRICS = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1', 'loss']


def load_tuning_results() -> pd.DataFrame:
    """Load all hyperparameter tuning results into a DataFrame."""
    all_results = []
    
    for csv_file in RESULTS_DIR.glob('*_val_results.csv'):
        try:
            df = pd.read_csv(csv_file)
            df['source_file'] = csv_file.name
            all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.concat(all_results, ignore_index=True)


def find_best_configs(results_df: pd.DataFrame) -> Dict[str, Dict]:
    """Find the best configuration for each model-dataset pair based on Val Acc@1."""
    best_configs = {}
    
    for model in ["pgt", 'mhsa', 'lstm']:
        for dataset in ['geolife', 'diy']:
            key = f"{model}_{dataset}"
            
            # Filter for this model-dataset pair with successful runs
            mask = (
                (results_df['model_name'] == model) & 
                (results_df['dataset'] == dataset) &
                (results_df['status'] == 'SUCCESS')
            )
            subset = results_df[mask]
            
            if len(subset) == 0:
                print(f"WARNING: No successful runs for {key}")
                continue
            
            # Find best by Val Acc@1
            best_idx = subset['acc_at_1'].idxmax()
            best_row = subset.loc[best_idx]
            
            best_configs[key] = {
                'config_name': best_row['config_name'],
                'config_path': best_row['config_path'],
                'model_name': model,
                'dataset': dataset,
                'val_acc_at_1': best_row['acc_at_1'],
                'hyperparameters': best_row['hyperparameters'],
                'num_params': best_row['num_params'],
            }
            
            print(f"{key}: Best Val Acc@1 = {best_row['acc_at_1']:.2f}% "
                  f"(Config: {best_row['config_name']})")
    
    return best_configs


def parse_results_from_json(json_path: str) -> Dict:
    """Parse results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except:
        return {}


def parse_num_params_from_log(log_path: str, model_name: str) -> int:
    """Parse number of parameters from training log."""
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        if model_name == "pgt":
            match = re.search(r'(?:Model\s+)?[Pp]arameters:\s*([\d,]+)', log_content)
        else:
            match = re.search(r'Total trainable parameters:\s*([\d,]+)', log_content)
        
        if match:
            return int(match.group(1).replace(',', ''))
    except:
        pass
    return 0


def find_experiment_dir(experiments_root: Path, model_name: str, 
                        dataset: str, start_time: float) -> Optional[str]:
    """Find the experiment directory created after start_time."""
    model_suffix = {
        "pgt": "pgt",
        'mhsa': 'MHSA',
        'lstm': 'LSTM',
    }[model_name]
    
    pattern = f"{dataset}_{model_suffix}_"
    
    candidates = []
    for d in experiments_root.iterdir():
        if d.is_dir() and d.name.startswith(pattern):
            try:
                dir_time = d.stat().st_mtime
                if dir_time >= start_time - 10:
                    candidates.append((d, dir_time))
            except:
                pass
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return str(candidates[0][0])
    
    return None


def run_single_evaluation(config_path: str, model_name: str, dataset: str, 
                          run_idx: int, seed: int) -> Tuple[Dict, Dict, int, str]:
    """Run a single evaluation and return val/test results."""
    print(f"  Run {run_idx + 1}/{NUM_FINAL_RUNS} (seed={seed})...")
    
    training_script = TRAINING_SCRIPTS[model_name]
    start_time = time.time()
    
    # Modify config to use specific seed
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['seed'] = seed
    
    # Create temporary config with modified seed
    temp_config_path = CONFIGS_DIR / f'temp_final_{model_name}_{dataset}_run{run_idx}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    cmd = [
        'bash', '-c',
        f'source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv && '
        f'cd {BASE_DIR} && python {training_script} --config {temp_config_path}'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 2
        )
        
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            return {}, {}, 0, ''
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return {}, {}, 0, ''
    finally:
        # Cleanup temp config
        if temp_config_path.exists():
            temp_config_path.unlink()
    
    # Find experiment directory
    experiments_root = BASE_DIR / 'experiments'
    experiment_dir = find_experiment_dir(experiments_root, model_name, dataset, start_time)
    
    if not experiment_dir:
        print(f"    WARNING: Could not find experiment directory")
        return {}, {}, 0, ''
    
    # Parse results
    val_results = parse_results_from_json(os.path.join(experiment_dir, 'val_results.json'))
    test_results = parse_results_from_json(os.path.join(experiment_dir, 'test_results.json'))
    num_params = parse_num_params_from_log(os.path.join(experiment_dir, 'training.log'), model_name)
    
    elapsed = time.time() - start_time
    print(f"    Val Acc@1: {val_results.get('acc@1', 0):.2f}%, "
          f"Test Acc@1: {test_results.get('acc@1', 0):.2f}% ({elapsed/60:.1f}min)")
    
    return val_results, test_results, num_params, experiment_dir


def run_final_evaluations(best_configs: Dict[str, Dict]) -> Dict[str, Dict]:
    """Run final evaluations for all best configurations."""
    final_results = {}
    
    # Different seeds for 5 runs (reproducible)
    seeds = [42, 123, 456, 789, 1011]
    
    for key, config_info in best_configs.items():
        print(f"\n{'='*60}")
        print(f"Running final evaluation: {key}")
        print(f"Config: {config_info['config_name']}")
        print(f"{'='*60}")
        
        val_results_list = []
        test_results_list = []
        experiment_dirs = []
        num_params = 0
        
        for run_idx in range(NUM_FINAL_RUNS):
            val_res, test_res, params, exp_dir = run_single_evaluation(
                config_info['config_path'],
                config_info['model_name'],
                config_info['dataset'],
                run_idx,
                seeds[run_idx]
            )
            
            if val_res and test_res:
                val_results_list.append(val_res)
                test_results_list.append(test_res)
                experiment_dirs.append(exp_dir)
                if params > 0:
                    num_params = params
        
        if not test_results_list:
            print(f"WARNING: No successful runs for {key}")
            continue
        
        # Compute mean ± std for all metrics
        final_results[key] = {
            'config_name': config_info['config_name'],
            'config_path': config_info['config_path'],
            'hyperparameters': config_info['hyperparameters'],
            'num_params': num_params,
            'num_runs': len(test_results_list),
            'experiment_dirs': experiment_dirs,
            'val': {},
            'test': {},
        }
        
        for split, results_list in [('val', val_results_list), ('test', test_results_list)]:
            for metric in METRICS:
                values = [r.get(metric, 0) for r in results_list if metric in r]
                if values:
                    final_results[key][split][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values,
                    }
    
    return final_results


def generate_results_tables(final_results: Dict[str, Dict]):
    """Generate final comparison tables."""
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    for dataset in ['geolife', 'diy']:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")
        
        # Header
        print(f"{'Model':<15} {'Params':<12} {'Acc@1':<15} {'Acc@5':<15} {'Acc@10':<15} {'MRR':<15}")
        print("-" * 90)
        
        for model in ["pgt", 'mhsa', 'lstm']:
            key = f"{model}_{dataset}"
            if key not in final_results:
                print(f"{model:<15} N/A")
                continue
            
            res = final_results[key]
            test = res['test']
            
            acc1 = test.get('acc@1', {})
            acc5 = test.get('acc@5', {})
            acc10 = test.get('acc@10', {})
            mrr = test.get('mrr', {})
            
            params_str = f"{res['num_params']:,}" if res['num_params'] else "N/A"
            
            print(f"{model:<15} {params_str:<12} "
                  f"{acc1.get('mean', 0):.2f}±{acc1.get('std', 0):.2f}  "
                  f"{acc5.get('mean', 0):.2f}±{acc5.get('std', 0):.2f}  "
                  f"{acc10.get('mean', 0):.2f}±{acc10.get('std', 0):.2f}  "
                  f"{mrr.get('mean', 0):.2f}±{mrr.get('std', 0):.2f}")


def save_final_results(final_results: Dict[str, Dict], best_configs: Dict[str, Dict]):
    """Save final results to files."""
    # Save as JSON
    json_path = RESULTS_DIR / 'final_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(final_results)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nFinal results saved to: {json_path}")
    
    # Save best configs
    best_configs_path = RESULTS_DIR / 'best_configs.json'
    with open(best_configs_path, 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    print(f"Best configs saved to: {best_configs_path}")
    
    # Generate CSV summary
    csv_path = RESULTS_DIR / 'final_results_summary.csv'
    rows = []
    
    for key, res in final_results.items():
        model, dataset = key.rsplit('_', 1)
        row = {
            'model': model,
            'dataset': dataset,
            'num_params': res['num_params'],
            'num_runs': res['num_runs'],
        }
        
        for split in ['val', 'test']:
            for metric in METRICS:
                if metric in res.get(split, {}):
                    row[f'{split}_{metric}_mean'] = res[split][metric]['mean']
                    row[f'{split}_{metric}_std'] = res[split][metric]['std']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")


def main():
    """Main entry point for final evaluation."""
    print("=" * 60)
    print("FINAL EVALUATION - BEST HYPERPARAMETERS")
    print("=" * 60)
    print(f"Number of runs per config: {NUM_FINAL_RUNS}")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 60)
    
    # Load tuning results
    print("\nLoading hyperparameter tuning results...")
    results_df = load_tuning_results()
    
    if len(results_df) == 0:
        print("ERROR: No tuning results found. Run hyperparameter tuning first.")
        sys.exit(1)
    
    print(f"Loaded {len(results_df)} tuning results")
    
    # Find best configurations
    print("\nFinding best configurations...")
    best_configs = find_best_configs(results_df)
    
    if not best_configs:
        print("ERROR: No valid configurations found.")
        sys.exit(1)
    
    # Run final evaluations
    print("\nRunning final evaluations...")
    final_results = run_final_evaluations(best_configs)
    
    # Generate tables
    generate_results_tables(final_results)
    
    # Save results
    save_final_results(final_results, best_configs)
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

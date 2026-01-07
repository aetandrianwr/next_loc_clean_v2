#!/usr/bin/env python3
"""
Training script to run all 6 model-dataset combinations for h3r8 datasets.

Models: PointerV45, MHSA, LSTM
Datasets: diy_h3r8, geolife_h3r8

Runs all 6 training sessions in parallel with 2-second delays.
"""

import os
import sys
import json
import time
import subprocess
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import glob as glob_module
import yaml

# Configuration
ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config" / "models"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
RESULTS_DIR = Path(__file__).parent

# Training configurations
TRAINING_CONFIGS = [
    # (model_name, dataset, config_file, training_script)
    ("pointer_v45", "diy_h3r8", "config_pointer_v45_diy_h3r8.yaml", "src/training/train_pointer_v45.py"),
    ("pointer_v45", "geolife_h3r8", "config_pointer_v45_geolife_h3r8.yaml", "src/training/train_pointer_v45.py"),
    ("MHSA", "diy_h3r8", "config_MHSA_diy_h3r8.yaml", "src/training/train_MHSA.py"),
    ("MHSA", "geolife_h3r8", "config_MHSA_geolife_h3r8.yaml", "src/training/train_MHSA.py"),
    ("LSTM", "diy_h3r8", "config_LSTM_diy_h3r8.yaml", "src/training/train_LSTM.py"),
    ("LSTM", "geolife_h3r8", "config_LSTM_geolife_h3r8.yaml", "src/training/train_LSTM.py"),
]

def get_timestamp():
    """Get current timestamp in GMT+7."""
    gmt7 = timezone(timedelta(hours=7))
    now = datetime.now(gmt7)
    return now.strftime("%Y%m%d_%H%M%S")


def run_training_parallel(delay_seconds: int = 2) -> List[Tuple[str, str, subprocess.Popen]]:
    """
    Start all 6 training sessions in parallel with delays.
    
    Args:
        delay_seconds: Delay between starting each training session
    
    Returns:
        List of (model_name, dataset, process) tuples
    """
    processes = []
    
    print("=" * 60)
    print("Starting parallel training for h3r8 datasets")
    print("=" * 60)
    
    for i, (model_name, dataset, config_file, train_script) in enumerate(TRAINING_CONFIGS):
        config_path = CONFIG_DIR / config_file
        
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            continue
        
        # Build command
        cmd = [
            sys.executable,
            str(ROOT_DIR / train_script),
            "--config",
            str(config_path)
        ]
        
        print(f"\n[{i+1}/6] Starting: {model_name} on {dataset}")
        print(f"  Config: {config_file}")
        print(f"  Command: python {train_script} --config config/models/{config_file}")
        
        # Start process
        log_file = RESULTS_DIR / f"training_log_{model_name}_{dataset}.txt"
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        
        processes.append((model_name, dataset, process, log_file))
        print(f"  PID: {process.pid}")
        print(f"  Log: {log_file}")
        
        # Delay before next start (except for last one)
        if i < len(TRAINING_CONFIGS) - 1:
            print(f"  Waiting {delay_seconds} seconds before next...")
            time.sleep(delay_seconds)
    
    return processes


def wait_for_all(processes: List) -> Dict[str, Dict]:
    """
    Wait for all training processes to complete.
    
    Returns:
        Dictionary of results for each model-dataset combination
    """
    print("\n" + "=" * 60)
    print("Waiting for all training processes to complete...")
    print("=" * 60)
    
    results = {}
    
    for model_name, dataset, process, log_file in processes:
        key = f"{model_name}_{dataset}"
        print(f"\nWaiting for {key} (PID: {process.pid})...")
        
        return_code = process.wait()
        
        if return_code == 0:
            print(f"  ✓ {key} completed successfully")
            results[key] = {"status": "success", "return_code": return_code}
        else:
            print(f"  ✗ {key} failed with return code {return_code}")
            results[key] = {"status": "failed", "return_code": return_code}
    
    return results


def find_experiment_dir(model_name: str, dataset: str) -> Path:
    """
    Find the latest experiment directory for a model-dataset combination.
    
    Args:
        model_name: Name of the model (pointer_v45, MHSA, LSTM)
        dataset: Name of the dataset (diy_h3r8, geolife_h3r8)
    
    Returns:
        Path to experiment directory
    """
    # Map dataset names to experiment prefixes
    dataset_map = {
        "diy_h3r8": "diy",
        "geolife_h3r8": "geolife"
    }
    dataset_prefix = dataset_map.get(dataset, dataset)
    
    # Search for matching experiment directories
    pattern = f"{dataset_prefix}_{model_name}_*"
    matches = sorted(EXPERIMENTS_DIR.glob(pattern), reverse=True)
    
    if matches:
        return matches[0]
    
    return None


def load_config_hyperparams(config_path: Path) -> Dict:
    """Load hyperparameters from config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def collect_results() -> Tuple[List[Dict], List[Dict]]:
    """
    Collect results from all experiment directories.
    
    Returns:
        Tuple of (validation_results, test_results) lists
    """
    val_results = []
    test_results = []
    
    timestamp = get_timestamp()
    
    for model_name, dataset, config_file, _ in TRAINING_CONFIGS:
        exp_dir = find_experiment_dir(model_name, dataset)
        config_path = CONFIG_DIR / config_file
        
        if exp_dir is None:
            print(f"WARNING: No experiment directory found for {model_name}_{dataset}")
            continue
        
        # Load results
        val_path = exp_dir / "val_results.json"
        test_path = exp_dir / "test_results.json"
        
        if not val_path.exists() or not test_path.exists():
            print(f"WARNING: Results not found in {exp_dir}")
            continue
        
        with open(val_path, "r") as f:
            val_data = json.load(f)
        with open(test_path, "r") as f:
            test_data = json.load(f)
        
        # Load config for hyperparameters
        hyperparams = load_config_hyperparams(config_path)
        
        # Get number of parameters from training log or config
        num_params = 0
        log_file = exp_dir / "training.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                content = f.read()
                if "Parameters:" in content:
                    import re
                    match = re.search(r'Parameters:\s*([\d,]+)', content)
                    if match:
                        num_params = int(match.group(1).replace(",", ""))
                elif "trainable parameters:" in content:
                    match = re.search(r'trainable parameters:\s*([\d,]+)', content)
                    if match:
                        num_params = int(match.group(1).replace(",", ""))
        
        # Extract trial_idx from config file name
        trial_idx = 0  # Default
        
        # Common fields
        common_fields = {
            "config_name": config_file.replace(".yaml", ""),
            "model_name": model_name,
            "dataset": dataset,
            "trial_idx": trial_idx,
            "num_params": num_params,
            "experiment_dir": str(exp_dir),
            "config_path": str(config_path),
            "hyperparameters": json.dumps(hyperparams),
            "status": "success",
            "timestamp": timestamp,
        }
        
        # Validation result
        val_row = {
            **common_fields,
            "correct_at_1": val_data.get("correct@1", 0),
            "correct_at_3": val_data.get("correct@3", 0),
            "correct_at_5": val_data.get("correct@5", 0),
            "correct_at_10": val_data.get("correct@10", 0),
            "total": val_data.get("total", 0),
            "rr": val_data.get("rr", 0),
            "ndcg": val_data.get("ndcg", 0),
            "f1": val_data.get("f1", 0),
            "acc_at_1": val_data.get("acc@1", 0),
            "acc_at_5": val_data.get("acc@5", 0),
            "acc_at_10": val_data.get("acc@10", 0),
            "mrr": val_data.get("mrr", 0),
            "loss": val_data.get("loss", 0),
        }
        val_results.append(val_row)
        
        # Test result
        test_row = {
            **common_fields,
            "correct_at_1": test_data.get("correct@1", 0),
            "correct_at_3": test_data.get("correct@3", 0),
            "correct_at_5": test_data.get("correct@5", 0),
            "correct_at_10": test_data.get("correct@10", 0),
            "total": test_data.get("total", 0),
            "rr": test_data.get("rr", 0),
            "ndcg": test_data.get("ndcg", 0),
            "f1": test_data.get("f1", 0),
            "acc_at_1": test_data.get("acc@1", 0),
            "acc_at_5": test_data.get("acc@5", 0),
            "acc_at_10": test_data.get("acc@10", 0),
            "mrr": test_data.get("mrr", 0),
            "loss": test_data.get("loss", 0),
        }
        test_results.append(test_row)
    
    return val_results, test_results


def save_results_csv(val_results: List[Dict], test_results: List[Dict]):
    """Save results to CSV files."""
    columns = [
        "config_name", "model_name", "dataset", "trial_idx", "num_params",
        "correct_at_1", "correct_at_3", "correct_at_5", "correct_at_10", "total",
        "rr", "ndcg", "f1", "acc_at_1", "acc_at_5", "acc_at_10", "mrr", "loss",
        "experiment_dir", "config_path", "hyperparameters", "status", "timestamp"
    ]
    
    # Save validation results
    val_csv_path = RESULTS_DIR / "h3r8_validation_results.csv"
    with open(val_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(val_results)
    print(f"Saved validation results to: {val_csv_path}")
    
    # Save test results
    test_csv_path = RESULTS_DIR / "h3r8_test_results.csv"
    with open(test_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(test_results)
    print(f"Saved test results to: {test_csv_path}")
    
    return val_csv_path, test_csv_path


def generate_summary_report(val_results: List[Dict], test_results: List[Dict]):
    """Generate a summary report."""
    report_path = RESULTS_DIR / "training_summary.txt"
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("H3R8 Dataset Training Summary Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Training Date: {}\n\n".format(get_timestamp()))
        
        f.write("-" * 80 + "\n")
        f.write("TEST RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Dataset':<15} {'Acc@1':<10} {'Acc@5':<10} {'Acc@10':<10} {'MRR':<10} {'NDCG':<10} {'F1':<10}\n")
        f.write("-" * 80 + "\n")
        
        for result in test_results:
            f.write(f"{result['model_name']:<15} {result['dataset']:<15} "
                   f"{result['acc_at_1']:<10.2f} {result['acc_at_5']:<10.2f} "
                   f"{result['acc_at_10']:<10.2f} {result['mrr']:<10.2f} "
                   f"{result['ndcg']:<10.2f} {result['f1']*100:<10.2f}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("VALIDATION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Dataset':<15} {'Acc@1':<10} {'Acc@5':<10} {'Acc@10':<10} {'MRR':<10} {'NDCG':<10} {'F1':<10}\n")
        f.write("-" * 80 + "\n")
        
        for result in val_results:
            f.write(f"{result['model_name']:<15} {result['dataset']:<15} "
                   f"{result['acc_at_1']:<10.2f} {result['acc_at_5']:<10.2f} "
                   f"{result['acc_at_10']:<10.2f} {result['mrr']:<10.2f} "
                   f"{result['ndcg']:<10.2f} {result['f1']*100:<10.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Saved summary report to: {report_path}")
    return report_path


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("H3R8 Dataset Training Script")
    print("Models: PointerV45, MHSA, LSTM")
    print("Datasets: diy_h3r8, geolife_h3r8")
    print("=" * 60 + "\n")
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start all training processes
    processes = run_training_parallel(delay_seconds=2)
    
    # Wait for all to complete
    completion_results = wait_for_all(processes)
    
    # Collect results
    print("\n" + "=" * 60)
    print("Collecting results...")
    print("=" * 60)
    
    val_results, test_results = collect_results()
    
    # Save to CSV
    val_csv, test_csv = save_results_csv(val_results, test_results)
    
    # Generate summary
    summary_path = generate_summary_report(val_results, test_results)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  - {val_csv}")
    print(f"  - {test_csv}")
    print(f"  - {summary_path}")
    
    print("\n" + "-" * 60)
    print("TEST RESULTS SUMMARY")
    print("-" * 60)
    print(f"{'Model':<15} {'Dataset':<15} {'Acc@1':<10} {'MRR':<10} {'NDCG':<10}")
    print("-" * 60)
    for result in test_results:
        print(f"{result['model_name']:<15} {result['dataset']:<15} "
              f"{result['acc_at_1']:<10.2f} {result['mrr']:<10.2f} {result['ndcg']:<10.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

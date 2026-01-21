#!/usr/bin/env python3
"""
Automation script for training, evaluating, and tuning Pointer Generator Transformer model
across different previous days configurations.

This script:
1. Runs training sessions in parallel (3 at a time)
2. Appends results to CSV files after each training completes
3. Supports both Geolife and DIY datasets

Usage:
    python scripts/run_prev_days_analysis.py --max_epochs 2 --test  # Test mode
    python scripts/run_prev_days_analysis.py --max_epochs 50        # Full training
"""

import os
import sys
import json
import csv
import yaml
import argparse
import subprocess
import time
import threading
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Configuration definitions
PREV_DAYS = [3, 7, 10, 14, 17, 21]

GEOLIFE_CONFIGS = {
    "baseline_d64_L2": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "learning_rate": 6.5e-4},
    "d80_L3_deeper": {"d_model": 80, "nhead": 4, "num_layers": 3, "dim_feedforward": 160, "learning_rate": 6.0e-4},
    "d64_L2_ff192_highLR": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 192, "learning_rate": 8.0e-4},
    "d64_L3_lowLR_highDrop": {"d_model": 64, "nhead": 4, "num_layers": 3, "dim_feedforward": 128, "learning_rate": 5.0e-4},
    "d64_L2_lowDropout": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "learning_rate": 6.0e-4},
    "d72_L2": {"d_model": 72, "nhead": 4, "num_layers": 2, "dim_feedforward": 144, "learning_rate": 6.5e-4},
}

DIY_CONFIGS = {
    "baseline_d128_L3": {"d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "learning_rate": 7.0e-4},
    "d128_L4_deeper": {"d_model": 128, "nhead": 4, "num_layers": 4, "dim_feedforward": 256, "learning_rate": 6.0e-4},
    "d128_L3_highLR": {"d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "learning_rate": 9.0e-4},
    "d144_L3_largerEmb": {"d_model": 144, "nhead": 4, "num_layers": 3, "dim_feedforward": 288, "learning_rate": 7.0e-4},
    "d128_L3_lowerLR": {"d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "learning_rate": 6.0e-4},
}

# CSV headers
CSV_HEADERS = [
    "config_name", "dataset", "prev_days", 
    "d_model", "nhead", "num_layers", "dim_feedforward", "learning_rate",
    "num_params", "experiment_dir",
    "correct@1", "correct@3", "correct@5", "correct@10", "total",
    "rr", "ndcg", "f1", "acc@1", "acc@5", "acc@10", "mrr", "loss"
]

# Thread-safe lock for CSV writing
csv_lock = threading.Lock()


def generate_config_file(dataset_type, prev_days, config_name, config_params, 
                         num_epochs, patience, seed, output_dir):
    """Generate a config YAML file and return its path."""
    if dataset_type == "geolife":
        data_dir = "data/geolife_eps20/processed"
        dataset_prefix = f"geolife_eps20_prev{prev_days}"
    else:
        data_dir = "data/diy_eps50/processed"
        dataset_prefix = f"diy_eps50_prev{prev_days}"
    
    config = {
        "seed": seed,
        "data": {
            "data_dir": data_dir,
            "dataset_prefix": dataset_prefix,
            "dataset": dataset_type,
            "experiment_root": "experiments",
            "num_workers": 0,
        },
        "model": {
            "d_model": config_params["d_model"],
            "nhead": config_params["nhead"],
            "num_layers": config_params["num_layers"],
            "dim_feedforward": config_params["dim_feedforward"],
            "dropout": 0.15,
        },
        "training": {
            "batch_size": 128,
            "num_epochs": num_epochs,
            "learning_rate": config_params["learning_rate"],
            "weight_decay": 0.015,
            "label_smoothing": 0.03,
            "grad_clip": 0.8,
            "patience": patience,
            "min_epochs": 8,
            "warmup_epochs": 5,
            "use_amp": True,
            "min_lr": 1e-6,
        },
    }
    
    filename = f"{dataset_type}_prev{prev_days}_{config_name}.yaml"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return filepath


def run_training(config_path, job_id, conda_env="mlenv"):
    """Run a single training session using the wrapper script with unique job ID."""
    cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && cd {PROJECT_ROOT} && python scripts/train_single_job.py --config {config_path} --job_id {job_id}"
    
    print(f"[START] Job {job_id}: {config_path}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"[ERROR] Job {job_id} failed")
            print(f"STDERR: {result.stderr[-1000:]}")
            return None, None
        
        # Extract experiment directory from output
        exp_dir = None
        for line in result.stdout.split("\n"):
            if "Experiment directory:" in line:
                exp_dir = line.split("Experiment directory:")[-1].strip()
                break
            if "Results saved to:" in line:
                exp_dir = line.split("Results saved to:")[-1].strip()
                break
        
        print(f"[DONE] Job {job_id} completed in {elapsed:.1f}s -> {exp_dir}")
        return exp_dir, result.stdout
        
    except Exception as e:
        print(f"[EXCEPTION] Job {job_id}: {str(e)}")
        return None, None


def extract_num_params(stdout):
    """Extract number of parameters from training output."""
    for line in stdout.split("\n"):
        if "Parameters:" in line:
            try:
                param_str = line.split("Parameters:")[-1].strip().replace(",", "")
                return int(param_str)
            except:
                pass
    return 0


def append_to_csv(csv_path, row_data):
    """Thread-safe append to CSV file."""
    with csv_lock:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)


def process_training_result(config_info, exp_dir, stdout, val_csv_path, test_csv_path):
    """Process training result and append to CSV files."""
    if exp_dir is None:
        print(f"[SKIP] No experiment directory for {config_info['config_name']}")
        return False
    
    # Load results
    val_results_path = os.path.join(exp_dir, "val_results.json")
    test_results_path = os.path.join(exp_dir, "test_results.json")
    
    if not os.path.exists(val_results_path) or not os.path.exists(test_results_path):
        print(f"[ERROR] Results files not found in {exp_dir}")
        return False
    
    with open(val_results_path, "r") as f:
        val_results = json.load(f)
    
    with open(test_results_path, "r") as f:
        test_results = json.load(f)
    
    # Get num_params from results file (now saved there)
    num_params = val_results.get('num_params', 0)
    if num_params == 0 and stdout:
        num_params = extract_num_params(stdout)
    
    # Common fields
    base_row = {
        "config_name": config_info["config_name"],
        "dataset": config_info["dataset"],
        "prev_days": config_info["prev_days"],
        "d_model": config_info["d_model"],
        "nhead": config_info["nhead"],
        "num_layers": config_info["num_layers"],
        "dim_feedforward": config_info["dim_feedforward"],
        "learning_rate": config_info["learning_rate"],
        "num_params": num_params,
        "experiment_dir": exp_dir,
    }
    
    # Validation results
    val_row = {**base_row}
    for key in ["correct@1", "correct@3", "correct@5", "correct@10", "total", 
                "rr", "ndcg", "f1", "acc@1", "acc@5", "acc@10", "mrr", "loss"]:
        val_row[key] = val_results.get(key, 0)
    
    # Test results
    test_row = {**base_row}
    for key in ["correct@1", "correct@3", "correct@5", "correct@10", "total", 
                "rr", "ndcg", "f1", "acc@1", "acc@5", "acc@10", "mrr", "loss"]:
        test_row[key] = test_results.get(key, 0)
    
    # Append to CSVs
    append_to_csv(val_csv_path, val_row)
    append_to_csv(test_csv_path, test_row)
    
    print(f"[CSV] Appended results for {config_info['config_name']} (prev{config_info['prev_days']})")
    return True


def run_single_job(job):
    """Run a single training job."""
    config_path, config_info, val_csv_path, test_csv_path, job_id = job
    exp_dir, stdout = run_training(config_path, job_id)
    success = process_training_result(config_info, exp_dir, stdout, val_csv_path, test_csv_path)
    return config_info["config_name"], success


def main():
    parser = argparse.ArgumentParser(description="Run previous days analysis")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs for training")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--parallel", type=int, default=3, help="Number of parallel training sessions")
    parser.add_argument("--test", action="store_true", help="Test mode (2 epochs)")
    args = parser.parse_args()
    
    if args.test:
        args.max_epochs = 2
        print("=" * 60)
        print("TEST MODE: Running with max_epochs=2")
        print("=" * 60)
    
    # Setup paths
    config_dir = PROJECT_ROOT / "config" / "analysis_prev_days"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped CSV files
    gmt7 = timezone(timedelta(hours=7))
    timestamp = datetime.now(gmt7).strftime("%Y%m%d_%H%M%S")
    val_csv_path = results_dir / f"prev_days_analysis_val_{timestamp}.csv"
    test_csv_path = results_dir / f"prev_days_analysis_test_{timestamp}.csv"
    
    print(f"Validation CSV: {val_csv_path}")
    print(f"Test CSV: {test_csv_path}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Patience: {args.patience}")
    print(f"Parallel jobs: {args.parallel}")
    
    # Build job list with unique job IDs
    jobs = []
    job_counter = 0
    
    # Geolife jobs
    for prev_days in PREV_DAYS:
        for config_name, config_params in GEOLIFE_CONFIGS.items():
            config_path = generate_config_file(
                "geolife", prev_days, config_name, config_params,
                args.max_epochs, args.patience, args.seed, str(config_dir)
            )
            config_info = {
                "config_name": config_name,
                "dataset": "geolife",
                "prev_days": prev_days,
                **config_params,
            }
            # Create unique job ID: dataset_prevDays_configName
            job_id = f"geo_p{prev_days}_{config_name}"
            jobs.append((config_path, config_info, str(val_csv_path), str(test_csv_path), job_id))
            job_counter += 1
    
    # DIY jobs
    for prev_days in PREV_DAYS:
        for config_name, config_params in DIY_CONFIGS.items():
            config_path = generate_config_file(
                "diy", prev_days, config_name, config_params,
                args.max_epochs, args.patience, args.seed, str(config_dir)
            )
            config_info = {
                "config_name": config_name,
                "dataset": "diy",
                "prev_days": prev_days,
                **config_params,
            }
            # Create unique job ID
            job_id = f"diy_p{prev_days}_{config_name}"
            jobs.append((config_path, config_info, str(val_csv_path), str(test_csv_path), job_id))
            job_counter += 1
    
    print(f"\nTotal jobs: {len(jobs)}")
    print(f"  Geolife: {len(PREV_DAYS) * len(GEOLIFE_CONFIGS)}")
    print(f"  DIY: {len(PREV_DAYS) * len(DIY_CONFIGS)}")
    print("=" * 60)
    
    # Run jobs in parallel
    completed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_single_job, job): job for job in jobs}
        
        for future in as_completed(futures):
            job = futures[future]
            try:
                config_name, success = future.result()
                completed += 1
                if not success:
                    failed += 1
                print(f"[PROGRESS] {completed}/{len(jobs)} completed, {failed} failed")
            except Exception as e:
                failed += 1
                completed += 1
                print(f"[EXCEPTION] Job failed: {str(e)}")
    
    print("=" * 60)
    print(f"COMPLETED: {completed}/{len(jobs)} jobs")
    print(f"FAILED: {failed} jobs")
    print(f"Validation results: {val_csv_path}")
    print(f"Test results: {test_csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

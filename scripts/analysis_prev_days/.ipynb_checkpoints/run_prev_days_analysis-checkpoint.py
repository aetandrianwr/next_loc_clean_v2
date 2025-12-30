#!/usr/bin/env python3
"""
Automation script for training, evaluating, and tuning PointerNetworkV45
across different previous days configurations.

This script:
1. Generates config files for each combination of (previous_days, dataset, tuning_config)
2. Runs training sessions in parallel (3 at a time with 5s delay)
3. Collects results and appends to val_results.csv and test_results.csv

Usage:
    python scripts/analysis_prev_days/run_prev_days_analysis.py --max_epochs 2 --dry_run  # Test
    python scripts/analysis_prev_days/run_prev_days_analysis.py --max_epochs 50          # Full run
"""

import os
import sys

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
import yaml
import json
import csv
import time
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import fcntl

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "analysis_prev_days"
RESULTS_DIR = PROJECT_ROOT / "results" / "analysis_prev_days"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Previous days to test
PREV_DAYS = [3, 7, 10, 14, 17, 21]

# GeoLife tuning configurations (6 configs)
GEOLIFE_CONFIGS = {
    "baseline_d64_L2": {
        "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "learning_rate": 6.5e-4
    },
    "d80_L3_deeper": {
        "d_model": 80, "nhead": 4, "num_layers": 3, "dim_feedforward": 160, "learning_rate": 6.0e-4
    },
    "d64_L2_ff192_highLR": {
        "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 192, "learning_rate": 8.0e-4
    },
    "d64_L3_lowLR_highDrop": {
        "d_model": 64, "nhead": 4, "num_layers": 3, "dim_feedforward": 128, "learning_rate": 5.0e-4
    },
    "d64_L2_lowDropout": {
        "d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "learning_rate": 6.0e-4
    },
    "d72_L2": {
        "d_model": 72, "nhead": 4, "num_layers": 2, "dim_feedforward": 144, "learning_rate": 6.5e-4
    },
}

# DIY tuning configurations (5 configs)
DIY_CONFIGS = {
    "baseline_d128_L3": {
        "d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "learning_rate": 7.0e-4
    },
    "d128_L4_deeper": {
        "d_model": 128, "nhead": 4, "num_layers": 4, "dim_feedforward": 256, "learning_rate": 6.0e-4
    },
    "d128_L3_highLR": {
        "d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "learning_rate": 9.0e-4
    },
    "d144_L3_largerEmb": {
        "d_model": 144, "nhead": 4, "num_layers": 3, "dim_feedforward": 288, "learning_rate": 7.0e-4
    },
    "d128_L3_lowerLR": {
        "d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "learning_rate": 6.0e-4
    },
}

# Lock for thread-safe CSV writing
csv_lock = threading.Lock()


def generate_config(dataset: str, prev_days: int, config_name: str, config_params: dict, 
                    max_epochs: int, seed: int = 42, patience: int = 5) -> str:
    """Generate a YAML config file for a specific training run."""
    
    if dataset == "geolife":
        data_dir = "data/geolife_eps20/processed"
        dataset_prefix = f"geolife_eps20_prev{prev_days}"
    else:  # diy
        data_dir = "data/diy_eps50/processed"
        dataset_prefix = f"diy_eps50_prev{prev_days}"
    
    config = {
        "seed": seed,
        "data": {
            "data_dir": data_dir,
            "dataset_prefix": dataset_prefix,
            "dataset": dataset,
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
            "num_epochs": max_epochs,
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
    
    # Create config file
    config_filename = f"{dataset}_prev{prev_days}_{config_name}.yaml"
    config_path = CONFIG_DIR / config_filename
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_path)


def get_all_training_jobs(max_epochs: int) -> list:
    """Generate all training job configurations."""
    jobs = []
    
    for prev_days in PREV_DAYS:
        # GeoLife jobs
        for config_name, config_params in GEOLIFE_CONFIGS.items():
            jobs.append({
                "dataset": "geolife",
                "prev_days": prev_days,
                "config_name": config_name,
                "config_params": config_params,
                "max_epochs": max_epochs,
            })
        
        # DIY jobs
        for config_name, config_params in DIY_CONFIGS.items():
            jobs.append({
                "dataset": "diy",
                "prev_days": prev_days,
                "config_name": config_name,
                "config_params": config_params,
                "max_epochs": max_epochs,
            })
    
    return jobs


def run_training(job: dict, job_id: int, conda_env: str = "mlenv") -> dict:
    """Run a single training job and return results."""
    
    dataset = job["dataset"]
    prev_days = job["prev_days"]
    config_name = job["config_name"]
    config_params = job["config_params"]
    max_epochs = job["max_epochs"]
    
    print(f"[Job {job_id}] Starting: {dataset} prev{prev_days} {config_name}")
    
    # Generate config file
    config_path = generate_config(
        dataset=dataset,
        prev_days=prev_days,
        config_name=config_name,
        config_params=config_params,
        max_epochs=max_epochs,
        seed=42,
        patience=5,
    )
    
    # Build command
    train_script = str(PROJECT_ROOT / "src" / "training" / "train_pointer_v45.py")
    cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && python {train_script} --config {config_path}"
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per job
        )
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
    except subprocess.TimeoutExpired:
        success = False
        output = ""
        error = "Training timeout exceeded"
    except Exception as e:
        success = False
        output = ""
        error = str(e)
    
    elapsed_time = time.time() - start_time
    
    # Find experiment directory from output
    experiment_dir = None
    if success:
        for line in output.split("\n"):
            if "Experiment directory:" in line:
                experiment_dir = line.split("Experiment directory:")[-1].strip()
                break
            elif "Results saved to:" in line:
                experiment_dir = line.split("Results saved to:")[-1].strip()
                break
    
    # Parse results
    result_data = {
        "dataset": dataset,
        "prev_days": prev_days,
        "config_name": config_name,
        "d_model": config_params["d_model"],
        "nhead": config_params["nhead"],
        "num_layers": config_params["num_layers"],
        "dim_feedforward": config_params["dim_feedforward"],
        "learning_rate": config_params["learning_rate"],
        "experiment_dir": experiment_dir,
        "success": success,
        "elapsed_time": elapsed_time,
        "error": error if not success else "",
    }
    
    # Read metrics from experiment directory
    if experiment_dir and os.path.exists(experiment_dir):
        val_results_path = os.path.join(experiment_dir, "val_results.json")
        test_results_path = os.path.join(experiment_dir, "test_results.json")
        
        # Read num_params from training log
        num_params = 0
        log_path = os.path.join(experiment_dir, "training.log")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    if "Model parameters:" in line or "Parameters:" in line:
                        try:
                            num_params = int(line.split(":")[-1].strip().replace(",", ""))
                        except:
                            pass
                        break
        
        result_data["num_params"] = num_params
        
        if os.path.exists(val_results_path):
            with open(val_results_path, "r") as f:
                val_metrics = json.load(f)
            result_data["val_metrics"] = val_metrics
        
        if os.path.exists(test_results_path):
            with open(test_results_path, "r") as f:
                test_metrics = json.load(f)
            result_data["test_metrics"] = test_metrics
    
    status = "SUCCESS" if success else "FAILED"
    print(f"[Job {job_id}] {status}: {dataset} prev{prev_days} {config_name} ({elapsed_time:.1f}s)")
    
    return result_data


def append_to_csv(result: dict, val_csv_path: str, test_csv_path: str):
    """Append a result to both val and test CSV files."""
    
    csv_columns = [
        "dataset", "prev_days", "config_name", 
        "d_model", "nhead", "num_layers", "dim_feedforward", "learning_rate",
        "num_params", "experiment_dir",
        "correct@1", "correct@3", "correct@5", "correct@10", "total",
        "rr", "ndcg", "f1", "acc@1", "acc@5", "acc@10", "mrr", "loss"
    ]
    
    def build_row(result: dict, metrics: dict) -> dict:
        row = {
            "dataset": result["dataset"],
            "prev_days": result["prev_days"],
            "config_name": result["config_name"],
            "d_model": result["d_model"],
            "nhead": result["nhead"],
            "num_layers": result["num_layers"],
            "dim_feedforward": result["dim_feedforward"],
            "learning_rate": result["learning_rate"],
            "num_params": result.get("num_params", ""),
            "experiment_dir": result.get("experiment_dir", ""),
        }
        
        if metrics:
            for key in ["correct@1", "correct@3", "correct@5", "correct@10", "total",
                        "rr", "ndcg", "f1", "acc@1", "acc@5", "acc@10", "mrr", "loss"]:
                row[key] = metrics.get(key, "")
        else:
            for key in ["correct@1", "correct@3", "correct@5", "correct@10", "total",
                        "rr", "ndcg", "f1", "acc@1", "acc@5", "acc@10", "mrr", "loss"]:
                row[key] = ""
        
        return row
    
    with csv_lock:
        # Write to val CSV
        val_exists = os.path.exists(val_csv_path)
        with open(val_csv_path, "a", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            if not val_exists:
                writer.writeheader()
            val_row = build_row(result, result.get("val_metrics", {}))
            writer.writerow(val_row)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        # Write to test CSV
        test_exists = os.path.exists(test_csv_path)
        with open(test_csv_path, "a", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            if not test_exists:
                writer.writeheader()
            test_row = build_row(result, result.get("test_metrics", {}))
            writer.writerow(test_row)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def run_worker(job_queue: Queue, worker_id: int, val_csv_path: str, test_csv_path: str, 
               conda_env: str, start_delay: float):
    """Worker thread that processes jobs from the queue."""
    
    # Apply staggered start delay
    time.sleep(start_delay)
    
    while True:
        try:
            job_id, job = job_queue.get(timeout=1)
        except:
            break
        
        if job is None:
            break
        
        result = run_training(job, job_id, conda_env)
        append_to_csv(result, val_csv_path, test_csv_path)
        job_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Run prev_days analysis for Pointer V45")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs per training")
    parser.add_argument("--parallel", type=int, default=3, help="Number of parallel training sessions")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between parallel starts (seconds)")
    parser.add_argument("--conda_env", type=str, default="mlenv", help="Conda environment name")
    parser.add_argument("--dry_run", action="store_true", help="Only generate configs, don't train")
    parser.add_argument("--output_prefix", type=str, default="prev_days_analysis", 
                        help="Prefix for output CSV files")
    args = parser.parse_args()
    
    # Create directories
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all jobs
    jobs = get_all_training_jobs(args.max_epochs)
    total_jobs = len(jobs)
    
    print("=" * 60)
    print("PREV DAYS ANALYSIS - PointerNetworkV45")
    print("=" * 60)
    print(f"Total jobs: {total_jobs}")
    print(f"Previous days: {PREV_DAYS}")
    print(f"GeoLife configs: {len(GEOLIFE_CONFIGS)}")
    print(f"DIY configs: {len(DIY_CONFIGS)}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Start delay: {args.delay}s")
    print("=" * 60)
    
    if args.dry_run:
        print("\nDRY RUN - Generating configs only...")
        for job in jobs:
            config_path = generate_config(
                dataset=job["dataset"],
                prev_days=job["prev_days"],
                config_name=job["config_name"],
                config_params=job["config_params"],
                max_epochs=args.max_epochs,
            )
            print(f"  Generated: {config_path}")
        print(f"\nGenerated {total_jobs} config files in {CONFIG_DIR}")
        return
    
    # Setup output CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    val_csv_path = str(RESULTS_DIR / f"{args.output_prefix}_val_{timestamp}.csv")
    test_csv_path = str(RESULTS_DIR / f"{args.output_prefix}_test_{timestamp}.csv")
    
    print(f"\nVal results: {val_csv_path}")
    print(f"Test results: {test_csv_path}")
    print("\nStarting training...\n")
    
    # Create job queue
    job_queue = Queue()
    for i, job in enumerate(jobs):
        job_queue.put((i + 1, job))
    
    # Add sentinel values to stop workers
    for _ in range(args.parallel):
        job_queue.put((None, None))
    
    # Start workers with staggered delays
    workers = []
    for i in range(args.parallel):
        worker = threading.Thread(
            target=run_worker,
            args=(job_queue, i, val_csv_path, test_csv_path, args.conda_env, i * args.delay),
        )
        worker.start()
        workers.append(worker)
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Val results saved to: {val_csv_path}")
    print(f"Test results saved to: {test_csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hyperparameter Tuning via Grid Search for Next Location Prediction Models.

This script performs grid search hyperparameter tuning for:
1. Pointer V45 (Proposed Model)
2. MHSA (Multi-Head Self-Attention)
3. LSTM

Models are trained in order: PointerV45 -> MHSA -> LSTM
Each model is tuned for both DIY and GeoLife datasets with different prev_days.

Features:
- 5 parallel training sessions with balanced workload
- 5-second delay between parallel job launches
- CSV logging of results (val and test)
- Config files stored in config/ht_grid_search/

Usage:
    python scripts/ht_grid_search/run_grid_search.py --max_epochs 2  # Test run
    python scripts/ht_grid_search/run_grid_search.py --max_epochs 50 # Full run
"""

import os
import sys
import json
import yaml
import itertools
import subprocess
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any
import threading
import queue
from dataclasses import dataclass
import copy

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
CONFIG_HT_DIR = PROJECT_ROOT / "config" / "ht_grid_search"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "ht_grid_search" / "results"


@dataclass
class TrainingJob:
    """Represents a single training job."""
    model_type: str  # pointer_v45, MHSA, LSTM
    dataset: str     # diy, geolife
    prev_days: int
    params: Dict[str, Any]
    config_path: str
    weight_score: int  # For balancing workload (higher = heavier)


def get_timestamp():
    """Get current timestamp in GMT+7."""
    gmt7 = timezone(timedelta(hours=7))
    return datetime.now(gmt7).strftime("%Y%m%d_%H%M%S")


def load_metadata(dataset: str, prev_days: int) -> Dict:
    """Load metadata for a dataset."""
    if dataset == "diy":
        metadata_path = PROJECT_ROOT / "data" / "diy_eps50" / "processed" / f"diy_eps50_prev{prev_days}_metadata.json"
    else:
        metadata_path = PROJECT_ROOT / "data" / "geolife_eps20" / "processed" / f"geolife_eps20_prev{prev_days}_metadata.json"
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def create_pointer_v45_config(
    dataset: str, 
    prev_days: int, 
    d_model: int, 
    n_head: int, 
    num_layer: int, 
    ff_dim: int, 
    lr: float,
    max_epochs: int,
    seed: int = 42,
    patience: int = 5
) -> Tuple[str, Dict]:
    """Create config for Pointer V45 model."""
    metadata = load_metadata(dataset, prev_days)
    
    if dataset == "diy":
        data_dir = "data/diy_eps50/processed"
        dataset_prefix = f"diy_eps50_prev{prev_days}"
    else:
        data_dir = "data/geolife_eps20/processed"
        dataset_prefix = f"geolife_eps20_prev{prev_days}"
    
    config = {
        "seed": seed,
        "data": {
            "data_dir": data_dir,
            "dataset_prefix": dataset_prefix,
            "dataset": dataset,
            "experiment_root": "experiments",
            "num_workers": 0
        },
        "model": {
            "d_model": d_model,
            "nhead": n_head,
            "num_layers": num_layer,
            "dim_feedforward": ff_dim,
            "dropout": 0.15
        },
        "training": {
            "batch_size": 128,
            "num_epochs": max_epochs,
            "learning_rate": lr,
            "weight_decay": 0.015,
            "label_smoothing": 0.03,
            "grad_clip": 0.8,
            "patience": patience,
            "min_epochs": 8,
            "warmup_epochs": 5,
            "use_amp": True,
            "min_lr": 0.000001
        }
    }
    
    # Generate unique config filename
    config_name = f"pointer_v45_{dataset}_prev{prev_days}_d{d_model}_h{n_head}_l{num_layer}_ff{ff_dim}_lr{lr}.yaml"
    config_path = CONFIG_HT_DIR / config_name
    
    return str(config_path), config


def create_mhsa_config(
    dataset: str, 
    prev_days: int, 
    base_emb_size: int, 
    n_head: int, 
    num_layer: int, 
    ff_dim: int, 
    lr: float,
    max_epochs: int,
    seed: int = 42,
    patience: int = 5
) -> Tuple[str, Dict]:
    """Create config for MHSA model."""
    metadata = load_metadata(dataset, prev_days)
    
    if dataset == "diy":
        data_dir = "data/diy_eps50/processed"
        dataset_prefix = f"diy_eps50_prev{prev_days}"
        batch_size = 256
    else:
        data_dir = "data/geolife_eps20/processed"
        dataset_prefix = f"geolife_eps20_prev{prev_days}"
        batch_size = 32
    
    config = {
        "seed": seed,
        "data": {
            "data_dir": data_dir,
            "dataset_prefix": dataset_prefix,
            "dataset": dataset,
            "experiment_root": "experiments"
        },
        "training": {
            "if_embed_user": True,
            "if_embed_poi": False,
            "if_embed_time": True,
            "if_embed_duration": True,
            "previous_day": prev_days,
            "verbose": True,
            "debug": False,
            "batch_size": batch_size,
            "print_step": 20 if dataset == "geolife" else 10,
            "num_workers": 0,
            "day_selection": "default"
        },
        "dataset_info": {
            "total_loc_num": metadata["total_loc_num"],
            "total_user_num": metadata["total_user_num"]
        },
        "embedding": {
            "base_emb_size": base_emb_size,
            "poi_original_size": 16
        },
        "model": {
            "networkName": "transformer",
            "num_encoder_layers": num_layer,
            "nhead": n_head,
            "dim_feedforward": ff_dim,
            "fc_dropout": 0.2 if dataset == "geolife" else 0.1
        },
        "optimiser": {
            "optimizer": "Adam",
            "max_epoch": max_epochs,
            "lr": lr,
            "weight_decay": 0.000001,
            "beta1": 0.9,
            "beta2": 0.999,
            "momentum": 0.98,
            "num_warmup_epochs": 2,
            "num_training_epochs": max_epochs,
            "patience": patience,
            "lr_step_size": 1,
            "lr_gamma": 0.1
        }
    }
    
    config_name = f"MHSA_{dataset}_prev{prev_days}_emb{base_emb_size}_h{n_head}_l{num_layer}_ff{ff_dim}_lr{lr}.yaml"
    config_path = CONFIG_HT_DIR / config_name
    
    return str(config_path), config


def create_lstm_config(
    dataset: str, 
    prev_days: int, 
    emb_size: int, 
    hidden_size: int, 
    num_layer: int, 
    ff_dim: int, 
    lr: float,
    max_epochs: int,
    seed: int = 42,
    patience: int = 5
) -> Tuple[str, Dict]:
    """Create config for LSTM model."""
    metadata = load_metadata(dataset, prev_days)
    
    if dataset == "diy":
        data_dir = "data/diy_eps50/processed"
        dataset_prefix = f"diy_eps50_prev{prev_days}"
        batch_size = 256
    else:
        data_dir = "data/geolife_eps20/processed"
        dataset_prefix = f"geolife_eps20_prev{prev_days}"
        batch_size = 32
    
    config = {
        "seed": seed,
        "data": {
            "data_dir": data_dir,
            "dataset_prefix": dataset_prefix,
            "dataset": dataset,
            "experiment_root": "experiments"
        },
        "training": {
            "if_embed_user": True,
            "if_embed_poi": False,
            "if_embed_time": True,
            "if_embed_duration": True,
            "previous_day": prev_days,
            "verbose": True,
            "debug": False,
            "batch_size": batch_size,
            "print_step": 20 if dataset == "geolife" else 10,
            "num_workers": 0,
            "day_selection": "default"
        },
        "dataset_info": {
            "total_loc_num": metadata["total_loc_num"],
            "total_user_num": metadata["total_user_num"]
        },
        "embedding": {
            "base_emb_size": emb_size,
            "poi_original_size": 16
        },
        "model": {
            "networkName": "lstm",
            "lstm_hidden_size": hidden_size,
            "lstm_num_layers": num_layer,
            "lstm_dropout": 0.2,
            "fc_dropout": 0.2 if dataset == "geolife" else 0.1
        },
        "optimiser": {
            "optimizer": "Adam",
            "max_epoch": max_epochs,
            "lr": lr,
            "weight_decay": 0.000001,
            "beta1": 0.9,
            "beta2": 0.999,
            "momentum": 0.98,
            "num_warmup_epochs": 2,
            "num_training_epochs": max_epochs,
            "patience": patience,
            "lr_step_size": 1,
            "lr_gamma": 0.1
        }
    }
    
    config_name = f"LSTM_{dataset}_prev{prev_days}_emb{emb_size}_h{hidden_size}_l{num_layer}_ff{ff_dim}_lr{lr}.yaml"
    config_path = CONFIG_HT_DIR / config_name
    
    return str(config_path), config


def calculate_weight_score(model_type: str, params: Dict) -> int:
    """
    Calculate weight score for balancing workload.
    Higher score = heavier computation.
    """
    if model_type == "pointer_v45":
        # d_model * n_head * num_layer * ff_dim approximates model size
        score = params["d_model"] * params["num_layer"] * params["ff_dim"] // 1000
    elif model_type == "MHSA":
        score = params["base_emb_size"] * params["num_layer"] * params["ff_dim"] // 1000
    else:  # LSTM
        score = params["emb_size"] * params["hidden_size"] * params["num_layer"] * params["ff_dim"] // 10000
    
    return max(1, score)


def generate_all_jobs(max_epochs: int, seed: int = 42, patience: int = 5) -> Dict[str, List[TrainingJob]]:
    """Generate all training jobs for grid search."""
    
    datasets = ["diy", "geolife"]
    prev_days_list = [3, 7, 10, 14]
    
    jobs = {
        "pointer_v45": [],
        "MHSA": [],
        "LSTM": []
    }
    
    # Pointer V45 search space
    pointer_space = {
        "d_model": [32, 64, 128],
        "n_head": [2, 4, 8],
        "num_layer": [2, 4, 8],
        "ff_dim": [256, 128, 512],
        "lr": [6e-4, 6.5e-4, 7e-4]
    }
    
    for dataset in datasets:
        for prev_days in prev_days_list:
            for d_model, n_head, num_layer, ff_dim, lr in itertools.product(
                pointer_space["d_model"],
                pointer_space["n_head"],
                pointer_space["num_layer"],
                pointer_space["ff_dim"],
                pointer_space["lr"]
            ):
                params = {
                    "d_model": d_model,
                    "n_head": n_head,
                    "num_layer": num_layer,
                    "ff_dim": ff_dim,
                    "lr": lr
                }
                config_path, config = create_pointer_v45_config(
                    dataset, prev_days, d_model, n_head, num_layer, ff_dim, lr, max_epochs, seed, patience
                )
                
                job = TrainingJob(
                    model_type="pointer_v45",
                    dataset=dataset,
                    prev_days=prev_days,
                    params=params,
                    config_path=config_path,
                    weight_score=calculate_weight_score("pointer_v45", params)
                )
                jobs["pointer_v45"].append(job)
    
    # MHSA search space
    mhsa_space = {
        "base_emb_size": [32, 64, 128],
        "n_head": [2, 4, 8],
        "num_layer": [2, 4, 8],
        "ff_dim": [256, 128, 512],
        "lr": [0.001, 0.002, 0.0005]
    }
    
    for dataset in datasets:
        for prev_days in prev_days_list:
            for base_emb_size, n_head, num_layer, ff_dim, lr in itertools.product(
                mhsa_space["base_emb_size"],
                mhsa_space["n_head"],
                mhsa_space["num_layer"],
                mhsa_space["ff_dim"],
                mhsa_space["lr"]
            ):
                params = {
                    "base_emb_size": base_emb_size,
                    "n_head": n_head,
                    "num_layer": num_layer,
                    "ff_dim": ff_dim,
                    "lr": lr
                }
                config_path, config = create_mhsa_config(
                    dataset, prev_days, base_emb_size, n_head, num_layer, ff_dim, lr, max_epochs, seed, patience
                )
                
                job = TrainingJob(
                    model_type="MHSA",
                    dataset=dataset,
                    prev_days=prev_days,
                    params=params,
                    config_path=config_path,
                    weight_score=calculate_weight_score("MHSA", params)
                )
                jobs["MHSA"].append(job)
    
    # LSTM search space
    lstm_space = {
        "emb_size": [32, 64, 128],
        "hidden_size": [32, 64, 128],
        "num_layer": [2, 4, 8],
        "ff_dim": [256, 128, 512],
        "lr": [0.001, 0.0015, 0.002]
    }
    
    for dataset in datasets:
        for prev_days in prev_days_list:
            for emb_size, hidden_size, num_layer, ff_dim, lr in itertools.product(
                lstm_space["emb_size"],
                lstm_space["hidden_size"],
                lstm_space["num_layer"],
                lstm_space["ff_dim"],
                lstm_space["lr"]
            ):
                params = {
                    "emb_size": emb_size,
                    "hidden_size": hidden_size,
                    "num_layer": num_layer,
                    "ff_dim": ff_dim,
                    "lr": lr
                }
                config_path, config = create_lstm_config(
                    dataset, prev_days, emb_size, hidden_size, num_layer, ff_dim, lr, max_epochs, seed, patience
                )
                
                job = TrainingJob(
                    model_type="LSTM",
                    dataset=dataset,
                    prev_days=prev_days,
                    params=params,
                    config_path=config_path,
                    weight_score=calculate_weight_score("LSTM", params)
                )
                jobs["LSTM"].append(job)
    
    return jobs


def balance_jobs(jobs: List[TrainingJob], num_workers: int = 5) -> List[List[TrainingJob]]:
    """
    Balance jobs across workers so each worker has a mix of heavy and light jobs.
    Uses round-robin assignment after sorting by weight.
    """
    # Sort by weight (heavy first)
    sorted_jobs = sorted(jobs, key=lambda x: x.weight_score, reverse=True)
    
    # Assign to workers using round-robin
    workers = [[] for _ in range(num_workers)]
    worker_weights = [0] * num_workers
    
    for job in sorted_jobs:
        # Assign to worker with lowest current weight
        min_idx = worker_weights.index(min(worker_weights))
        workers[min_idx].append(job)
        worker_weights[min_idx] += job.weight_score
    
    return workers


def save_config(config_path: str, config: Dict):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_training_job(job: TrainingJob, results_queue: queue.Queue, job_idx: int):
    """Run a single training job."""
    print(f"[Job {job_idx}] Starting {job.model_type} on {job.dataset} prev{job.prev_days}")
    
    # Determine training script
    if job.model_type == "pointer_v45":
        script = "src/training/train_pointer_v45.py"
    elif job.model_type == "MHSA":
        script = "src/training/train_MHSA.py"
    else:
        script = "src/training/train_LSTM.py"
    
    # Run training
    cmd = [
        "python", str(PROJECT_ROOT / script),
        "--config", job.config_path
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        success = False
        output = "TIMEOUT"
    except Exception as e:
        success = False
        output = str(e)
    
    elapsed = time.time() - start_time
    
    # Find experiment directory from output
    experiment_dir = None
    if success:
        for line in output.split("\n"):
            if "Experiment directory:" in line:
                experiment_dir = line.split(":")[-1].strip()
                break
            if "Results saved to:" in line:
                experiment_dir = line.split(":")[-1].strip()
                break
    
    result_data = {
        "job": job,
        "success": success,
        "elapsed": elapsed,
        "experiment_dir": experiment_dir,
        "output": output,
        "job_idx": job_idx
    }
    
    results_queue.put(result_data)
    print(f"[Job {job_idx}] Finished {job.model_type} on {job.dataset} prev{job.prev_days} - {'SUCCESS' if success else 'FAILED'} ({elapsed:.1f}s)")


def parse_results(experiment_dir: str) -> Tuple[Dict, Dict, int]:
    """Parse validation and test results from experiment directory.
    
    Reads results from val_results.json and test_results.json files.
    Extracts num_params from training.log.
    
    Returns:
        val_results, test_results, num_params
    """
    val_results = {}
    test_results = {}
    num_params = 0
    
    if experiment_dir and os.path.exists(experiment_dir):
        val_path = os.path.join(experiment_dir, "val_results.json")
        test_path = os.path.join(experiment_dir, "test_results.json")
        
        # Read validation results from JSON
        if os.path.exists(val_path):
            with open(val_path, "r") as f:
                val_results = json.load(f)
        
        # Read test results from JSON
        if os.path.exists(test_path):
            with open(test_path, "r") as f:
                test_results = json.load(f)
        
        # Parse training log for num_params only
        log_path = os.path.join(experiment_dir, "training.log")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log_content = f.read()
            
            import re
            
            # Extract num_params - try different formats
            # Pointer V45 format: "Model parameters: X"
            params_match = re.search(r"Model parameters:\s*([\d,]+)", log_content)
            if params_match:
                num_params = int(params_match.group(1).replace(",", ""))
            else:
                # MHSA/LSTM format: "Total trainable parameters: X"
                params_match = re.search(r"Total trainable parameters:\s*([\d,]+)", log_content)
                if params_match:
                    num_params = int(params_match.group(1).replace(",", ""))
                else:
                    # Alternative format: "Parameters: X"
                    params_match = re.search(r"Parameters:\s*([\d,]+)", log_content)
                    if params_match:
                        num_params = int(params_match.group(1).replace(",", ""))
    
    return val_results, test_results, num_params


def get_csv_headers(model_type: str) -> List[str]:
    """Get CSV headers for a model type."""
    base_headers = [
        "config_path", "experiment_path", "dataset", "prev_days"
    ]
    
    if model_type == "pointer_v45":
        param_headers = ["d_model", "n_head", "num_layer", "ff_dim", "lr"]
    elif model_type == "MHSA":
        param_headers = ["base_emb_size", "n_head", "num_layer", "ff_dim", "lr"]
    else:  # LSTM
        param_headers = ["emb_size", "hidden_size", "num_layer", "ff_dim", "lr"]
    
    # Add num_params column
    param_headers.append("num_params")
    
    metric_headers = [
        "correct@1", "correct@3", "correct@5", "correct@10", "total",
        "rr", "ndcg", "f1", "acc@1", "acc@5", "acc@10", "mrr", "loss"
    ]
    
    return base_headers + param_headers + metric_headers


def write_result_to_csv(
    csv_path: str,
    model_type: str,
    job: TrainingJob,
    experiment_dir: str,
    metrics: Dict,
    num_params: int,
    write_header: bool = False
):
    """Write a single result to CSV."""
    headers = get_csv_headers(model_type)
    
    # Prepare row data
    row = {
        "config_path": job.config_path,
        "experiment_path": experiment_dir or "",
        "dataset": job.dataset,
        "prev_days": job.prev_days,
        "num_params": num_params if num_params > 0 else ""
    }
    
    # Add params
    for key, value in job.params.items():
        row[key] = value
    
    # Add metrics
    for header in headers:
        if header not in row:
            row[header] = metrics.get(header, "")
    
    # Write to CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header or not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_parallel_jobs(
    jobs: List[TrainingJob],
    num_workers: int,
    delay_seconds: int,
    val_csv_path: str,
    test_csv_path: str,
    model_type: str,
    max_epochs: int,
    seed: int,
    patience: int
):
    """Run jobs in parallel with balanced workload."""
    
    # Balance jobs across workers
    worker_batches = balance_jobs(jobs, num_workers)
    
    print(f"\n{'='*60}")
    print(f"Running {len(jobs)} {model_type} jobs with {num_workers} parallel workers")
    print(f"{'='*60}")
    
    # Create and save all config files first
    for job in jobs:
        if model_type == "pointer_v45":
            _, config = create_pointer_v45_config(
                job.dataset, job.prev_days, 
                job.params["d_model"], job.params["n_head"], 
                job.params["num_layer"], job.params["ff_dim"], 
                job.params["lr"], 
                max_epochs,
                seed, patience
            )
        elif model_type == "MHSA":
            _, config = create_mhsa_config(
                job.dataset, job.prev_days,
                job.params["base_emb_size"], job.params["n_head"],
                job.params["num_layer"], job.params["ff_dim"],
                job.params["lr"],
                max_epochs,
                seed, patience
            )
        else:
            _, config = create_lstm_config(
                job.dataset, job.prev_days,
                job.params["emb_size"], job.params["hidden_size"],
                job.params["num_layer"], job.params["ff_dim"],
                job.params["lr"],
                max_epochs,
                seed, patience
            )
        save_config(job.config_path, config)
    
    # Process jobs in rounds
    total_processed = 0
    write_header = True
    
    # Get max jobs per worker
    max_jobs = max(len(batch) for batch in worker_batches)
    
    for round_idx in range(max_jobs):
        # Collect jobs for this round (one from each worker that has jobs left)
        round_jobs = []
        for worker_idx, batch in enumerate(worker_batches):
            if round_idx < len(batch):
                round_jobs.append((batch[round_idx], total_processed + len(round_jobs)))
        
        if not round_jobs:
            break
        
        print(f"\n--- Round {round_idx + 1}: Processing {len(round_jobs)} jobs ---")
        
        # Start threads with delay
        results_queue = queue.Queue()
        threads = []
        
        for i, (job, job_idx) in enumerate(round_jobs):
            thread = threading.Thread(
                target=run_training_job,
                args=(job, results_queue, job_idx)
            )
            threads.append(thread)
            thread.start()
            
            if i < len(round_jobs) - 1:
                time.sleep(delay_seconds)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Process results
        while not results_queue.empty():
            result = results_queue.get()
            job = result["job"]
            experiment_dir = result["experiment_dir"]
            
            if result["success"]:
                val_metrics, test_metrics, num_params = parse_results(experiment_dir)
                
                # Write to CSVs
                write_result_to_csv(val_csv_path, model_type, job, experiment_dir, val_metrics, num_params, write_header)
                write_result_to_csv(test_csv_path, model_type, job, experiment_dir, test_metrics, num_params, write_header)
                write_header = False
            else:
                print(f"WARNING: Job failed for {job.config_path}")
                # Write empty results
                write_result_to_csv(val_csv_path, model_type, job, "", {}, 0, write_header)
                write_result_to_csv(test_csv_path, model_type, job, "", {}, 0, write_header)
                write_header = False
        
        total_processed += len(round_jobs)
        print(f"Progress: {total_processed}/{len(jobs)} jobs completed")


def main():
    parser = argparse.ArgumentParser(description="Grid Search Hyperparameter Tuning")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs for training")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--delay", type=int, default=5, help="Delay between parallel job starts (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--model", type=str, default="all", choices=["all", "pointer_v45", "MHSA", "LSTM"],
                        help="Model to tune (default: all)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of jobs per model (0=no limit, for testing)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "diy", "geolife"],
                        help="Dataset to tune (default: all)")
    args = parser.parse_args()
    
    print("="*60)
    print("Grid Search Hyperparameter Tuning")
    print("="*60)
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Num Workers: {args.num_workers}")
    print(f"Delay: {args.delay}s")
    print(f"Seed: {args.seed}")
    print(f"Patience: {args.patience}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Limit: {args.limit if args.limit > 0 else 'No limit'}")
    print("="*60)
    
    # Create directories
    CONFIG_HT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all jobs
    print("\nGenerating training jobs...")
    all_jobs = generate_all_jobs(args.max_epochs, args.seed, args.patience)
    
    for model_type, jobs in all_jobs.items():
        print(f"  {model_type}: {len(jobs)} jobs")
    
    # Filter by dataset if specified
    if args.dataset != "all":
        for model_type in all_jobs:
            all_jobs[model_type] = [j for j in all_jobs[model_type] if j.dataset == args.dataset]
        print(f"\nFiltered to {args.dataset} dataset:")
        for model_type, jobs in all_jobs.items():
            print(f"  {model_type}: {len(jobs)} jobs")
    
    # Timestamp for result files
    timestamp = get_timestamp()
    
    # Run models in order: PointerV45 -> MHSA -> LSTM
    models_to_run = ["pointer_v45", "MHSA", "LSTM"] if args.model == "all" else [args.model]
    
    for model_type in models_to_run:
        if model_type not in all_jobs:
            continue
        
        jobs = all_jobs[model_type]
        
        # Apply limit if specified
        if args.limit > 0 and len(jobs) > args.limit:
            print(f"\nLimiting {model_type} jobs from {len(jobs)} to {args.limit}")
            jobs = jobs[:args.limit]
        
        val_csv_path = RESULTS_DIR / f"{model_type}_val_results_{timestamp}.csv"
        test_csv_path = RESULTS_DIR / f"{model_type}_test_results_{timestamp}.csv"
        
        print(f"\n{'='*60}")
        print(f"Starting {model_type} Grid Search")
        print(f"Total jobs: {len(jobs)}")
        print(f"Val CSV: {val_csv_path}")
        print(f"Test CSV: {test_csv_path}")
        print(f"{'='*60}")
        
        run_parallel_jobs(
            jobs,
            args.num_workers,
            args.delay,
            str(val_csv_path),
            str(test_csv_path),
            model_type,
            args.max_epochs,
            args.seed,
            args.patience
        )
        
        print(f"\n{model_type} Grid Search Complete!")
    
    print("\n" + "="*60)
    print("All Grid Search Complete!")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

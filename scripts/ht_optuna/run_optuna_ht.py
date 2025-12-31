#!/usr/bin/env python
"""
Hyperparameter Tuning Script using Optuna with TPE Sampler and Hyperband Pruner.

This script performs hyperparameter optimization for:
- Pointer V45 (proposed model) 
- MHSA (Multi-Head Self-Attention)
- LSTM

Usage:
    # Activate mlenv first:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
    
    # Run hyperparameter tuning:
    python scripts/ht_optuna/run_optuna_ht.py --model pointer_v45 --n_jobs 10
    python scripts/ht_optuna/run_optuna_ht.py --model MHSA --n_jobs 10
    python scripts/ht_optuna/run_optuna_ht.py --model LSTM --n_jobs 4
    python scripts/ht_optuna/run_optuna_ht.py --model all --n_jobs 4

Hyperparameter tuning configuration:
- Objective: Maximize Acc@1 on Validation Set
- Sampler: TPE (Tree-structured Parzen Estimator)
- Pruner: Hyperband
- Trials: 50 per dataset per prev_days (total: 2 datasets * 4 prev_days * 50 = 400 trials per model)
- Fixed: seed=42, patience=5, max_epoch=50
"""

import os
import sys
import json
import yaml
import time
import argparse
import subprocess
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration directories
CONFIG_DIR = PROJECT_ROOT / "config" / "ht_optuna"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "ht_optuna" / "results"

# Fixed hyperparameters
SEED = 42
PATIENCE = 5
MAX_EPOCH = 50

# Dataset configurations
DATASETS = {
    "diy": {
        "data_dir": "data/diy_eps50/processed",
        "dataset_prefix_template": "diy_eps50_prev{prev_day}",
        "prev_days": [3, 7, 10, 14],
    },
    "geolife": {
        "data_dir": "data/geolife_eps20/processed",
        "dataset_prefix_template": "geolife_eps20_prev{prev_day}",
        "prev_days": [3, 7, 10, 14],
    },
}

# Search spaces for each model
SEARCH_SPACES = {
    "pointer_v45": {
        "d_model": [32, 64, 128],
        "nhead": [2, 4, 8],
        "num_layers": [2, 4, 8],
        "dim_feedforward": [128, 256, 512],
        "lr": [6e-4, 6.5e-4, 7e-4],
    },
    "MHSA": {
        "base_emb_size": [32, 64, 128],
        "nhead": [2, 4, 8],
        "num_encoder_layers": [2, 4, 8],
        "dim_feedforward": [128, 256, 512],
        "lr": [0.0005, 0.001, 0.002],
    },
    "LSTM": {
        "base_emb_size": [32, 64, 128],
        "lstm_hidden_size": [32, 64, 128],
        "lstm_num_layers": [2, 4, 8],
        "dim_feedforward": [128, 256, 512],
        "lr": [0.001, 0.0015, 0.002],
    },
}

# Base config paths
BASE_CONFIGS = {
    "pointer_v45": {
        "diy": "config/models/config_pointer_v45_diy.yaml",
        "geolife": "config/models/config_pointer_v45_geolife.yaml",
    },
    "MHSA": {
        "diy": "config/models/config_MHSA_diy.yaml",
        "geolife": "config/models/config_MHSA_geolife.yaml",
    },
    "LSTM": {
        "diy": "config/models/config_LSTM_diy.yaml",
        "geolife": "config/models/config_LSTM_geolife.yaml",
    },
}

# Training scripts
TRAINING_SCRIPTS = {
    "pointer_v45": "src/training/train_pointer_v45.py",
    "MHSA": "src/training/train_MHSA.py",
    "LSTM": "src/training/train_LSTM.py",
}


def get_timestamp():
    """Get current timestamp in GMT+7."""
    gmt7 = timezone(timedelta(hours=7))
    return datetime.now(gmt7).strftime("%Y%m%d_%H%M%S")


def load_metadata(dataset: str, prev_day: int) -> Dict:
    """Load metadata for a specific dataset and prev_day."""
    dataset_info = DATASETS[dataset]
    prefix = dataset_info["dataset_prefix_template"].format(prev_day=prev_day)
    metadata_path = PROJECT_ROOT / dataset_info["data_dir"] / f"{prefix}_metadata.json"
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def create_pointer_v45_config(
    dataset: str,
    prev_day: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    lr: float,
    trial_id: int,
) -> str:
    """Create a config file for Pointer V45 model."""
    import uuid
    
    metadata = load_metadata(dataset, prev_day)
    dataset_info = DATASETS[dataset]
    prefix = dataset_info["dataset_prefix_template"].format(prev_day=prev_day)
    
    # Generate unique experiment root to avoid checkpoint conflicts in parallel execution
    timestamp = get_timestamp()
    unique_id = str(uuid.uuid4())[:8]
    experiment_root = f"experiments/ht_optuna_{timestamp}_{unique_id}"
    
    config = {
        "seed": SEED,
        "data": {
            "data_dir": dataset_info["data_dir"],
            "dataset_prefix": prefix,
            "dataset": dataset,
            "experiment_root": experiment_root,
            "num_workers": 0,
        },
        "model": {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": 0.15,
        },
        "training": {
            "batch_size": 128,
            "num_epochs": MAX_EPOCH,
            "learning_rate": lr,
            "weight_decay": 0.015,
            "label_smoothing": 0.03,
            "grad_clip": 0.8,
            "patience": PATIENCE,
            "min_epochs": 3,
            "warmup_epochs": 5,
            "use_amp": True,
            "min_lr": 0.000001,
        },
    }
    
    # Generate config filename with timestamp and uuid to avoid conflicts
    config_name = f"optuna_pointer_v45_{dataset}_prev{prev_day}_t{trial_id}_{timestamp}_{unique_id}.yaml"
    config_path = CONFIG_DIR / config_name
    
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_path)


def create_mhsa_config(
    dataset: str,
    prev_day: int,
    base_emb_size: int,
    nhead: int,
    num_encoder_layers: int,
    dim_feedforward: int,
    lr: float,
    trial_id: int,
) -> str:
    """Create a config file for MHSA model."""
    import uuid
    
    metadata = load_metadata(dataset, prev_day)
    dataset_info = DATASETS[dataset]
    prefix = dataset_info["dataset_prefix_template"].format(prev_day=prev_day)
    
    # Batch sizes based on dataset
    batch_size = 256 if dataset == "diy" else 32
    print_step = 10 if dataset == "diy" else 20
    
    # Generate unique experiment root to avoid checkpoint conflicts in parallel execution
    timestamp = get_timestamp()
    unique_id = str(uuid.uuid4())[:8]
    experiment_root = f"experiments/ht_optuna_{timestamp}_{unique_id}"
    
    config = {
        "seed": SEED,
        "data": {
            "data_dir": dataset_info["data_dir"],
            "dataset_prefix": prefix,
            "dataset": dataset,
            "experiment_root": experiment_root,
        },
        "training": {
            "if_embed_user": True,
            "if_embed_poi": False,
            "if_embed_time": True,
            "if_embed_duration": True,
            "previous_day": prev_day,
            "verbose": True,
            "debug": False,
            "batch_size": batch_size,
            "print_step": print_step,
            "num_workers": 0,
            "day_selection": "default",
        },
        "dataset_info": {
            "total_loc_num": metadata["total_loc_num"],
            "total_user_num": metadata["total_user_num"],
        },
        "embedding": {
            "base_emb_size": base_emb_size,
            "poi_original_size": 16,
        },
        "model": {
            "networkName": "transformer",
            "num_encoder_layers": num_encoder_layers,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "fc_dropout": 0.2 if dataset == "geolife" else 0.1,
        },
        "optimiser": {
            "optimizer": "Adam",
            "max_epoch": MAX_EPOCH,
            "lr": lr,
            "weight_decay": 0.000001,
            "beta1": 0.9,
            "beta2": 0.999,
            "momentum": 0.98,
            "num_warmup_epochs": 2,
            "num_training_epochs": MAX_EPOCH,
            "patience": PATIENCE,
            "lr_step_size": 1,
            "lr_gamma": 0.1,
        },
    }
    
    # Generate config filename with timestamp and uuid to avoid conflicts
    config_name = f"optuna_MHSA_{dataset}_prev{prev_day}_t{trial_id}_{timestamp}_{unique_id}.yaml"
    config_path = CONFIG_DIR / config_name
    
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_path)


def create_lstm_config(
    dataset: str,
    prev_day: int,
    base_emb_size: int,
    lstm_hidden_size: int,
    lstm_num_layers: int,
    dim_feedforward: int,
    lr: float,
    trial_id: int,
) -> str:
    """Create a config file for LSTM model."""
    import uuid
    
    metadata = load_metadata(dataset, prev_day)
    dataset_info = DATASETS[dataset]
    prefix = dataset_info["dataset_prefix_template"].format(prev_day=prev_day)
    
    # Batch sizes based on dataset
    batch_size = 256 if dataset == "diy" else 32
    print_step = 10 if dataset == "diy" else 20
    
    # Generate unique experiment root to avoid checkpoint conflicts in parallel execution
    timestamp = get_timestamp()
    unique_id = str(uuid.uuid4())[:8]
    experiment_root = f"experiments/ht_optuna_{timestamp}_{unique_id}"
    
    config = {
        "seed": SEED,
        "data": {
            "data_dir": dataset_info["data_dir"],
            "dataset_prefix": prefix,
            "dataset": dataset,
            "experiment_root": experiment_root,
        },
        "training": {
            "if_embed_user": True,
            "if_embed_poi": False,
            "if_embed_time": True,
            "if_embed_duration": True,
            "previous_day": prev_day,
            "verbose": True,
            "debug": False,
            "batch_size": batch_size,
            "print_step": print_step,
            "num_workers": 0,
            "day_selection": "default",
        },
        "dataset_info": {
            "total_loc_num": metadata["total_loc_num"],
            "total_user_num": metadata["total_user_num"],
        },
        "embedding": {
            "base_emb_size": base_emb_size,
            "poi_original_size": 16,
        },
        "model": {
            "networkName": "lstm",
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_num_layers,
            "lstm_dropout": 0.2,
            "fc_dropout": 0.2 if dataset == "geolife" else 0.1,
        },
        "optimiser": {
            "optimizer": "Adam",
            "max_epoch": MAX_EPOCH,
            "lr": lr,
            "weight_decay": 0.000001,
            "beta1": 0.9,
            "beta2": 0.999,
            "momentum": 0.98,
            "num_warmup_epochs": 2,
            "num_training_epochs": MAX_EPOCH,
            "patience": PATIENCE,
            "lr_step_size": 1,
            "lr_gamma": 0.1,
        },
    }
    
    # Generate config filename with timestamp and uuid to avoid conflicts
    config_name = f"optuna_LSTM_{dataset}_prev{prev_day}_t{trial_id}_{timestamp}_{unique_id}.yaml"
    config_path = CONFIG_DIR / config_name
    
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_path)


def run_training(model_name: str, config_path: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
    """Run training script and return validation and test metrics."""
    script_path = PROJECT_ROOT / TRAINING_SCRIPTS[model_name]
    
    # Run training
    cmd = [
        "python", str(script_path),
        "--config", config_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600 * 2,  # 2 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")
            print(f"STDERR: {result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr}")
            return None, None, None
        
        # Parse experiment directory from output
        experiment_dir = None
        for line in result.stdout.split("\n"):
            if "Experiment directory:" in line:
                experiment_dir = line.split("Experiment directory:")[-1].strip()
                break
            elif "Results saved to:" in line:
                experiment_dir = line.split("Results saved to:")[-1].strip()
                break
        
        if not experiment_dir:
            print("Could not find experiment directory in output")
            print(f"STDOUT (last 2000 chars): {result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")
            return None, None, None
        
        # Load results
        val_results_path = Path(experiment_dir) / "val_results.json"
        test_results_path = Path(experiment_dir) / "test_results.json"
        
        val_metrics = None
        test_metrics = None
        
        if val_results_path.exists():
            with open(val_results_path) as f:
                val_metrics = json.load(f)
        
        if test_results_path.exists():
            with open(test_results_path) as f:
                test_metrics = json.load(f)
        
        # For MHSA and LSTM, val_results may be empty. Use test_metrics as proxy for val if empty
        if val_metrics is not None and len(val_metrics) == 0 and test_metrics is not None:
            val_metrics = test_metrics.copy()
        
        return val_metrics, test_metrics, experiment_dir
    
    except subprocess.TimeoutExpired:
        print("Training timed out")
        return None, None, None
    except Exception as e:
        print(f"Error running training: {e}")
        return None, None, None


def append_results_to_csv(
    csv_path: str,
    model_name: str,
    dataset: str,
    prev_day: int,
    params: Dict,
    metrics: Dict,
    config_path: str,
    experiment_path: str,
):
    """Append results to CSV file with thread-safe locking."""
    from filelock import FileLock
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define CSV columns
    base_columns = [
        "timestamp", "model", "dataset", "prev_day", "config_path", "experiment_path"
    ]
    
    # Parameter columns based on model
    if model_name == "pointer_v45":
        param_columns = ["d_model", "nhead", "num_layers", "dim_feedforward", "lr"]
    elif model_name == "MHSA":
        param_columns = ["base_emb_size", "nhead", "num_encoder_layers", "dim_feedforward", "lr"]
    else:  # LSTM
        param_columns = ["base_emb_size", "lstm_hidden_size", "lstm_num_layers", "dim_feedforward", "lr"]
    
    metric_columns = [
        "correct@1", "correct@3", "correct@5", "correct@10", "total",
        "rr", "ndcg", "f1", "acc@1", "acc@5", "acc@10", "mrr", "loss"
    ]
    
    all_columns = base_columns + param_columns + metric_columns
    
    # Use file lock for thread safety
    lock_path = csv_path + ".lock"
    lock = FileLock(lock_path, timeout=60)
    
    with lock:
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            
            if not file_exists:
                writer.writeheader()
            
            # Prepare row
            row = {
                "timestamp": get_timestamp(),
                "model": model_name,
                "dataset": dataset,
                "prev_day": prev_day,
                "config_path": config_path,
                "experiment_path": experiment_path,
            }
            
            # Add parameters
            for col in param_columns:
                row[col] = params.get(col, "")
            
            # Add metrics
            for col in metric_columns:
                if col == "loss" and "loss" in metrics:
                    row[col] = metrics.get("loss", "")
                elif col == "loss" and "val_loss" in metrics:
                    row[col] = metrics.get("val_loss", "")
                else:
                    row[col] = metrics.get(col, "")
            
            writer.writerow(row)


def create_objective(
    model_name: str,
    dataset: str,
    prev_day: int,
    val_csv_path: str,
    test_csv_path: str,
):
    """Create an Optuna objective function for a specific model/dataset/prev_day."""
    
    def objective(trial: optuna.Trial) -> float:
        # Add delay to avoid naming conflicts in parallel execution
        time.sleep(1)
        
        search_space = SEARCH_SPACES[model_name]
        
        if model_name == "pointer_v45":
            d_model = trial.suggest_categorical("d_model", search_space["d_model"])
            nhead = trial.suggest_categorical("nhead", search_space["nhead"])
            num_layers = trial.suggest_categorical("num_layers", search_space["num_layers"])
            dim_feedforward = trial.suggest_categorical("dim_feedforward", search_space["dim_feedforward"])
            lr = trial.suggest_categorical("lr", search_space["lr"])
            
            config_path = create_pointer_v45_config(
                dataset=dataset,
                prev_day=prev_day,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                lr=lr,
                trial_id=trial.number,
            )
            
            params = {
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "lr": lr,
            }
            
        elif model_name == "MHSA":
            base_emb_size = trial.suggest_categorical("base_emb_size", search_space["base_emb_size"])
            nhead = trial.suggest_categorical("nhead", search_space["nhead"])
            num_encoder_layers = trial.suggest_categorical("num_encoder_layers", search_space["num_encoder_layers"])
            dim_feedforward = trial.suggest_categorical("dim_feedforward", search_space["dim_feedforward"])
            lr = trial.suggest_categorical("lr", search_space["lr"])
            
            config_path = create_mhsa_config(
                dataset=dataset,
                prev_day=prev_day,
                base_emb_size=base_emb_size,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                lr=lr,
                trial_id=trial.number,
            )
            
            params = {
                "base_emb_size": base_emb_size,
                "nhead": nhead,
                "num_encoder_layers": num_encoder_layers,
                "dim_feedforward": dim_feedforward,
                "lr": lr,
            }
            
        else:  # LSTM
            base_emb_size = trial.suggest_categorical("base_emb_size", search_space["base_emb_size"])
            lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", search_space["lstm_hidden_size"])
            lstm_num_layers = trial.suggest_categorical("lstm_num_layers", search_space["lstm_num_layers"])
            dim_feedforward = trial.suggest_categorical("dim_feedforward", search_space["dim_feedforward"])
            lr = trial.suggest_categorical("lr", search_space["lr"])
            
            config_path = create_lstm_config(
                dataset=dataset,
                prev_day=prev_day,
                base_emb_size=base_emb_size,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                dim_feedforward=dim_feedforward,
                lr=lr,
                trial_id=trial.number,
            )
            
            params = {
                "base_emb_size": base_emb_size,
                "lstm_hidden_size": lstm_hidden_size,
                "lstm_num_layers": lstm_num_layers,
                "dim_feedforward": dim_feedforward,
                "lr": lr,
            }
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: {model_name} on {dataset} prev{prev_day}")
        print(f"Params: {params}")
        print(f"Config: {config_path}")
        print(f"{'='*60}\n")
        
        # Run training
        val_metrics, test_metrics, experiment_path = run_training(model_name, config_path)
        
        if val_metrics is None:
            print(f"Trial {trial.number} failed!")
            return float("-inf")
        
        # Save results to CSV
        append_results_to_csv(
            val_csv_path, model_name, dataset, prev_day,
            params, val_metrics, config_path, experiment_path or ""
        )
        
        if test_metrics is not None:
            append_results_to_csv(
                test_csv_path, model_name, dataset, prev_day,
                params, test_metrics, config_path, experiment_path or ""
            )
        
        # Return validation Acc@1 as the objective
        val_acc1 = val_metrics.get("acc@1", 0.0)
        print(f"Trial {trial.number} completed: Val Acc@1 = {val_acc1:.2f}%")
        
        return val_acc1
    
    return objective


def run_hyperparameter_tuning(
    model_name: str,
    n_jobs: int = 1,
    n_trials: int = 50,
):
    """Run hyperparameter tuning for a specific model."""
    print(f"\n{'='*80}")
    print(f"Starting hyperparameter tuning for {model_name}")
    print(f"n_jobs: {n_jobs}, n_trials per dataset/prev_day: {n_trials}")
    print(f"Fixed params: seed={SEED}, patience={PATIENCE}, max_epoch={MAX_EPOCH}")
    print(f"{'='*80}\n")
    
    # CSV paths
    timestamp = get_timestamp()
    val_csv_path = str(RESULTS_DIR / f"{model_name}_val_results_{timestamp}.csv")
    test_csv_path = str(RESULTS_DIR / f"{model_name}_test_results_{timestamp}.csv")
    
    # Run for each dataset and prev_day combination
    for dataset in ["diy", "geolife"]:
        for prev_day in DATASETS[dataset]["prev_days"]:
            print(f"\n{'='*80}")
            print(f"Tuning {model_name} on {dataset} with prev_day={prev_day}")
            print(f"{'='*80}\n")
            
            # Create study
            study_name = f"{model_name}_{dataset}_prev{prev_day}_{timestamp}"
            
            sampler = TPESampler(seed=SEED)
            pruner = HyperbandPruner(
                min_resource=1,
                max_resource=MAX_EPOCH,
                reduction_factor=3,
            )
            
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",  # Maximize Acc@1
                sampler=sampler,
                pruner=pruner,
            )
            
            # Create objective
            objective = create_objective(
                model_name=model_name,
                dataset=dataset,
                prev_day=prev_day,
                val_csv_path=val_csv_path,
                test_csv_path=test_csv_path,
            )
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                show_progress_bar=True,
            )
            
            # Print best results
            print(f"\nBest trial for {model_name} on {dataset} prev{prev_day}:")
            print(f"  Value (Val Acc@1): {study.best_value:.2f}%")
            print(f"  Params: {study.best_params}")
            
            # Save study summary
            study_summary_path = RESULTS_DIR / f"{model_name}_{dataset}_prev{prev_day}_study_summary_{timestamp}.json"
            study_summary = {
                "model": model_name,
                "dataset": dataset,
                "prev_day": prev_day,
                "n_trials": n_trials,
                "n_jobs": n_jobs,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_trial_number": study.best_trial.number,
            }
            with open(study_summary_path, "w") as f:
                json.dump(study_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Hyperparameter tuning completed for {model_name}")
    print(f"Val results saved to: {val_csv_path}")
    print(f"Test results saved to: {test_csv_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning using Optuna with TPE sampler and Hyperband pruner"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["pointer_v45", "MHSA", "LSTM", "all"],
        help="Model to tune (pointer_v45, MHSA, LSTM, or all)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=10,
        help="Number of parallel jobs for Optuna (default: 1)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials per dataset/prev_day combination (default: 50)",
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config directory: {CONFIG_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Run tuning
    if args.model == "all":
        # Run in order: pointer_v45 -> MHSA -> LSTM
        for model in ["pointer_v45", "MHSA", "LSTM"]:
            run_hyperparameter_tuning(
                model_name=model,
                n_jobs=args.n_jobs,
                n_trials=args.n_trials,
            )
    else:
        run_hyperparameter_tuning(
            model_name=args.model,
            n_jobs=args.n_jobs,
            n_trials=args.n_trials,
        )


if __name__ == "__main__":
    main()

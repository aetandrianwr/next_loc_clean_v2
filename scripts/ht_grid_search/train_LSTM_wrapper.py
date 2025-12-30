#!/usr/bin/env python3
"""
Wrapper script for LSTM training that ensures val_results.json is properly saved.

This wrapper calls the original train_LSTM.py and then re-evaluates on validation set
to save complete val_results.json.

Usage:
    python scripts/ht_grid_search/train_LSTM_wrapper.py --config config/path.yaml
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from sklearn.metrics import f1_score

from src.models.baseline.LSTM import LSTMModel
from src.evaluation.metrics import calculate_correct_total_prediction, get_performance_dict


def load_config(path):
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for _, value in cfg.items():
        if isinstance(value, dict):
            for k, v in value.items():
                config[k] = v
        else:
            config[_] = value

    return config


class EasyDict(dict):
    """Dictionary with attribute-style access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def find_latest_experiment(experiment_root, dataset_name, model_name="LSTM"):
    """Find the latest experiment directory."""
    import glob
    pattern = os.path.join(experiment_root, f"{dataset_name}_{model_name}_*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getctime)


def evaluate_on_validation(config, experiment_dir, device):
    """Re-evaluate the model on validation set and save val_results.json."""
    import pickle
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    import torch.nn as nn
    
    # Load validation data
    data_dir = config.data_dir
    dataset_prefix = config.dataset_prefix
    val_path = os.path.join(data_dir, f"{dataset_prefix}_validation.pk")
    
    with open(val_path, "rb") as f:
        val_data = pickle.load(f)
    
    # Simple dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            selected = self.data[idx]
            return_dict = {}
            x = torch.tensor(selected["X"])
            y = torch.tensor(selected["Y"])
            return_dict["user"] = torch.tensor(selected["user_X"][0])
            return_dict["time"] = torch.tensor(selected["start_min_X"] // 15)
            return_dict["diff"] = torch.tensor(selected["diff"])
            return_dict["duration"] = torch.tensor(selected["dur_X"] // 30, dtype=torch.long)
            return_dict["weekday"] = torch.tensor(selected["weekday_X"])
            if "poi_X" in selected:
                return_dict["poi"] = torch.tensor(np.array(selected["poi_X"]), dtype=torch.float32)
            return x, y, return_dict
    
    def collate_fn(batch):
        x_batch, y_batch = [], []
        x_dict_batch = {"len": []}
        for key in batch[0][-1]:
            x_dict_batch[key] = []
        for src_sample, tgt_sample, return_dict in batch:
            x_batch.append(src_sample)
            y_batch.append(tgt_sample)
            x_dict_batch["len"].append(len(src_sample))
            for key in return_dict:
                x_dict_batch[key].append(return_dict[key])
        x_batch = pad_sequence(x_batch)
        y_batch = torch.tensor(y_batch, dtype=torch.int64)
        x_dict_batch["user"] = torch.tensor(x_dict_batch["user"], dtype=torch.int64)
        x_dict_batch["len"] = torch.tensor(x_dict_batch["len"], dtype=torch.int64)
        for key in x_dict_batch:
            if key in ["user", "len", "history_count"]:
                continue
            x_dict_batch[key] = pad_sequence(x_dict_batch[key])
        return x_batch, y_batch, x_dict_batch
    
    val_dataset = SimpleDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    
    # Load model
    model = LSTMModel(config=config, total_loc_num=config.total_loc_num).to(device)
    checkpoint_path = os.path.join(experiment_dir, "checkpoints", "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Evaluate
    model.eval()
    total_val_loss = 0
    true_ls = []
    top1_ls = []
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    CEL = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    
    with torch.no_grad():
        for x, y, x_dict in val_loader:
            x = x.to(device)
            y = y.to(device)
            for key in x_dict:
                x_dict[key] = x_dict[key].to(device)
            
            logits = model(x, x_dict, device)
            loss = CEL(logits.view(-1, logits.shape[-1]), y.reshape(-1))
            total_val_loss += loss.item()
            
            batch_result_arr, batch_true, batch_top1 = calculate_correct_total_prediction(logits, y)
            result_arr += batch_result_arr
            true_ls.extend(batch_true.tolist())
            if not batch_top1.shape:
                top1_ls.extend([batch_top1.tolist()])
            else:
                top1_ls.extend(batch_top1.tolist())
    
    val_loss = total_val_loss / len(val_loader)
    f1 = f1_score(true_ls, top1_ls, average="weighted", zero_division=0)
    
    val_return_dict = {
        "val_loss": val_loss,
        "correct@1": result_arr[0],
        "correct@3": result_arr[1],
        "correct@5": result_arr[2],
        "correct@10": result_arr[3],
        "f1": f1,
        "rr": result_arr[4],
        "ndcg": result_arr[5],
        "total": result_arr[6],
    }
    
    val_performance = get_performance_dict(val_return_dict)
    val_performance["loss"] = val_loss
    
    # Save val_results.json
    val_results_path = os.path.join(experiment_dir, "val_results.json")
    val_results = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in val_performance.items()}
    with open(val_results_path, "w") as f:
        json.dump(val_results, f, indent=2)
    
    return val_performance


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model with proper val_results saving")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    # First run the original training script
    import subprocess
    train_script = str(PROJECT_ROOT / "src" / "training" / "train_LSTM.py")
    result = subprocess.run(
        ["python", train_script, "--config", args.config],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        sys.exit(result.returncode)
    
    # Find experiment directory from output
    experiment_dir = None
    for line in result.stdout.split("\n"):
        if "Experiment directory:" in line:
            experiment_dir = line.split(":")[-1].strip()
            break
        if "Results saved to:" in line:
            experiment_dir = line.split(":")[-1].strip()
            break
    
    if not experiment_dir:
        # Try to find latest experiment
        config = load_config(args.config)
        config = EasyDict(config)
        metadata_path = os.path.join(config.data_dir, f"{config.dataset_prefix}_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        dataset_name = metadata["dataset_name"]
        experiment_dir = find_latest_experiment(config.experiment_root, dataset_name, "LSTM")
    
    if experiment_dir and os.path.exists(experiment_dir):
        # Re-evaluate on validation set
        config = load_config(args.config)
        config = EasyDict(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("\nRe-evaluating on validation set...")
        val_perf = evaluate_on_validation(config, experiment_dir, device)
        print(f"Validation Acc@1: {val_perf['acc@1']:.2f}%")
        print(f"val_results.json saved to: {experiment_dir}")


if __name__ == "__main__":
    main()

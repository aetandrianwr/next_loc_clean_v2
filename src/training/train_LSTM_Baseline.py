"""
Training script for LSTM Baseline Model.

This script trains the LSTM Baseline model for next location prediction.
The LSTM baseline follows the architecture described in Hong et al. 2023,
achieving ~28.4% Acc@1 on the Geolife dataset.

Usage:
    python src/training/train_LSTM_Baseline.py --config config/models/config_LSTM_Baseline_geolife.yaml
    python src/training/train_LSTM_Baseline.py --config config/models/config_LSTM_Baseline_diy.yaml

This script handles:
- Configuration loading from YAML
- Data loading from preprocessed pickle files
- Model training with early stopping
- Evaluation on validation and test sets
- Saving checkpoints, logs, and metrics to experiments/ directory
"""

import os
import sys
import json
import yaml
import random
import argparse
import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score

from transformers import get_linear_schedule_with_warmup

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.baseline.LSTM_Baseline import LSTMBaseline
from src.evaluation.metrics import (
    calculate_metrics,
    calculate_correct_total_prediction,
    get_performance_dict,
)


def setup_seed(seed):
    """Fix random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to YAML config file
    
    Returns:
        Flattened configuration dictionary
    """
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


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after given patience."""

    def __init__(self, log_dir, patience=5, verbose=False, delta=0):
        self.log_dir = log_dir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_return_dict = {"val_loss": np.inf}
        self.delta = delta

    def __call__(self, return_dict, model):
        score = return_dict["val_loss"]

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(return_dict, model)
            return

        if score < self.best_score - self.delta:
            self.best_score = score
            self.save_checkpoint(return_dict, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, return_dict, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.best_return_dict['val_loss']:.6f} --> {return_dict['val_loss']:.6f}). Saving model..."
            )
        checkpoint_path = os.path.join(self.log_dir, "checkpoints", "checkpoint.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        self.best_return_dict = return_dict


class LocationDataset(Dataset):
    """
    Dataset for next location prediction.
    
    Loads preprocessed data from pickle files.
    """
    def __init__(self, data_path, dataset_name="geolife"):
        """
        Args:
            data_path: Path to pickle file containing preprocessed data
            dataset_name: Name of dataset (for POI handling)
        """
        self.data = pickle.load(open(data_path, "rb"))
        self.dataset_name = dataset_name
        self.len = len(self.data)

    def __len__(self):
        return self.len

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

        return x, y, return_dict


def collate_fn(batch):
    """Collate function for DataLoader to handle variable-length sequences."""
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


def get_dataloaders(config):
    """Create train, validation, and test data loaders."""
    data_dir = config.data_dir
    dataset_prefix = config.dataset_prefix

    train_path = os.path.join(data_dir, f"{dataset_prefix}_train.pk")
    val_path = os.path.join(data_dir, f"{dataset_prefix}_validation.pk")
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")

    dataset_train = LocationDataset(train_path, config.dataset)
    dataset_val = LocationDataset(val_path, config.dataset)
    dataset_test = LocationDataset(test_path, config.dataset)

    kwds_train = {
        "shuffle": True,
        "num_workers": config.num_workers,
        "drop_last": True,
        "batch_size": config.batch_size,
        "pin_memory": True,
    }
    kwds_val = {
        "shuffle": False,
        "num_workers": config.num_workers,
        "batch_size": config.batch_size,
        "pin_memory": True,
    }

    train_loader = DataLoader(dataset_train, collate_fn=collate_fn, **kwds_train)
    val_loader = DataLoader(dataset_val, collate_fn=collate_fn, **kwds_val)
    test_loader = DataLoader(dataset_test, collate_fn=collate_fn, **kwds_val)

    return train_loader, val_loader, test_loader


def send_to_device(inputs, device, config):
    """Move batch data to device."""
    x, y, x_dict = inputs
    x = x.to(device)
    for key in x_dict:
        x_dict[key] = x_dict[key].to(device)
    y = y.to(device)
    return x, y, x_dict


def get_optimizer(config, model):
    """Create optimizer based on config."""
    if config.optimizer == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=True,
        )
    elif config.optimizer == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
    return optim


def train_epoch(config, model, train_loader, optim, device, epoch, scheduler, scheduler_count, globaliter, log_file):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    n_batches = len(train_loader)

    CEL = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    import time
    start_time = time.time()
    optim.zero_grad(set_to_none=True)

    for i, inputs in enumerate(train_loader):
        globaliter += 1

        x, y, x_dict = send_to_device(inputs, device, config)
        logits = model(x, x_dict, device)

        loss_size = CEL(logits.view(-1, logits.shape[-1]), y.reshape(-1))

        optim.zero_grad(set_to_none=True)
        loss_size.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        if scheduler_count == 0:
            scheduler.step()

        running_loss += loss_size.item()

        batch_result_arr, _, _ = calculate_correct_total_prediction(logits, y)
        result_arr += batch_result_arr

        if config.verbose and ((i + 1) % config.print_step == 0):
            msg = (
                f"Epoch {epoch + 1}, {100 * (i + 1) / n_batches:.1f}%\t "
                f"loss: {running_loss / config.print_step:.3f} "
                f"acc@1: {100 * result_arr[0] / result_arr[-1]:.2f} "
                f"mrr: {100 * result_arr[4] / result_arr[-1]:.2f}, "
                f"ndcg: {100 * result_arr[5] / result_arr[-1]:.2f}, "
                f"took: {time.time() - start_time:.2f}s"
            )
            print(msg, flush=True)
            log_file.write(msg + "\n")

            running_loss = 0.0
            result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            start_time = time.time()

        if config.debug and i > 20:
            break

    return globaliter


def validate(config, model, data_loader, device):
    """Validate model on validation set."""
    total_val_loss = 0
    true_ls = []
    top1_ls = []

    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    CEL = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            x, y, x_dict = send_to_device(inputs, device, config)
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

    val_loss = total_val_loss / len(data_loader)
    f1 = f1_score(true_ls, top1_ls, average="weighted")

    if config.verbose:
        print(
            f"Validation loss = {val_loss:.2f} "
            f"acc@1 = {100 * result_arr[0] / result_arr[-1]:.2f} "
            f"f1 = {100 * f1:.2f} "
            f"mrr = {100 * result_arr[4] / result_arr[-1]:.2f}, "
            f"ndcg = {100 * result_arr[5] / result_arr[-1]:.2f}"
        )

    return {
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


def test(config, model, data_loader, device):
    """Test model on test set."""
    true_ls = []
    top1_ls = []

    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    result_dict = {}
    for i in range(1, config.total_user_num):
        result_dict[i] = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            x, y, x_dict = send_to_device(inputs, device, config)
            logits = model(x, x_dict, device)

            user_arr = x_dict["user"].cpu().detach().numpy()
            unique = np.unique(user_arr)
            for user in unique:
                index = np.nonzero(user_arr == user)[0]
                batch_user, _, _ = calculate_correct_total_prediction(logits[index, :], y[index])
                result_dict[user] += batch_user

            batch_result_arr, batch_true, batch_top1 = calculate_correct_total_prediction(logits, y)
            result_arr += batch_result_arr
            true_ls.extend(batch_true.numpy())
            top1_ls.extend(batch_top1.numpy())

    f1 = f1_score(true_ls, top1_ls, average="weighted")

    if config.verbose:
        print(
            f"Test acc@1 = {100 * result_arr[0] / result_arr[-1]:.2f} "
            f"f1 = {100 * f1:.2f} "
            f"mrr = {100 * result_arr[4] / result_arr[-1]:.2f} "
            f"ndcg = {100 * result_arr[5] / result_arr[-1]:.2f}"
        )

    return (
        {
            "correct@1": result_arr[0],
            "correct@3": result_arr[1],
            "correct@5": result_arr[2],
            "correct@10": result_arr[3],
            "f1": f1,
            "rr": result_arr[4],
            "ndcg": result_arr[5],
            "total": result_arr[6],
        },
        result_dict,
    )


def train_model(config, model, train_loader, val_loader, device, log_dir, log_file):
    """Main training loop with early stopping."""
    import time

    optim = get_optimizer(config, model)

    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
        num_training_steps=len(train_loader) * config.num_training_epochs,
    )
    scheduler_ES = StepLR(optim, step_size=config.lr_step_size, gamma=config.lr_gamma)

    if config.verbose:
        print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

    training_start_time = time.time()
    globaliter = 0
    scheduler_count = 0

    early_stopping = EarlyStopping(log_dir, patience=config.patience, verbose=config.verbose, delta=0.001)

    performance = {}

    for epoch in range(config.max_epoch):
        log_file.write(f"\n=== Epoch {epoch + 1} ===\n")
        
        globaliter = train_epoch(
            config, model, train_loader, optim, device, epoch, scheduler, scheduler_count, globaliter, log_file
        )

        return_dict = validate(config, model, val_loader, device)
        
        val_msg = (
            f"Validation - loss: {return_dict['val_loss']:.4f}, "
            f"acc@1: {100 * return_dict['correct@1'] / return_dict['total']:.2f}%, "
            f"f1: {100 * return_dict['f1']:.2f}%, "
            f"mrr: {100 * return_dict['rr'] / return_dict['total']:.2f}%, "
            f"ndcg: {100 * return_dict['ndcg'] / return_dict['total']:.2f}%"
        )
        log_file.write(val_msg + "\n")

        early_stopping(return_dict, model)

        if early_stopping.early_stop:
            if config.verbose:
                print("=" * 50)
                print("Early stopping")
            if scheduler_count == 2:
                performance = get_performance_dict(early_stopping.best_return_dict)
                msg = f"Training finished.\t Time: {time.time() - training_start_time:.2f}s.\t acc@1: {performance['acc@1']:.2f}%"
                print(msg)
                log_file.write(msg + "\n")
                break

            scheduler_count += 1
            checkpoint_path = os.path.join(log_dir, "checkpoints", "checkpoint.pt")
            model.load_state_dict(torch.load(checkpoint_path))
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler_ES.step()

        if config.verbose:
            lr_msg = f"Current learning rate: {optim.param_groups[0]['lr']:.6f}"
            print(lr_msg)
            print("=" * 50)
            log_file.write(lr_msg + "\n")

        if config.debug:
            break

    return model, performance


def init_experiment_dir(config, dataset_name):
    """
    Create experiment directory with dataset name, model name, and timestamp.
    
    Format: experiments/{dataset_name}_LSTM_Baseline_{yyyyMMdd_hhmmss}/
    """
    # Get current time in GMT+7
    gmt7 = timezone(timedelta(hours=7))
    now = datetime.now(gmt7)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{dataset_name}_LSTM_Baseline_{timestamp}"
    experiment_dir = os.path.join(config.experiment_root, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    return experiment_dir


def save_results(experiment_dir, config, val_perf, test_perf, config_path):
    """Save all results to experiment directory."""
    # Save configuration
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        # Convert config to regular dict for YAML serialization
        config_dict = dict(config)
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Also copy the original config file
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(experiment_dir, "config_original.yaml"))
    
    # Save validation results
    val_results_path = os.path.join(experiment_dir, "val_results.json")
    val_results = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in val_perf.items()}
    with open(val_results_path, "w") as f:
        json.dump(val_results, f, indent=2)
    
    # Save test results
    test_results_path = os.path.join(experiment_dir, "test_results.json")
    test_results = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in test_perf.items()}
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Baseline model for next location prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config = EasyDict(config)

    # Setup seed
    seed = config.get("seed", 42)
    setup_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata to get dataset name
    metadata_path = os.path.join(config.data_dir, f"{config.dataset_prefix}_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    dataset_name = metadata["dataset_name"]

    # Initialize experiment directory
    experiment_dir = init_experiment_dir(config, dataset_name)
    print(f"Experiment directory: {experiment_dir}")

    # Create log file
    log_path = os.path.join(experiment_dir, "training.log")
    log_file = open(log_path, "w")
    log_file.write(f"Training LSTM Baseline model\n")
    log_file.write(f"Dataset: {dataset_name}\n")
    log_file.write(f"Config: {args.config}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write(f"Seed: {seed}\n")
    log_file.write("=" * 50 + "\n")

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"Data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    log_file.write(f"Data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}\n")

    # Create model
    model = LSTMBaseline(config=config, total_loc_num=config.total_loc_num).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    log_file.write(f"Total trainable parameters: {total_params}\n")
    log_file.write(f"Model config: emb_size={config.base_emb_size}, hidden_size={config.hidden_size}, num_layers={config.num_layers}\n")

    # Train
    model, val_performance = train_model(config, model, train_loader, val_loader, device, experiment_dir, log_file)

    # Load best checkpoint for testing
    checkpoint_path = os.path.join(experiment_dir, "checkpoints", "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    # Test
    log_file.write("\n=== Test Results ===\n")
    test_return_dict, test_user_dict = test(config, model, test_loader, device)
    test_performance = get_performance_dict(test_return_dict)

    test_msg = (
        f"Test Results:\n"
        f"  acc@1: {test_performance['acc@1']:.2f}%\n"
        f"  acc@5: {test_performance['acc@5']:.2f}%\n"
        f"  acc@10: {test_performance['acc@10']:.2f}%\n"
        f"  mrr: {test_performance['mrr']:.2f}%\n"
        f"  ndcg: {test_performance['ndcg']:.2f}%\n"
        f"  f1: {test_performance['f1'] * 100:.2f}%\n"
    )
    print(test_msg)
    log_file.write(test_msg)

    # Save results
    save_results(experiment_dir, config, val_performance, test_performance, args.config)

    log_file.write("\n=== Training Complete ===\n")
    log_file.close()

    print(f"\nResults saved to: {experiment_dir}")
    print(f"Test Acc@1: {test_performance['acc@1']:.2f}%")


if __name__ == "__main__":
    main()

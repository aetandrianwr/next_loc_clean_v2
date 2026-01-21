"""
Training script for LSTM and RNN Baseline Models.

This script handles training and evaluation of LSTM and RNN baseline models
for next location prediction. It is designed to be scientifically comparable
with the Pointer Generator Transformer model.

Key Design Decisions for Fair Comparison:
- Same data loading and preprocessing as train_pgt.py
- Same evaluation metrics from src/evaluation/metrics.py
- Same training procedure (optimizer, warmup + cosine LR schedule, early stopping)
- Same experiment directory structure
- Same random seed (42) for reproducibility

Usage:
    # Train LSTM on Geolife
    python scripts/baseline_lstm_rnn/train_baseline.py --config scripts/baseline_lstm_rnn/config_lstm_geolife.yaml
    
    # Train RNN on Geolife
    python scripts/baseline_lstm_rnn/train_baseline.py --config scripts/baseline_lstm_rnn/config_rnn_geolife.yaml
    
    # Train LSTM on DIY
    python scripts/baseline_lstm_rnn/train_baseline.py --config scripts/baseline_lstm_rnn/config_lstm_diy.yaml
    
    # Train RNN on DIY
    python scripts/baseline_lstm_rnn/train_baseline.py --config scripts/baseline_lstm_rnn/config_rnn_diy.yaml

This script handles:
- Configuration loading from YAML
- Data loading from preprocessed pickle files
- Model training with warmup + cosine LR schedule
- Early stopping on validation loss
- Evaluation using standard metrics from src/evaluation/metrics.py
- Saving checkpoints, logs, and metrics to experiments/ directory
"""

import os
import sys
import json
import yaml
import random
import argparse
import pickle
import math
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Tuple, Any
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.baselines.lstm_baseline import LSTMBaseline
from src.models.baselines.rnn_baseline import RNNBaseline
from src.evaluation.metrics import (
    calculate_correct_total_prediction,
    get_performance_dict,
)


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all relevant libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(path: str) -> Dict:
    """
    Load configuration from YAML file and flatten it.
    
    Args:
        path: Path to YAML config file
    
    Returns:
        Flattened configuration dictionary
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Keep original structure for model and training
    config = {
        'model': cfg.get('model', {}),
        'training': cfg.get('training', {}),
        'data': cfg.get('data', {}),
    }
    
    # Add seed at top level
    config['seed'] = cfg.get('seed', 42)
    
    # Add model type
    config['model_type'] = cfg.get('model_type', 'lstm')
    
    return config


# =============================================================================
# Dataset and DataLoader
# =============================================================================

class NextLocationDataset(Dataset):
    """
    Dataset for next location prediction.
    
    Loads preprocessed data from pickle files and provides:
    - Location sequence (X)
    - Target location (Y)
    - Temporal features: user, time, weekday, duration, diff
    """
    
    def __init__(self, data_path: str, build_user_history: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to pickle file containing preprocessed data
            build_user_history: Whether to build user history index
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.num_samples = len(self.data)
        self._compute_statistics()
        
        self.user_location_history = None
        self.user_location_freq = None
        if build_user_history:
            self._build_user_history()
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        all_locs = set()
        all_users = set()
        max_seq_len = 0
        
        for sample in self.data:
            all_locs.update(sample['X'].tolist())
            all_locs.add(sample['Y'])
            all_users.add(sample['user_X'][0])
            max_seq_len = max(max_seq_len, len(sample['X']))
        
        self.num_locations = max(all_locs) + 1
        self.num_users = max(all_users) + 1
        self.max_seq_len = max_seq_len
        self.unique_locations = len(all_locs)
    
    def _build_user_history(self):
        """Build user location history for personalized prediction."""
        from collections import defaultdict
        self.user_location_history = defaultdict(set)
        self.user_location_freq = defaultdict(lambda: defaultdict(int))
        
        for sample in self.data:
            user = sample['user_X'][0]
            for loc in sample['X']:
                self.user_location_history[user].add(loc)
                self.user_location_freq[user][loc] += 1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample = self.data[idx]
        
        return_dict = {
            'user': torch.tensor(sample['user_X'][0], dtype=torch.long),
            'weekday': torch.tensor(sample['weekday_X'], dtype=torch.long),
            'time': torch.tensor(sample['start_min_X'] // 15, dtype=torch.long),
            'duration': torch.tensor(sample['dur_X'] // 30, dtype=torch.long),
            'diff': torch.tensor(sample['diff'], dtype=torch.long),
        }
        
        x = torch.tensor(sample['X'], dtype=torch.long)
        y = torch.tensor(sample['Y'], dtype=torch.long)
        
        return x, y, return_dict


def collate_fn(batch):
    """
    Collate function to handle variable length sequences.
    
    Returns:
        x_batch: Padded sequence tensor (seq_len, batch_size)
        y_batch: Target tensor (batch_size,)
        x_dict: Dictionary with additional features
    """
    x_batch, y_batch = [], []
    
    x_dict_batch = {'len': []}
    for key in batch[0][-1]:
        x_dict_batch[key] = []
    
    for x, y, return_dict in batch:
        x_batch.append(x)
        y_batch.append(y)
        
        x_dict_batch['len'].append(len(x))
        for key in return_dict:
            x_dict_batch[key].append(return_dict[key])
    
    # Pad sequences (padding_value=0)
    x_batch = pad_sequence(x_batch, batch_first=False, padding_value=0)
    y_batch = torch.stack(y_batch)
    
    # Convert lists to tensors
    x_dict_batch['user'] = torch.stack(x_dict_batch['user'])
    x_dict_batch['len'] = torch.tensor(x_dict_batch['len'], dtype=torch.long)
    
    # Pad variable length features
    for key in ['weekday', 'time', 'duration', 'diff']:
        x_dict_batch[key] = pad_sequence(x_dict_batch[key], batch_first=False, padding_value=0)
    
    return x_batch, y_batch, x_dict_batch


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    data_dir = config['data']['data_dir']
    dataset_prefix = config['data']['dataset_prefix']
    batch_size = config['training'].get('batch_size', 128)
    num_workers = config['data'].get('num_workers', 0)
    
    train_path = os.path.join(data_dir, f"{dataset_prefix}_train.pk")
    val_path = os.path.join(data_dir, f"{dataset_prefix}_validation.pk")
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")
    
    # Create datasets
    train_ds = NextLocationDataset(train_path, build_user_history=True)
    val_ds = NextLocationDataset(val_path, build_user_history=False)
    test_ds = NextLocationDataset(test_path, build_user_history=False)
    
    # Share user history
    val_ds.user_location_history = train_ds.user_location_history
    val_ds.user_location_freq = train_ds.user_location_freq
    test_ds.user_location_history = train_ds.user_location_history
    test_ds.user_location_freq = train_ds.user_location_freq
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    info = {
        'num_locations': train_ds.num_locations,
        'num_users': train_ds.num_users,
        'max_seq_len': max(train_ds.max_seq_len, val_ds.max_seq_len, test_ds.max_seq_len),
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
    }
    
    return train_loader, val_loader, test_loader, info


# =============================================================================
# Trainer Class
# =============================================================================

class BaselineTrainer:
    """
    Trainer for LSTM and RNN baseline models.
    
    Features:
    - Mixed precision training (AMP)
    - Warmup + Cosine LR schedule (same as Pointer Generator Transformer)
    - Early stopping on validation loss
    - Gradient clipping
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device,
        experiment_dir: str,
        model_name: str,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.model_name = model_name
        
        # Create directories
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract training config
        train_cfg = config['training']
        
        # Loss function (same as Pointer Generator Transformer)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=train_cfg.get('label_smoothing', 0.03),
        )
        
        # Optimizer (same as Pointer Generator Transformer)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg.get('learning_rate', 3e-4),
            weight_decay=train_cfg.get('weight_decay', 0.015),
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        
        # Learning rate schedule parameters (same as Pointer Generator Transformer)
        self.num_epochs = train_cfg.get('num_epochs', 50)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 5)
        self.base_lr = train_cfg.get('learning_rate', 3e-4)
        self.min_lr = train_cfg.get('min_lr', 1e-6)
        
        # Mixed precision
        self.use_amp = train_cfg.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Gradient clipping (same as Pointer Generator Transformer)
        self.grad_clip = train_cfg.get('grad_clip', 0.8)
        
        # Early stopping
        self.patience = train_cfg.get('patience', 5)  # Required: patience=5
        self.min_epochs = train_cfg.get('min_epochs', 8)
        
        # State
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self._setup_logging()
    
    def _get_lr(self, epoch: int) -> float:
        """Get learning rate with warmup and cosine decay (same as Pointer Generator Transformer)."""
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        
        progress = (epoch - self.warmup_epochs) / max(1, self.num_epochs - self.warmup_epochs)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _setup_logging(self):
        """Setup logging."""
        log_file = self.experiment_dir / 'training.log'
        
        # Configure root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("=" * 60)
        self.logger.info(f"{self.model_name.upper()} BASELINE - Training Started")
        self.logger.info("=" * 60)
        self.logger.info(f"Model config: {self.config.get('model', {})}")
        self.logger.info(f"Training config: {self.config.get('training', {})}")
        self.logger.info("=" * 60)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for x, y, x_dict in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(x, x_dict)
                    loss = self.criterion(logits, y)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x, x_dict)
                loss = self.criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg': f"{total_loss/num_batches:.4f}",
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> Dict:
        """Evaluate on a dataset split."""
        self.model.eval()
        
        all_results = []
        all_true_y = []
        all_pred_y = []
        total_loss = 0.0
        num_batches = 0
        
        for x, y, x_dict in tqdm(loader, desc=f"Eval {split}"):
            x = x.to(self.device)
            y = y.to(self.device)
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(x, x_dict)
                    loss = self.criterion(logits, y)
            else:
                logits = self.model(x, x_dict)
                loss = self.criterion(logits, y)
            
            total_loss += loss.item()
            num_batches += 1
            
            results, true_y, pred_y = calculate_correct_total_prediction(logits, y)
            all_results.append(results)
            all_true_y.append(true_y)
            all_pred_y.append(pred_y)
        
        # Aggregate metrics
        total_results = np.sum(np.stack(all_results), axis=0)
        metrics = {
            "correct@1": total_results[0],
            "correct@3": total_results[1],
            "correct@5": total_results[2],
            "correct@10": total_results[3],
            "rr": total_results[4],
            "ndcg": total_results[5],
            "total": total_results[6],
        }
        
        # F1 score
        all_true_y = torch.cat(all_true_y).numpy()
        all_pred_y_flat = []
        for pred in all_pred_y:
            if not pred.shape:
                all_pred_y_flat.append(pred.item())
            else:
                all_pred_y_flat.extend(pred.tolist())
        metrics['f1'] = f1_score(all_true_y.tolist(), all_pred_y_flat, average='weighted', zero_division=0)
        
        perf = get_performance_dict(metrics)
        perf['loss'] = total_loss / num_batches
        
        return perf
    
    def train(self) -> Dict:
        """Full training loop."""
        self.logger.info(f"Training for {self.num_epochs} epochs")
        self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            
            # Set learning rate
            lr = self._get_lr(epoch)
            self._set_lr(lr)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.evaluate(self.val_loader, "val")
            
            self.logger.info(
                f"Epoch {self.current_epoch}/{self.num_epochs} | "
                f"LR: {lr:.2e} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_metrics['loss']:.4f} | "
                f"Acc@1: {val_metrics['acc@1']:.2f}%"
            )
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self._save_checkpoint("best.pt")
                self.logger.info(f"  ✓ New best (Acc@1: {val_metrics['acc@1']:.2f}%)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience and self.current_epoch >= self.min_epochs:
                    self.logger.info(f"Early stopping at epoch {self.current_epoch}")
                    break
        
        # Final test
        self._load_checkpoint("best.pt")
        val_metrics = self.evaluate(self.val_loader, "val")
        test_metrics = self.evaluate(self.test_loader, "test")
        
        self.logger.info("=" * 60)
        self.logger.info("FINAL VALIDATION RESULTS")
        self.logger.info(f"  Acc@1:  {val_metrics['acc@1']:.2f}%")
        self.logger.info(f"  Acc@5:  {val_metrics['acc@5']:.2f}%")
        self.logger.info(f"  Acc@10: {val_metrics['acc@10']:.2f}%")
        self.logger.info(f"  MRR:    {val_metrics['mrr']:.2f}%")
        self.logger.info(f"  NDCG:   {val_metrics['ndcg']:.2f}%")
        self.logger.info("=" * 60)
        self.logger.info("FINAL TEST RESULTS")
        self.logger.info(f"  Acc@1:  {test_metrics['acc@1']:.2f}%")
        self.logger.info(f"  Acc@5:  {test_metrics['acc@5']:.2f}%")
        self.logger.info(f"  Acc@10: {test_metrics['acc@10']:.2f}%")
        self.logger.info(f"  MRR:    {test_metrics['mrr']:.2f}%")
        self.logger.info(f"  NDCG:   {test_metrics['ndcg']:.2f}%")
        self.logger.info("=" * 60)
        
        return val_metrics, test_metrics
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'config': self.config,
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


# =============================================================================
# Experiment Management
# =============================================================================

def init_experiment_dir(config: Dict, dataset_name: str, model_name: str) -> str:
    """
    Create experiment directory with dataset name, model name, and timestamp.
    
    Format: experiments/{dataset_name}_{model_name}_{yyyyMMdd_hhmmss}/
    
    Args:
        config: Configuration dictionary
        dataset_name: Name of the dataset
        model_name: Name of the model (lstm_baseline or rnn_baseline)
        
    Returns:
        Path to experiment directory
    """
    experiment_root = config['data'].get('experiment_root', 'experiments')
    
    # Get current time in GMT+7
    gmt7 = timezone(timedelta(hours=7))
    now = datetime.now(gmt7)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{dataset_name}_{model_name}_{timestamp}"
    experiment_dir = os.path.join(experiment_root, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    return experiment_dir


def save_results(experiment_dir: str, config: Dict, val_perf: Dict, test_perf: Dict, config_path: str):
    """
    Save all results to experiment directory.
    
    Saves:
    - Configuration (YAML)
    - Validation results (JSON)
    - Test results (JSON)
    """
    # Save configuration
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Copy original config file
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(experiment_dir, "config_original.yaml"))
    
    # Save validation results
    val_results_path = os.path.join(experiment_dir, "val_results.json")
    val_results = {k: float(v) if isinstance(v, (np.floating, np.integer, float)) else v for k, v in val_perf.items()}
    with open(val_results_path, "w") as f:
        json.dump(val_results, f, indent=2)
    
    # Save test results
    test_results_path = os.path.join(experiment_dir, "test_results.json")
    test_results = {k: float(v) if isinstance(v, (np.floating, np.integer, float)) else v for k, v in test_perf.items()}
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LSTM/RNN baseline for next location prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup seed (REQUIRED: seed=42)
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model type
    model_type = config.get('model_type', 'lstm').lower()
    model_name = f"{model_type}_baseline"
    
    # Load metadata to get dataset name
    data_dir = config['data']['data_dir']
    dataset_prefix = config['data']['dataset_prefix']
    metadata_path = os.path.join(data_dir, f"{dataset_prefix}_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    dataset_name = metadata["dataset_name"]
    
    # Initialize experiment directory
    experiment_dir = init_experiment_dir(config, dataset_name, model_name)
    print(f"Experiment directory: {experiment_dir}")
    
    print("=" * 60)
    print(f"{model_name.upper()} - Baseline Model")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    
    # Get dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, info = get_dataloaders(config)
    
    print(f"  Locations: {info['num_locations']}")
    print(f"  Users: {info['num_users']}")
    print(f"  Max sequence length: {info['max_seq_len']}")
    print(f"  Train: {info['train_size']}, Val: {info['val_size']}, Test: {info['test_size']}")
    print(f"  Data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    # Create model
    model_cfg = config['model']
    
    if model_type == 'lstm':
        model = LSTMBaseline(
            num_locations=info['num_locations'],
            num_users=info['num_users'],
            d_model=model_cfg.get('d_model', 64),
            hidden_size=model_cfg.get('hidden_size', 128),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.15),
            max_seq_len=info['max_seq_len'] + 10,
        )
    elif model_type == 'rnn':
        model = RNNBaseline(
            num_locations=info['num_locations'],
            num_users=info['num_users'],
            d_model=model_cfg.get('d_model', 64),
            hidden_size=model_cfg.get('hidden_size', 128),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.15),
            max_seq_len=info['max_seq_len'] + 10,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm' or 'rnn'.")
    
    print(f"\nModel: {model_name}")
    print(f"  d_model: {model_cfg.get('d_model', 64)}")
    print(f"  hidden_size: {model_cfg.get('hidden_size', 128)}")
    print(f"  num_layers: {model_cfg.get('num_layers', 2)}")
    print(f"  dropout: {model_cfg.get('dropout', 0.15)}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        experiment_dir=experiment_dir,
        model_name=model_name,
    )
    
    # Train
    print(f"\nStarting training...")
    val_metrics, test_metrics = trainer.train()
    
    # Save results
    save_results(experiment_dir, config, val_metrics, test_metrics, args.config)
    
    print(f"\n" + "=" * 60)
    print(f"{model_name.upper()} RESULTS - {dataset_name.upper()}")
    print("=" * 60)
    print(f"Acc@1:  {test_metrics['acc@1']:.2f}%")
    print(f"Acc@5:  {test_metrics['acc@5']:.2f}%")
    print(f"Acc@10: {test_metrics['acc@10']:.2f}%")
    print(f"MRR:    {test_metrics['mrr']:.2f}%")
    print(f"NDCG:   {test_metrics['ndcg']:.2f}%")
    print("=" * 60)
    print(f"\nResults saved to: {experiment_dir}")
    
    return test_metrics


if __name__ == "__main__":
    main()

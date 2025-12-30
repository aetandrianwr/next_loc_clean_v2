# coding: utf-8
"""
Training script for DeepMove Model.

This script implements the training pipeline for DeepMove model adapted to work
with the next_loc_clean_v2 data format.

Usage:
    python src/training/train_deepmove.py --config config/models/config_deepmove_geolife.yaml
    python src/training/train_deepmove.py --config config/models/config_deepmove_diy.yaml

Paper Reference:
    DeepMove: Predicting Human Mobility with Attentional Recurrent Networks
    Jie Feng, Yong Li, Chao Zhang, et al.
    WWW 2018

This script handles:
- Configuration loading from YAML
- Data loading and preprocessing for DeepMove format
- Model training with early stopping
- Evaluation using src/evaluation/metrics.py
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
from collections import deque, Counter
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.baseline.deepmove import (
    DeepMoveModel,
    TrajPreAttnAvgLongUser,
    TrajPreSimple,
    TrajPreLocalAttnLong,
    create_deepmove_config
)
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


def load_config(path):
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        if isinstance(value, dict):
            for k, v in value.items():
                config[k] = v
        else:
            config[key] = value

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


class DeepMoveDataLoader:
    """
    Data loader that converts next_loc_clean_v2 format to DeepMove format.
    
    The original DeepMove expects data with:
    - Sessions organized by user
    - Historical trajectories accumulated from ALL previous sessions
    - Time slots (48 half-hour slots per day)
    
    This loader converts from the preprocessed format (list of dicts with X, Y, etc.)
    to the format expected by DeepMove training functions, properly accumulating
    history across sessions for each user.
    """
    
    def __init__(self, data_path, mode='train', history_mode='avg', use_cuda=True, 
                 train_data_path=None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to pickle file
            mode: 'train', 'val', or 'test'
            history_mode: 'avg' for averaging embeddings at same time slot
            use_cuda: Whether to use GPU tensors
            train_data_path: Path to training data (for val/test to include train history)
        """
        self.data = pickle.load(open(data_path, 'rb'))
        self.mode = mode
        self.history_mode = history_mode
        self.use_cuda = use_cuda
        
        # Load training data for history (for val/test modes)
        if mode in ['val', 'test'] and train_data_path:
            self.train_data = pickle.load(open(train_data_path, 'rb'))
        else:
            self.train_data = None
        
        # Group data by user and convert to DeepMove format
        self.processed_data = self._convert_data_with_accumulated_history()
    
    def _convert_data_with_accumulated_history(self):
        """
        Convert data to DeepMove format with proper accumulated history.
        
        In DeepMove, history for each sample includes:
        - For training: all previous training samples from the same user
        - For val/test: all training samples + all previous val/test samples
        """
        processed = []
        
        # Group samples by user
        user_samples = {}
        for idx, sample in enumerate(self.data):
            user = sample['user_X'][0]
            if user not in user_samples:
                user_samples[user] = []
            user_samples[user].append((idx, sample))
        
        # If val/test, pre-compute user history from training data
        user_train_history = {}
        if self.train_data is not None:
            for sample in self.train_data:
                user = sample['user_X'][0]
                if user not in user_train_history:
                    user_train_history[user] = []
                
                loc_seq = sample['X']
                time_seq = sample['start_min_X']
                tim_seq = (time_seq // 30) % 48
                
                for loc, tim in zip(loc_seq, tim_seq):
                    user_train_history[user].append((int(loc), int(tim)))
        
        # Process each user's samples with accumulated history
        for user, samples in user_samples.items():
            # Initialize accumulated history for this user
            accumulated_history = []
            
            # Add training history for val/test
            if user in user_train_history:
                accumulated_history = list(user_train_history[user])
            
            for sample_idx, (idx, sample) in enumerate(samples):
                loc_seq = sample['X']
                time_seq = sample['start_min_X']
                target = sample['Y']
                
                tim_seq = (time_seq // 30) % 48
                
                if len(loc_seq) < 2:
                    # Add to history even if skipping
                    for loc, tim in zip(loc_seq, tim_seq):
                        accumulated_history.append((int(loc), int(tim)))
                    continue
                
                # Skip first training sample (no history) - matching original DeepMove
                if self.mode == 'train' and sample_idx == 0:
                    # Still accumulate this sample's data for future history
                    for loc, tim in zip(loc_seq, tim_seq):
                        accumulated_history.append((int(loc), int(tim)))
                    continue
                
                # Use accumulated history (sorted by time)
                if len(accumulated_history) > 0:
                    history_sorted = sorted(accumulated_history, key=lambda x: x[1])
                    history_loc = np.array([h[0] for h in history_sorted])
                    history_tim = np.array([h[1] for h in history_sorted])
                    history_count = self._compute_history_count(history_tim)
                else:
                    # Fallback: use current sequence as history if no accumulated history
                    history_loc = loc_seq[:-1]
                    history_tim = tim_seq[:-1]
                    history_count = self._compute_history_count(history_tim)
                
                # Current sequence
                loc = loc_seq
                tim = tim_seq
                
                trace = {
                    'loc': Variable(torch.LongTensor(loc.reshape(-1, 1))),
                    'tim': Variable(torch.LongTensor(tim.reshape(-1, 1))),
                    'target': Variable(torch.LongTensor([target])),
                    'history_loc': Variable(torch.LongTensor(history_loc.reshape(-1, 1))),
                    'history_tim': Variable(torch.LongTensor(history_tim.reshape(-1, 1))),
                    'history_count': history_count,
                    'uid': user,
                    'idx': idx
                }
                
                processed.append(trace)
                
                # Add current sequence to accumulated history for future samples
                for loc_val, tim_val in zip(loc_seq, tim_seq):
                    accumulated_history.append((int(loc_val), int(tim_val)))
        
        return processed
    
    def _compute_history_count(self, history_tim):
        """Compute count of locations per time slot for averaging."""
        if len(history_tim) == 0:
            return [1]
        
        history_count = [1]
        last_t = history_tim[0]
        count = 1
        
        for t in history_tim[1:]:
            if t == last_t:
                count += 1
            else:
                history_count[-1] = count
                history_count.append(1)
                last_t = t
                count = 1
        history_count[-1] = count
        
        return history_count
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]


def generate_queue(data_loader, mode='random'):
    """Generate a queue of (index) for training/testing."""
    indices = list(range(len(data_loader)))
    queue = deque()
    
    if mode == 'random':
        np.random.shuffle(indices)
    
    for idx in indices:
        queue.append(idx)
    
    return queue


def get_acc_from_scores(target, scores, k_list=[1, 5, 10]):
    """
    Calculate top-k accuracy from scores.
    
    Args:
        target: Target tensor
        scores: Score tensor [batch, num_classes]
        k_list: List of k values for top-k accuracy
    
    Returns:
        Dictionary with acc@k values
    """
    target_np = target.data.cpu().numpy()
    val, idxx = scores.data.topk(max(k_list), 1)
    predx = idxx.cpu().numpy()
    
    acc = {}
    for k in k_list:
        correct = 0
        for i, p in enumerate(predx):
            t = target_np[i]
            if t in p[:k] and t > 0:
                correct += 1
        acc[k] = correct
    
    return acc, len(target_np)


def run_deepmove(data_loader, mode, lr, clip, model, optimizer, criterion, 
                 model_mode='attn_avg_long_user', use_cuda=True, verbose=True):
    """
    Run one epoch of training or evaluation for DeepMove.
    
    Uses per-user macro-averaging for metrics (matching original DeepMove paper).
    
    Args:
        data_loader: DeepMoveDataLoader instance
        mode: 'train' or 'test'
        lr: Learning rate
        clip: Gradient clipping value
        model: DeepMove model
        optimizer: Optimizer
        criterion: Loss function (NLLLoss)
        model_mode: Model variant
        use_cuda: Whether to use GPU
        verbose: Print progress
    
    Returns:
        For train: model, avg_loss, metrics
        For test: avg_loss, metrics
    """
    if mode == 'train':
        model.train()
        run_queue = generate_queue(data_loader, 'random')
    else:
        model.eval()
        run_queue = generate_queue(data_loader, 'normal')
    
    total_loss = []
    
    # Per-user accumulation for macro-averaging (matching DeepMove paper)
    users_acc = {}  # user -> [total_samples, correct@1, correct@5, correct@10]
    
    # Also collect for micro-averaged metrics 
    all_scores = []
    all_targets = []
    
    queue_len = len(run_queue)
    
    for c in range(queue_len):
        idx = run_queue.popleft()
        sample = data_loader[idx]
        user_id = sample['uid']
        
        if user_id not in users_acc:
            users_acc[user_id] = [0, 0, 0, 0]  # total, correct@1, correct@5, correct@10
        
        if mode == 'train':
            optimizer.zero_grad()
        
        # Get data
        loc = sample['loc']
        tim = sample['tim']
        target = sample['target']
        uid = Variable(torch.LongTensor([user_id]))
        
        if use_cuda:
            loc = loc.cuda()
            tim = tim.cuda()
            target = target.cuda()
            uid = uid.cuda()
        
        # Handle different model modes
        if model_mode in ['attn_avg_long_user']:
            history_loc = sample['history_loc']
            history_tim = sample['history_tim']
            history_count = sample['history_count']
            
            if use_cuda:
                history_loc = history_loc.cuda()
                history_tim = history_tim.cuda()
            
            target_len = target.size()[0]
            scores = model.model(loc, tim, history_loc, history_tim, history_count, uid, target_len)
        elif model_mode in ['simple', 'simple_long']:
            scores = model.model(loc, tim)
        elif model_mode == 'attn_local_long':
            target_len = target.size()[0]
            scores = model.model(loc, tim, target_len)
        else:
            # Default to attn_avg_long_user
            history_loc = sample['history_loc']
            history_tim = sample['history_tim']
            history_count = sample['history_count']
            
            if use_cuda:
                history_loc = history_loc.cuda()
                history_tim = history_tim.cuda()
            
            target_len = target.size()[0]
            scores = model.model(loc, tim, history_loc, history_tim, history_count, uid, target_len)
        
        # Handle score dimensions
        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]
        
        loss = criterion(scores, target)
        
        if mode == 'train':
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.data.add_(p.grad.data, alpha=-lr)
            except:
                pass
            optimizer.step()
        
        total_loss.append(loss.item())
        
        # Per-user accuracy accumulation
        target_np = target.data.cpu().numpy()
        val, idxx = scores.data.topk(10, 1)
        predx = idxx.cpu().numpy()
        
        for i, p in enumerate(predx):
            t = target_np[i]
            users_acc[user_id][0] += 1  # total
            if t in p[:10] and t > 0:
                users_acc[user_id][3] += 1  # correct@10
            if t in p[:5] and t > 0:
                users_acc[user_id][2] += 1  # correct@5
            if t == p[0] and t > 0:
                users_acc[user_id][1] += 1  # correct@1
        
        # Collect for micro-averaged metrics 
        all_scores.append(scores.detach())
        all_targets.append(target.detach())
    
    avg_loss = np.mean(total_loss, dtype=np.float64)
    
    # Calculate macro-averaged metrics (per-user average, matching DeepMove paper)
    user_acc1_list = []
    user_acc5_list = []
    user_acc10_list = []
    
    for user_id in users_acc:
        total = users_acc[user_id][0]
        if total > 0:
            user_acc1_list.append(users_acc[user_id][1] / total)
            user_acc5_list.append(users_acc[user_id][2] / total)
            user_acc10_list.append(users_acc[user_id][3] / total)
    
    macro_acc1 = np.mean(user_acc1_list) * 100 if user_acc1_list else 0
    macro_acc5 = np.mean(user_acc5_list) * 100 if user_acc5_list else 0
    macro_acc10 = np.mean(user_acc10_list) * 100 if user_acc10_list else 0
    
    # Also calculate micro-averaged metrics using standard evaluation
    all_scores_tensor = torch.cat(all_scores, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    
    micro_metrics = calculate_metrics(all_scores_tensor.exp(), all_targets_tensor)
    
    # Combine metrics - use macro for reporting (matches DeepMove paper)
    metrics = {
        'acc@1': macro_acc1,
        'acc@5': macro_acc5,
        'acc@10': macro_acc10,
        'micro_acc@1': micro_metrics['acc@1'],
        'micro_acc@5': micro_metrics['acc@5'],
        'micro_acc@10': micro_metrics['acc@10'],
        'mrr': micro_metrics['mrr'],
        'ndcg': micro_metrics['ndcg'],
        'f1': micro_metrics['f1'],
    }
    
    if mode == 'train':
        return model, avg_loss, metrics
    else:
        return avg_loss, metrics


def init_experiment_dir(config, dataset_name):
    """
    Create experiment directory with dataset name, model name, and timestamp.
    
    Format: experiments/{dataset_name}_deepmove_{yyyyMMdd_hhmmss}/
    """
    gmt7 = timezone(timedelta(hours=7))
    now = datetime.now(gmt7)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{dataset_name}_deepmove_{timestamp}"
    experiment_dir = os.path.join(config.experiment_root, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    return experiment_dir


def save_results(experiment_dir, config, val_perf, test_perf, config_path):
    """Save all results to experiment directory."""
    # Save configuration
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        config_dict = dict(config)
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Copy original config
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


class EarlyStopping:
    """Early stopping based on validation loss."""
    
    def __init__(self, log_dir, patience=7, verbose=False, delta=0):
        self.log_dir = log_dir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metrics = None
        self.delta = delta
    
    def __call__(self, val_loss, metrics, model):
        score = -val_loss  # We want to minimize loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, metrics, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, metrics, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, metrics, model):
        if self.verbose:
            print(f"Validation loss decreased. Saving model...")
        checkpoint_path = os.path.join(self.log_dir, "checkpoints", "checkpoint.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        self.best_metrics = metrics


def main():
    parser = argparse.ArgumentParser(description="Train DeepMove model for next location prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config = EasyDict(config)

    # Setup seed
    seed = config.get("seed", 42)
    setup_seed(seed)

    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Load metadata
    metadata_path = os.path.join(config.data_dir, f"{config.dataset_prefix}_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    dataset_name = metadata["dataset_name"]
    
    # Update config with metadata
    config.loc_size = metadata["total_loc_num"]
    config.uid_size = metadata["total_user_num"]

    # Initialize experiment directory
    experiment_dir = init_experiment_dir(config, dataset_name)
    print(f"Experiment directory: {experiment_dir}")

    # Create log file
    log_path = os.path.join(experiment_dir, "training.log")
    log_file = open(log_path, "w")
    log_file.write(f"Training DeepMove model\n")
    log_file.write(f"Dataset: {dataset_name}\n")
    log_file.write(f"Config: {args.config}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write(f"Seed: {seed}\n")
    log_file.write("=" * 50 + "\n")

    # Load data with accumulated history
    train_path = os.path.join(config.data_dir, f"{config.dataset_prefix}_train.pk")
    val_path = os.path.join(config.data_dir, f"{config.dataset_prefix}_validation.pk")
    test_path = os.path.join(config.data_dir, f"{config.dataset_prefix}_test.pk")

    train_loader = DeepMoveDataLoader(train_path, mode='train', 
                                      history_mode=config.get('history_mode', 'avg'),
                                      use_cuda=use_cuda)
    val_loader = DeepMoveDataLoader(val_path, mode='val',
                                    history_mode=config.get('history_mode', 'avg'),
                                    use_cuda=use_cuda,
                                    train_data_path=train_path)
    test_loader = DeepMoveDataLoader(test_path, mode='test',
                                     history_mode=config.get('history_mode', 'avg'),
                                     use_cuda=use_cuda,
                                     train_data_path=train_path)

    print(f"Data loaded: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    log_file.write(f"Data: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}\n")

    # Create model configuration
    model_config = create_deepmove_config(
        loc_size=config.loc_size,
        uid_size=config.uid_size,
        loc_emb_size=config.get('loc_emb_size', 500),
        tim_emb_size=config.get('tim_emb_size', 10),
        uid_emb_size=config.get('uid_emb_size', 40),
        hidden_size=config.get('hidden_size', 500),
        dropout_p=config.get('dropout_p', 0.3),
        rnn_type=config.get('rnn_type', 'GRU'),
        attn_type=config.get('attn_type', 'dot'),
        model_mode=config.get('model_mode', 'attn_avg_long_user'),
        use_cuda=use_cuda,
        tim_size=config.get('tim_size', 48)
    )

    # Create model
    model = DeepMoveModel(model_config)
    if use_cuda:
        model = model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    log_file.write(f"Total trainable parameters: {total_params}\n")

    # Setup optimizer and criterion
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('lr', 5e-4),
        weight_decay=config.get('L2', 1e-5)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', 
        patience=config.get('lr_step', 2),
        factor=config.get('lr_decay', 0.1),
        threshold=1e-3
    )

    # Training settings
    lr = config.get('lr', 5e-4)
    clip = config.get('clip', 5.0)
    max_epoch = config.get('epoch_max', 30)
    model_mode = config.get('model_mode', 'attn_avg_long_user')
    
    # Early stopping
    early_stopping = EarlyStopping(
        experiment_dir, 
        patience=config.get('patience', 5),
        verbose=config.get('verbose', True)
    )

    print("=" * 50)
    print("Starting training...")
    log_file.write("\n=== Training ===\n")

    best_val_acc = 0
    best_metrics = None

    for epoch in range(max_epoch):
        # Train
        model, train_loss, train_metrics = run_deepmove(
            train_loader, 'train', lr, clip, model, optimizer, criterion,
            model_mode=model_mode, use_cuda=use_cuda, verbose=config.get('verbose', True)
        )
        
        train_msg = (
            f"Epoch {epoch+1}/{max_epoch} - Train Loss: {train_loss:.4f}, "
            f"Acc@1: {train_metrics['acc@1']:.2f}%, "
            f"Acc@5: {train_metrics['acc@5']:.2f}%"
        )
        print(train_msg)
        log_file.write(train_msg + "\n")

        # Validate
        val_loss, val_metrics = run_deepmove(
            val_loader, 'test', lr, clip, model, optimizer, criterion,
            model_mode=model_mode, use_cuda=use_cuda, verbose=config.get('verbose', True)
        )
        
        val_msg = (
            f"         Val   Loss: {val_loss:.4f}, "
            f"Acc@1: {val_metrics['acc@1']:.2f}%, "
            f"Acc@5: {val_metrics['acc@5']:.2f}%, "
            f"Acc@10: {val_metrics['acc@10']:.2f}%"
        )
        print(val_msg)
        log_file.write(val_msg + "\n")

        # Update learning rate scheduler
        scheduler.step(val_metrics['acc@1'])
        
        # Update learning rate from optimizer
        lr = optimizer.param_groups[0]['lr']

        # Early stopping check
        early_stopping(val_loss, val_metrics, model)
        
        if val_metrics['acc@1'] > best_val_acc:
            best_val_acc = val_metrics['acc@1']
            best_metrics = val_metrics

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            log_file.write(f"Early stopping at epoch {epoch+1}\n")
            break

        log_file.flush()

    # Load best model for testing
    checkpoint_path = os.path.join(experiment_dir, "checkpoints", "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    # Test
    print("\n" + "=" * 50)
    print("Testing...")
    log_file.write("\n=== Test Results ===\n")

    test_loss, test_metrics = run_deepmove(
        test_loader, 'test', lr, clip, model, optimizer, criterion,
        model_mode=model_mode, use_cuda=use_cuda, verbose=config.get('verbose', True)
    )

    test_msg = (
        f"Test Results:\n"
        f"  Acc@1:  {test_metrics['acc@1']:.2f}%\n"
        f"  Acc@5:  {test_metrics['acc@5']:.2f}%\n"
        f"  Acc@10: {test_metrics['acc@10']:.2f}%\n"
        f"  MRR:    {test_metrics['mrr']:.2f}%\n"
        f"  NDCG:   {test_metrics['ndcg']:.2f}%\n"
        f"  F1:     {test_metrics['f1']:.2f}%\n"
    )
    print(test_msg)
    log_file.write(test_msg)

    # Save results
    val_perf = early_stopping.best_metrics if early_stopping.best_metrics else best_metrics
    save_results(experiment_dir, config, val_perf, test_metrics, args.config)

    log_file.write("\n=== Training Complete ===\n")
    log_file.close()

    print(f"\nResults saved to: {experiment_dir}")
    print(f"Test Acc@1: {test_metrics['acc@1']:.2f}%")
    print(f"Test Acc@5: {test_metrics['acc@5']:.2f}%")
    print(f"Test Acc@10: {test_metrics['acc@10']:.2f}%")


if __name__ == "__main__":
    main()

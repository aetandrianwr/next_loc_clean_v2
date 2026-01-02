#!/usr/bin/env python3
"""
Sequence Length Days Experiment - Nature Journal Standard

This script evaluates the impact of historical sequence length (in days) on
next location prediction performance. It uses pre-trained checkpoints and
filters test data to simulate different previous day windows.

Scientific Methodology:
1. Load pre-trained model checkpoint (trained on prev7 data)
2. For each sequence length (prev1 to prev7):
   - Filter test sequences to include only visits within the specified window
   - Evaluate model performance using standard metrics
3. Aggregate results for statistical analysis

Metrics Computed:
- correct@1, correct@3, correct@5, correct@10: Top-k hit counts
- acc@1, acc@5, acc@10: Top-k accuracy percentages
- mrr: Mean Reciprocal Rank
- ndcg: Normalized Discounted Cumulative Gain @ 10
- f1: Weighted F1 score
- loss: Cross-entropy loss
- total: Total number of samples
- rr: Sum of reciprocal ranks (raw)

Usage:
    python evaluate_sequence_length.py --dataset diy
    python evaluate_sequence_length.py --dataset geolife
    python evaluate_sequence_length.py --dataset all

Author: PhD Research - Next Location Prediction
Date: 2026-01-02
"""

import os
import sys
import json
import yaml
import pickle
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.proposed.pointer_v45 import PointerNetworkV45
from src.evaluation.metrics import (
    calculate_correct_total_prediction,
    get_performance_dict,
)

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'diy': {
        'checkpoint_path': '/data/next_loc_clean_v2/experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt',
        'config_path': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pointer_v45_diy_trial09.yaml',
        'test_data_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_test.pk',
        'train_data_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_train.pk',
        'dataset_name': 'DIY',
    },
    'geolife': {
        'checkpoint_path': '/data/next_loc_clean_v2/experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt',
        'config_path': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml',
        'test_data_path': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_test.pk',
        'train_data_path': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_train.pk',
        'dataset_name': 'GeoLife',
    },
}

SEED = 42


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    config = {
        'model': cfg.get('model', {}),
        'training': cfg.get('training', {}),
        'data': cfg.get('data', {}),
    }
    config['seed'] = cfg.get('seed', 42)
    return config


class SequenceLengthDataset(Dataset):
    """
    Dataset that filters sequences based on previous day window.
    
    This dataset loads pre-processed data and filters each sequence
    to only include visits within the specified number of previous days.
    This allows evaluation of how different sequence lengths affect
    model performance.
    
    Args:
        data_path: Path to pickle file containing preprocessed data
        previous_days: Number of previous days to include (1-7)
        min_seq_length: Minimum sequence length after filtering (default: 1)
    """
    
    def __init__(
        self, 
        data_path: str, 
        previous_days: int = 7,
        min_seq_length: int = 1,
    ):
        with open(data_path, 'rb') as f:
            original_data = pickle.load(f)
        
        self.previous_days = previous_days
        self.min_seq_length = min_seq_length
        
        # Filter data based on previous_days
        self.data = self._filter_sequences(original_data, previous_days)
        self.num_samples = len(self.data)
        self._compute_statistics()
    
    def _filter_sequences(
        self, 
        original_data: List[Dict], 
        previous_days: int
    ) -> List[Dict]:
        """
        Filter sequences to include only visits within previous_days window.
        
        The 'diff' field in each sample contains the number of days ago
        for each visit. We filter to keep only visits where diff <= previous_days.
        """
        filtered_data = []
        
        for sample in original_data:
            # diff contains days ago for each position in sequence
            diff = sample['diff']
            
            # Create mask for positions within the time window
            mask = diff <= previous_days
            
            # Skip if no valid positions after filtering
            if mask.sum() < self.min_seq_length:
                continue
            
            # Filter all sequence features using the mask
            filtered_sample = {
                'X': sample['X'][mask],
                'user_X': sample['user_X'][mask],
                'weekday_X': sample['weekday_X'][mask],
                'start_min_X': sample['start_min_X'][mask],
                'dur_X': sample['dur_X'][mask],
                'diff': sample['diff'][mask],
                'Y': sample['Y'],
            }
            
            filtered_data.append(filtered_sample)
        
        return filtered_data
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        all_locs = set()
        all_users = set()
        max_seq_len = 0
        seq_lengths = []
        
        for sample in self.data:
            all_locs.update(sample['X'].tolist())
            all_locs.add(sample['Y'])
            all_users.add(sample['user_X'][0])
            seq_len = len(sample['X'])
            max_seq_len = max(max_seq_len, seq_len)
            seq_lengths.append(seq_len)
        
        self.num_locations = max(all_locs) + 1 if all_locs else 0
        self.num_users = max(all_users) + 1 if all_users else 0
        self.max_seq_len = max_seq_len
        self.unique_locations = len(all_locs)
        self.avg_seq_len = np.mean(seq_lengths) if seq_lengths else 0
        self.std_seq_len = np.std(seq_lengths) if seq_lengths else 0
    
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
    """Collate function for variable length sequences."""
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
    
    # Pad sequences
    x_batch = pad_sequence(x_batch, batch_first=False, padding_value=0)
    y_batch = torch.stack(y_batch)
    
    # Convert lists to tensors
    x_dict_batch['user'] = torch.stack(x_dict_batch['user'])
    x_dict_batch['len'] = torch.tensor(x_dict_batch['len'], dtype=torch.long)
    
    # Pad variable length features
    for key in ['weekday', 'time', 'duration', 'diff']:
        x_dict_batch[key] = pad_sequence(x_dict_batch[key], batch_first=False, padding_value=0)
    
    return x_batch, y_batch, x_dict_batch


def get_dataset_info(train_path: str) -> Dict:
    """Get dataset information from training data for model initialization."""
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    all_locs = set()
    all_users = set()
    max_seq_len = 0
    
    for sample in train_data:
        all_locs.update(sample['X'].tolist())
        all_locs.add(sample['Y'])
        all_users.add(sample['user_X'][0])
        max_seq_len = max(max_seq_len, len(sample['X']))
    
    return {
        'num_locations': max(all_locs) + 1,
        'num_users': max(all_users) + 1,
        'max_seq_len': max_seq_len,
    }


def load_model(
    checkpoint_path: str,
    config: Dict,
    dataset_info: Dict,
    device: torch.device
) -> PointerNetworkV45:
    """Load model from checkpoint."""
    model_cfg = config['model']
    
    # Load checkpoint first to determine max_seq_len from saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer max_seq_len from checkpoint (position_bias has shape [max_seq_len])
    checkpoint_max_seq_len = checkpoint['model_state_dict']['position_bias'].shape[0]
    
    model = PointerNetworkV45(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=model_cfg.get('d_model', 128),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 3),
        dim_feedforward=model_cfg.get('dim_feedforward', 256),
        dropout=model_cfg.get('dropout', 0.15),
        max_seq_len=checkpoint_max_seq_len,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_smoothing: float = 0.0
) -> Dict:
    """
    Evaluate model and return comprehensive metrics.
    
    Returns dict with:
    - correct@1, correct@3, correct@5, correct@10
    - acc@1, acc@5, acc@10
    - mrr, ndcg, f1
    - loss, total, rr
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
    
    all_results = []
    all_true_y = []
    all_pred_y = []
    total_loss = 0.0
    num_batches = 0
    
    for x, y, x_dict in tqdm(dataloader, desc="Evaluating", leave=False):
        x = x.to(device)
        y = y.to(device)
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        
        with torch.cuda.amp.autocast():
            logits = model(x, x_dict)
            loss = criterion(logits, y)
        
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
    perf['loss'] = total_loss / max(num_batches, 1)
    
    return perf


def run_sequence_length_experiment(
    dataset_key: str,
    output_dir: str,
    batch_size: int = 64
) -> Dict:
    """
    Run sequence length experiment for a specific dataset.
    
    Args:
        dataset_key: 'diy' or 'geolife'
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing results for all sequence lengths
    """
    config = EXPERIMENT_CONFIG[dataset_key]
    
    print(f"\n{'='*80}")
    print(f"SEQUENCE LENGTH EXPERIMENT - {config['dataset_name'].upper()}")
    print(f"{'='*80}")
    
    # Load model config
    model_config = load_config(config['config_path'])
    
    # Get dataset info from training data
    dataset_info = get_dataset_info(config['train_data_path'])
    print(f"Dataset Info: {dataset_info}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from: {config['checkpoint_path']}")
    model = load_model(
        config['checkpoint_path'],
        model_config,
        dataset_info,
        device
    )
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Run experiments for prev1 to prev7
    results = {}
    sequence_lengths = list(range(1, 8))  # 1 to 7
    
    for prev_days in sequence_lengths:
        print(f"\n{'-'*60}")
        print(f"Evaluating with previous_days = {prev_days}")
        print(f"{'-'*60}")
        
        # Create filtered dataset
        test_dataset = SequenceLengthDataset(
            config['test_data_path'],
            previous_days=prev_days,
            min_seq_length=1
        )
        
        print(f"  Samples: {len(test_dataset)}")
        print(f"  Avg sequence length: {test_dataset.avg_seq_len:.2f} ± {test_dataset.std_seq_len:.2f}")
        print(f"  Max sequence length: {test_dataset.max_seq_len}")
        
        if len(test_dataset) == 0:
            print(f"  WARNING: No samples for prev_days={prev_days}")
            continue
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Evaluate
        label_smoothing = model_config['training'].get('label_smoothing', 0.0)
        metrics = evaluate_model(model, test_loader, device, label_smoothing)
        
        # Store results
        results[prev_days] = {
            'metrics': metrics,
            'num_samples': len(test_dataset),
            'avg_seq_len': float(test_dataset.avg_seq_len),
            'std_seq_len': float(test_dataset.std_seq_len),
            'max_seq_len': int(test_dataset.max_seq_len),
        }
        
        # Print summary
        print(f"  Results:")
        print(f"    Acc@1:  {metrics['acc@1']:.2f}%")
        print(f"    Acc@5:  {metrics['acc@5']:.2f}%")
        print(f"    Acc@10: {metrics['acc@10']:.2f}%")
        print(f"    MRR:    {metrics['mrr']:.2f}%")
        print(f"    NDCG:   {metrics['ndcg']:.2f}%")
        print(f"    F1:     {metrics['f1']*100:.2f}%")
        print(f"    Loss:   {metrics['loss']:.4f}")
    
    # Save results
    output_path = os.path.join(output_dir, f'{dataset_key}_sequence_length_results.json')
    
    # Convert numpy types for JSON serialization
    results_json = {}
    for k, v in results.items():
        results_json[str(k)] = {
            'metrics': {mk: float(mv) if isinstance(mv, (np.floating, np.integer)) else mv 
                       for mk, mv in v['metrics'].items()},
            'num_samples': v['num_samples'],
            'avg_seq_len': v['avg_seq_len'],
            'std_seq_len': v['std_seq_len'],
            'max_seq_len': v['max_seq_len'],
        }
    
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': config['dataset_name'],
            'experiment_date': datetime.now().isoformat(),
            'checkpoint': config['checkpoint_path'],
            'results': results_json
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sequence Length Days Experiment for Next Location Prediction"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['diy', 'geolife', 'all'],
        default='all',
        help='Dataset to evaluate (diy, geolife, or all)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/next_loc_clean_v2/scripts/experiment_sequence_len_days/results',
        help='Output directory for results'
    )
    args = parser.parse_args()
    
    # Set seed
    set_seed(SEED)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    all_results = {}
    
    if args.dataset == 'all':
        datasets = ['diy', 'geolife']
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        results = run_sequence_length_experiment(
            dataset,
            args.output_dir,
            args.batch_size
        )
        all_results[dataset] = results
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

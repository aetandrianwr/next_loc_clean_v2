#!/usr/bin/env python
"""
Day-of-Week Analysis Experiment for Next Location Prediction.

This script evaluates model performance across different days of the week
to analyze temporal patterns in human mobility prediction. The hypothesis
is that weekend performance drops due to less routine behavior.

Scientific Rationale:
- Human mobility shows strong weekly periodicity
- Weekdays exhibit routine patterns (commute, work, lunch)
- Weekends show more exploratory behavior
- Prediction difficulty varies with behavioral regularity

Experiment Design:
- Filter test data by target prediction day (Monday-Sunday)
- Evaluate pre-trained models on each day subset
- Compute comprehensive metrics for each day
- Statistical analysis of weekday vs weekend performance

Output:
- Detailed metrics for each day of week
- Aggregated weekday vs weekend comparison
- Statistical significance tests
- Publication-quality visualizations

Usage:
    python run_days_analysis.py --dataset diy
    python run_days_analysis.py --dataset geolife
    python run_days_analysis.py --dataset both

Author: Experiment Script for PhD Thesis
"""

import os
import sys
import json
import pickle
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from tqdm import tqdm
from scipy import stats

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.proposed.pgt import PointerGeneratorTransformer
from src.evaluation.metrics import (
    calculate_correct_total_prediction,
    get_performance_dict,
)

# Configuration
SEED = 42
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
WEEKDAY_INDICES = [0, 1, 2, 3, 4]  # Monday to Friday
WEEKEND_INDICES = [5, 6]  # Saturday, Sunday

# Dataset configurations
CONFIGS = {
    'diy': {
        'name': 'DIY',
        'test_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_test.pk',
        'train_path': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_train.pk',
        'checkpoint': '/data/next_loc_clean_v2/experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt',
        'config': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pgt_diy_trial09.yaml',
        'model_config': {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.2,
        }
    },
    'geolife': {
        'name': 'Geolife',
        'test_path': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_test.pk',
        'train_path': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_train.pk',
        'checkpoint': '/data/next_loc_clean_v2/experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt',
        'config': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pgt_geolife_trial01.yaml',
        'model_config': {
            'd_model': 96,
            'nhead': 2,
            'num_layers': 2,
            'dim_feedforward': 192,
            'dropout': 0.25,
        }
    }
}

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_y_weekday(sample):
    """
    Compute the weekday of the target Y from sample data.
    
    The target Y's weekday is derived from the last element of weekday_X
    plus the diff value (days offset).
    
    Args:
        sample: Dictionary containing 'weekday_X' and 'diff' arrays
        
    Returns:
        int: Weekday of Y (0=Monday, 6=Sunday)
    """
    last_weekday = sample['weekday_X'][-1]
    last_diff = sample['diff'][-1]
    return (last_weekday + last_diff) % 7


class DayFilteredDataset(Dataset):
    """
    Dataset filtered by target day of week.
    
    This dataset wraps the test data and filters samples
    based on the day of week of the target prediction.
    """
    
    def __init__(self, data_path: str, day_filter: int = None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to pickle file
            day_filter: If set, filter to only include samples where Y falls on this day
                       (0=Monday, 1=Tuesday, ..., 6=Sunday). If None, include all.
        """
        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)
        
        # Compute Y weekday for each sample
        self.y_weekdays = [compute_y_weekday(sample) for sample in self.all_data]
        
        # Filter data if day_filter is specified
        if day_filter is not None:
            self.data = [
                sample for sample, wd in zip(self.all_data, self.y_weekdays)
                if wd == day_filter
            ]
            self.filtered_y_weekdays = [day_filter] * len(self.data)
        else:
            self.data = self.all_data
            self.filtered_y_weekdays = self.y_weekdays
        
        self._compute_statistics()
    
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
        
        self.num_locations = max(all_locs) + 1 if all_locs else 0
        self.num_users = max(all_users) + 1 if all_users else 0
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
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
    
    x_batch = pad_sequence(x_batch, batch_first=False, padding_value=0)
    y_batch = torch.stack(y_batch)
    x_dict_batch['user'] = torch.stack(x_dict_batch['user'])
    x_dict_batch['len'] = torch.tensor(x_dict_batch['len'], dtype=torch.long)
    
    for key in ['weekday', 'time', 'duration', 'diff']:
        x_dict_batch[key] = pad_sequence(x_dict_batch[key], batch_first=False, padding_value=0)
    
    return x_batch, y_batch, x_dict_batch


def load_model(dataset_key: str, device: torch.device):
    """
    Load pre-trained model from checkpoint.
    
    Args:
        dataset_key: 'diy' or 'geolife'
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    config = CONFIGS[dataset_key]
    
    # Load checkpoint first to get the correct max_seq_len
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    
    # Get max_seq_len from position_bias shape in checkpoint
    max_seq_len = checkpoint['model_state_dict']['position_bias'].shape[0]
    
    # Get number of locations and users from training data
    with open(config['train_path'], 'rb') as f:
        train_data = pickle.load(f)
    
    all_locs = set()
    all_users = set()
    for sample in train_data:
        all_locs.update(sample['X'].tolist())
        all_locs.add(sample['Y'])
        all_users.add(sample['user_X'][0])
    
    num_locations = max(all_locs) + 1
    num_users = max(all_users) + 1
    
    # Create model with correct max_seq_len
    model_cfg = config['model_config']
    model = PointerGeneratorTransformer(
        num_locations=num_locations,
        num_users=num_users,
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_layers'],
        dim_feedforward=model_cfg['dim_feedforward'],
        dropout=model_cfg['dropout'],
        max_seq_len=max_seq_len,
    )
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {config['checkpoint']}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Locations: {num_locations}, Users: {num_users}")
    print(f"  Max seq len: {max_seq_len}")
    
    return model


@torch.no_grad()
def evaluate_on_day(model, dataset: DayFilteredDataset, device: torch.device, batch_size: int = 64):
    """
    Evaluate model on a specific day's data.
    
    Args:
        model: Pre-trained model
        dataset: Day-filtered dataset
        device: Evaluation device
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of metrics
    """
    if len(dataset) == 0:
        return None
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    model.eval()
    all_results = []
    all_true_y = []
    all_pred_y = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
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
    perf['loss'] = total_loss / num_batches if num_batches > 0 else 0
    
    return perf


def run_day_analysis(dataset_key: str, output_dir: str):
    """
    Run day-of-week analysis for a dataset.
    
    Args:
        dataset_key: 'diy' or 'geolife'
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results for each day
    """
    config = CONFIGS[dataset_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print(f"DAY-OF-WEEK ANALYSIS: {config['name']} Dataset")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Test data: {config['test_path']}")
    print()
    
    # Load model
    model = load_model(dataset_key, device)
    
    # Analyze weekday distribution in test set
    with open(config['test_path'], 'rb') as f:
        test_data = pickle.load(f)
    
    y_weekdays = [compute_y_weekday(sample) for sample in test_data]
    print("\nTest Set Distribution by Target Day:")
    for day_idx in range(7):
        count = y_weekdays.count(day_idx)
        print(f"  {DAY_NAMES[day_idx]:12s}: {count:6d} samples ({count/len(y_weekdays)*100:.1f}%)")
    print()
    
    # Evaluate each day
    results = {}
    
    for day_idx in range(7):
        day_name = DAY_NAMES[day_idx]
        print(f"\n[{day_idx+1}/7] Evaluating {day_name}...")
        
        dataset = DayFilteredDataset(config['test_path'], day_filter=day_idx)
        
        if len(dataset) == 0:
            print(f"  No samples for {day_name}")
            continue
        
        metrics = evaluate_on_day(model, dataset, device)
        results[day_name] = {
            'day_index': day_idx,
            'samples': len(dataset),
            'is_weekend': day_idx >= 5,
            **{k: float(v) if isinstance(v, (np.floating, np.integer, float)) else v 
               for k, v in metrics.items()}
        }
        
        print(f"  Samples: {len(dataset)}")
        print(f"  Acc@1: {metrics['acc@1']:.2f}%, Acc@5: {metrics['acc@5']:.2f}%, MRR: {metrics['mrr']:.2f}%")
    
    # Add overall test set evaluation
    print(f"\n[Overall] Evaluating full test set...")
    full_dataset = DayFilteredDataset(config['test_path'], day_filter=None)
    full_metrics = evaluate_on_day(model, full_dataset, device)
    results['Overall'] = {
        'day_index': -1,
        'samples': len(full_dataset),
        'is_weekend': None,
        **{k: float(v) if isinstance(v, (np.floating, np.integer, float)) else v 
           for k, v in full_metrics.items()}
    }
    print(f"  Samples: {len(full_dataset)}")
    print(f"  Acc@1: {full_metrics['acc@1']:.2f}%, Acc@5: {full_metrics['acc@5']:.2f}%, MRR: {full_metrics['mrr']:.2f}%")
    
    # Compute weekday vs weekend aggregates
    weekday_metrics = []
    weekend_metrics = []
    
    for day_name, data in results.items():
        if data['is_weekend'] is None:
            continue
        if data['is_weekend']:
            weekend_metrics.append(data)
        else:
            weekday_metrics.append(data)
    
    # Weighted average by samples
    def weighted_avg(metrics_list, key):
        total_samples = sum(m['samples'] for m in metrics_list)
        return sum(m[key] * m['samples'] for m in metrics_list) / total_samples
    
    results['Weekday_Avg'] = {
        'day_index': -2,
        'samples': sum(m['samples'] for m in weekday_metrics),
        'is_weekend': False,
        'acc@1': weighted_avg(weekday_metrics, 'acc@1'),
        'acc@5': weighted_avg(weekday_metrics, 'acc@5'),
        'acc@10': weighted_avg(weekday_metrics, 'acc@10'),
        'mrr': weighted_avg(weekday_metrics, 'mrr'),
        'ndcg': weighted_avg(weekday_metrics, 'ndcg'),
        'f1': weighted_avg(weekday_metrics, 'f1'),
        'loss': weighted_avg(weekday_metrics, 'loss'),
    }
    
    results['Weekend_Avg'] = {
        'day_index': -3,
        'samples': sum(m['samples'] for m in weekend_metrics),
        'is_weekend': True,
        'acc@1': weighted_avg(weekend_metrics, 'acc@1'),
        'acc@5': weighted_avg(weekend_metrics, 'acc@5'),
        'acc@10': weighted_avg(weekend_metrics, 'acc@10'),
        'mrr': weighted_avg(weekend_metrics, 'mrr'),
        'ndcg': weighted_avg(weekend_metrics, 'ndcg'),
        'f1': weighted_avg(weekend_metrics, 'f1'),
        'loss': weighted_avg(weekend_metrics, 'loss'),
    }
    
    # Statistical tests (t-test for weekday vs weekend)
    weekday_acc1 = [results[DAY_NAMES[i]]['acc@1'] for i in WEEKDAY_INDICES]
    weekend_acc1 = [results[DAY_NAMES[i]]['acc@1'] for i in WEEKEND_INDICES]
    
    if len(weekday_acc1) > 1 and len(weekend_acc1) > 1:
        t_stat, p_value = stats.ttest_ind(weekday_acc1, weekend_acc1)
        results['Statistical_Test'] = {
            'test': 'Independent t-test',
            'comparison': 'Weekday vs Weekend Acc@1',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_005': bool(p_value < 0.05),
            'significant_at_001': bool(p_value < 0.01),
            'weekday_mean': float(np.mean(weekday_acc1)),
            'weekend_mean': float(np.mean(weekend_acc1)),
            'difference': float(np.mean(weekday_acc1) - np.mean(weekend_acc1)),
        }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f'{dataset_key}_days_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


def create_results_table(results: dict, dataset_name: str):
    """
    Create a formatted results table.
    
    Args:
        results: Dictionary of results
        dataset_name: Name of dataset
        
    Returns:
        pandas DataFrame
    """
    rows = []
    
    # Individual days
    for day_name in DAY_NAMES:
        if day_name in results:
            r = results[day_name]
            rows.append({
                'Day': day_name,
                'Type': 'Weekend' if r['is_weekend'] else 'Weekday',
                'Samples': r['samples'],
                'Acc@1': r['acc@1'],
                'Acc@5': r['acc@5'],
                'Acc@10': r['acc@10'],
                'MRR': r['mrr'],
                'NDCG': r['ndcg'],
                'F1': r['f1'],
                'Loss': r['loss'],
            })
    
    # Aggregates
    for key in ['Weekday_Avg', 'Weekend_Avg', 'Overall']:
        if key in results:
            r = results[key]
            type_str = 'Weekend' if r.get('is_weekend') else ('Weekday' if r.get('is_weekend') is False else 'All')
            rows.append({
                'Day': key.replace('_', ' '),
                'Type': type_str,
                'Samples': r['samples'],
                'Acc@1': r.get('acc@1', 0),
                'Acc@5': r.get('acc@5', 0),
                'Acc@10': r.get('acc@10', 0),
                'MRR': r.get('mrr', 0),
                'NDCG': r.get('ndcg', 0),
                'F1': r.get('f1', 0),
                'Loss': r.get('loss', 0),
            })
    
    df = pd.DataFrame(rows)
    return df


def print_summary(diy_results: dict = None, geolife_results: dict = None):
    """Print experiment summary."""
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY: Day-of-Week Impact on Next Location Prediction")
    print("=" * 100)
    
    for name, results in [('DIY', diy_results), ('Geolife', geolife_results)]:
        if results is None:
            continue
        
        print(f"\n{name} Dataset Results:")
        print("-" * 80)
        
        df = create_results_table(results, name)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        
        if 'Statistical_Test' in results:
            st = results['Statistical_Test']
            print(f"\nStatistical Analysis ({st['test']}):")
            print(f"  Weekday Mean Acc@1: {st['weekday_mean']:.2f}%")
            print(f"  Weekend Mean Acc@1: {st['weekend_mean']:.2f}%")
            print(f"  Difference: {st['difference']:.2f}% (weekday - weekend)")
            print(f"  t-statistic: {st['t_statistic']:.4f}")
            print(f"  p-value: {st['p_value']:.4f}")
            if st['significant_at_001']:
                print(f"  ** Highly significant (p < 0.01) **")
            elif st['significant_at_005']:
                print(f"  * Significant (p < 0.05) *")
            else:
                print(f"  Not statistically significant")


def main():
    parser = argparse.ArgumentParser(description="Day-of-Week Analysis Experiment")
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['diy', 'geolife', 'both'],
        default='both',
        help='Dataset to analyze'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/next_loc_clean_v2/scripts/experiment_days_analysis_v2/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Run analysis
    diy_results = None
    geolife_results = None
    
    if args.dataset in ['diy', 'both']:
        diy_results = run_day_analysis('diy', args.output_dir)
    
    if args.dataset in ['geolife', 'both']:
        geolife_results = run_day_analysis('geolife', args.output_dir)
    
    # Print summary
    print_summary(diy_results, geolife_results)
    
    print("\n" + "=" * 80)
    print("Day-of-Week Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

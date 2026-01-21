"""
Comprehensive Inference and Evaluation Script for Pointer Generator Transformer.

This script provides Nature Journal standard experiment evaluation for the 
PointerGeneratorTransformer model for next location prediction task. It includes:

1. Model loading from checkpoints
2. Comprehensive test set evaluation with multiple metrics
3. Detailed statistical analysis
4. Per-user and per-sequence analysis
5. Sample selection for demonstration (positive/negative predictions)
6. Results export in tabular and JSON formats

Author: Research Team
Date: 2026-01-02
Seed: 42
"""

import os
import sys
import json
import yaml
import pickle
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)
from scipy import stats
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.proposed.pgt import PointerGeneratorTransformer
from src.evaluation.metrics import (
    calculate_correct_total_prediction,
    get_performance_dict,
)


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment configurations
EXPERIMENTS = {
    'diy': {
        'checkpoint_dir': '/data/next_loc_clean_v2/experiments/diy_pointer_v45_20260101_155348',
        'config_path': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pgt_diy_trial09.yaml',
        'data_dir': '/data/next_loc_clean_v2/data/diy_eps50/processed',
        'dataset_prefix': 'diy_eps50_prev7',
        'locations_path': '/data/next_loc_clean_v2/data/diy_eps50/interim/locations_eps50.csv',
        'dataset_name': 'DIY (Helsinki City Bikes)',
    },
    'geolife': {
        'checkpoint_dir': '/data/next_loc_clean_v2/experiments/geolife_pointer_v45_20260101_151038',
        'config_path': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pgt_geolife_trial01.yaml',
        'data_dir': '/data/next_loc_clean_v2/data/geolife_eps20/processed',
        'dataset_prefix': 'geolife_eps20_prev7',
        'locations_path': '/data/next_loc_clean_v2/data/geolife_eps20/interim/locations_eps20.csv',
        'dataset_name': 'GeoLife (Microsoft Research Asia)',
    }
}


# =============================================================================
# Utility Functions
# =============================================================================

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


# =============================================================================
# Dataset and DataLoader
# =============================================================================

class NextLocationDataset(Dataset):
    """Dataset for next location prediction with raw sample access."""
    
    def __init__(self, data_path: str, build_user_history: bool = True):
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
        """Build user location history."""
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
        """Get a single sample with full metadata."""
        sample = self.data[idx]
        
        return_dict = {
            'user': torch.tensor(sample['user_X'][0], dtype=torch.long),
            'weekday': torch.tensor(sample['weekday_X'], dtype=torch.long),
            'time': torch.tensor(sample['start_min_X'] // 15, dtype=torch.long),
            'duration': torch.tensor(sample['dur_X'] // 30, dtype=torch.long),
            'diff': torch.tensor(sample['diff'], dtype=torch.long),
            'idx': idx,  # Keep track of original index
        }
        
        x = torch.tensor(sample['X'], dtype=torch.long)
        y = torch.tensor(sample['Y'], dtype=torch.long)
        
        return x, y, return_dict
    
    def get_raw_sample(self, idx: int) -> Dict:
        """Get raw sample data without tensor conversion."""
        return self.data[idx]


def collate_fn(batch):
    """Collate function to handle variable length sequences."""
    x_batch, y_batch = [], []
    
    x_dict_batch = {'len': [], 'idx': []}
    for key in ['user', 'weekday', 'time', 'duration', 'diff']:
        x_dict_batch[key] = []
    
    for x, y, return_dict in batch:
        x_batch.append(x)
        y_batch.append(y)
        
        x_dict_batch['len'].append(len(x))
        x_dict_batch['idx'].append(return_dict['idx'])
        for key in ['user', 'weekday', 'time', 'duration', 'diff']:
            x_dict_batch[key].append(return_dict[key])
    
    x_batch = pad_sequence(x_batch, batch_first=False, padding_value=0)
    y_batch = torch.stack(y_batch)
    
    x_dict_batch['user'] = torch.stack(x_dict_batch['user'])
    x_dict_batch['len'] = torch.tensor(x_dict_batch['len'], dtype=torch.long)
    x_dict_batch['idx'] = torch.tensor(x_dict_batch['idx'], dtype=torch.long)
    
    for key in ['weekday', 'time', 'duration', 'diff']:
        x_dict_batch[key] = pad_sequence(x_dict_batch[key], batch_first=False, padding_value=0)
    
    return x_batch, y_batch, x_dict_batch


def get_test_dataloader(config: Dict) -> Tuple[DataLoader, Dataset, Dict]:
    """Create test dataloader."""
    data_dir = config['data']['data_dir']
    dataset_prefix = config['data']['dataset_prefix']
    batch_size = config['training'].get('batch_size', 64)
    
    # Load train dataset for user history
    train_path = os.path.join(data_dir, f"{dataset_prefix}_train.pk")
    train_ds = NextLocationDataset(train_path, build_user_history=True)
    
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")
    test_ds = NextLocationDataset(test_path, build_user_history=False)
    
    # Share user history
    test_ds.user_location_history = train_ds.user_location_history
    test_ds.user_location_freq = train_ds.user_location_freq
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )
    
    info = {
        'num_locations': train_ds.num_locations,
        'num_users': train_ds.num_users,
        'max_seq_len': max(train_ds.max_seq_len, test_ds.max_seq_len),
        'train_size': len(train_ds),
        'test_size': len(test_ds),
    }
    
    return test_loader, test_ds, info


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_dir: str, config: Dict, info: Dict, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints', 'best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use config from checkpoint for consistency
    ckpt_config = checkpoint.get('config', {})
    model_cfg = ckpt_config.get('model', config['model'])
    
    # Determine max_seq_len from checkpoint
    position_bias_shape = checkpoint['model_state_dict']['position_bias'].shape[0]
    
    model = PointerGeneratorTransformer(
        num_locations=info['num_locations'],
        num_users=info['num_users'],
        d_model=model_cfg.get('d_model', 128),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 3),
        dim_feedforward=model_cfg.get('dim_feedforward', 256),
        dropout=model_cfg.get('dropout', 0.15),
        max_seq_len=position_bias_shape,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    return model


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

class DetailedEvaluator:
    """Comprehensive evaluation with detailed per-sample analysis."""
    
    def __init__(self, model: nn.Module, test_loader: DataLoader, 
                 test_dataset: Dataset, device: torch.device):
        self.model = model
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.device = device
        
        # Results storage
        self.sample_results = []
        self.metrics = {}
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run comprehensive evaluation."""
        self.model.eval()
        
        all_results = []
        all_true_y = []
        all_pred_y = []
        all_probs = []
        all_indices = []
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        print("Running comprehensive evaluation...")
        for x, y, x_dict in tqdm(self.test_loader, desc="Evaluating"):
            x = x.to(self.device)
            y = y.to(self.device)
            batch_indices = x_dict['idx'].numpy()
            x_dict_gpu = {k: v.to(self.device) for k, v in x_dict.items() if k != 'idx'}
            
            # Forward pass
            logits = self.model(x, x_dict_gpu)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions and probabilities
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(10, probs.shape[-1]), dim=-1)
            
            results, true_y, pred_y = calculate_correct_total_prediction(logits, y)
            all_results.append(results)
            all_true_y.append(true_y)
            all_pred_y.append(pred_y)
            
            # Store per-sample results
            for i in range(len(y)):
                sample_idx = batch_indices[i]
                true_label = y[i].item()
                pred_label = top_indices[i, 0].item()
                pred_prob = top_probs[i, 0].item()
                
                # Check if prediction is in top-k
                top_k_preds = top_indices[i].cpu().numpy().tolist()
                is_correct_1 = true_label == pred_label
                is_correct_5 = true_label in top_k_preds[:5]
                is_correct_10 = true_label in top_k_preds[:10]
                
                # Get rank of true label
                rank = -1
                for r, idx in enumerate(top_k_preds):
                    if idx == true_label:
                        rank = r + 1
                        break
                if rank == -1:
                    # Search in full distribution
                    sorted_indices = torch.argsort(probs[i], descending=True).cpu().numpy()
                    for r, idx in enumerate(sorted_indices):
                        if idx == true_label:
                            rank = r + 1
                            break
                
                self.sample_results.append({
                    'sample_idx': sample_idx,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'pred_prob': pred_prob,
                    'top_5_preds': top_k_preds[:5],
                    'top_5_probs': top_probs[i, :5].cpu().numpy().tolist(),
                    'is_correct_1': is_correct_1,
                    'is_correct_5': is_correct_5,
                    'is_correct_10': is_correct_10,
                    'rank': rank,
                    'user': x_dict['user'][i].item(),
                    'seq_len': x_dict['len'][i].item(),
                })
            
            all_indices.extend(batch_indices.tolist())
        
        # Aggregate metrics
        total_results = np.sum(np.stack(all_results), axis=0)
        self.metrics = {
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
        
        self.metrics['f1'] = f1_score(all_true_y.tolist(), all_pred_y_flat, average='weighted', zero_division=0)
        self.metrics['precision'] = precision_score(all_true_y.tolist(), all_pred_y_flat, average='weighted', zero_division=0)
        self.metrics['recall'] = recall_score(all_true_y.tolist(), all_pred_y_flat, average='weighted', zero_division=0)
        
        perf = get_performance_dict(self.metrics)
        perf['loss'] = total_loss / num_batches
        
        # Add confidence interval calculations
        perf = self._add_confidence_intervals(perf)
        
        return perf
    
    def _add_confidence_intervals(self, perf: Dict) -> Dict:
        """Add 95% confidence intervals using bootstrap."""
        n_samples = len(self.sample_results)
        n_bootstrap = 1000
        
        correct_1 = np.array([r['is_correct_1'] for r in self.sample_results])
        correct_5 = np.array([r['is_correct_5'] for r in self.sample_results])
        correct_10 = np.array([r['is_correct_10'] for r in self.sample_results])
        ranks = np.array([r['rank'] for r in self.sample_results])
        
        # Bootstrap for confidence intervals
        acc1_samples = []
        acc5_samples = []
        acc10_samples = []
        mrr_samples = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            acc1_samples.append(correct_1[indices].mean() * 100)
            acc5_samples.append(correct_5[indices].mean() * 100)
            acc10_samples.append(correct_10[indices].mean() * 100)
            mrr_samples.append(np.mean(1.0 / ranks[indices]) * 100)
        
        perf['acc@1_ci_lower'] = np.percentile(acc1_samples, 2.5)
        perf['acc@1_ci_upper'] = np.percentile(acc1_samples, 97.5)
        perf['acc@5_ci_lower'] = np.percentile(acc5_samples, 2.5)
        perf['acc@5_ci_upper'] = np.percentile(acc5_samples, 97.5)
        perf['acc@10_ci_lower'] = np.percentile(acc10_samples, 2.5)
        perf['acc@10_ci_upper'] = np.percentile(acc10_samples, 97.5)
        perf['mrr_ci_lower'] = np.percentile(mrr_samples, 2.5)
        perf['mrr_ci_upper'] = np.percentile(mrr_samples, 97.5)
        
        return perf
    
    def get_samples_by_prediction(self, correct: bool = True, 
                                   min_confidence: float = 0.0,
                                   max_confidence: float = 1.0,
                                   n_samples: int = 10) -> List[Dict]:
        """Get samples by prediction correctness and confidence."""
        filtered = [
            r for r in self.sample_results
            if r['is_correct_1'] == correct
            and min_confidence <= r['pred_prob'] <= max_confidence
        ]
        
        # Sort by confidence (descending for correct, ascending for incorrect)
        if correct:
            filtered.sort(key=lambda x: x['pred_prob'], reverse=True)
        else:
            # For incorrect predictions, get those with high confidence (harder cases)
            filtered.sort(key=lambda x: x['pred_prob'], reverse=True)
        
        return filtered[:n_samples]
    
    def get_user_statistics(self) -> pd.DataFrame:
        """Get per-user statistics."""
        user_stats = defaultdict(lambda: {'correct_1': 0, 'correct_5': 0, 'total': 0, 'ranks': []})
        
        for r in self.sample_results:
            user = r['user']
            user_stats[user]['correct_1'] += int(r['is_correct_1'])
            user_stats[user]['correct_5'] += int(r['is_correct_5'])
            user_stats[user]['total'] += 1
            user_stats[user]['ranks'].append(r['rank'])
        
        rows = []
        for user, stats in user_stats.items():
            rows.append({
                'user_id': user,
                'n_samples': stats['total'],
                'acc@1': stats['correct_1'] / stats['total'] * 100,
                'acc@5': stats['correct_5'] / stats['total'] * 100,
                'mrr': np.mean([1/r for r in stats['ranks']]) * 100,
            })
        
        return pd.DataFrame(rows)
    
    def get_sequence_length_analysis(self) -> pd.DataFrame:
        """Analyze performance by sequence length."""
        len_stats = defaultdict(lambda: {'correct_1': 0, 'correct_5': 0, 'total': 0, 'ranks': []})
        
        for r in self.sample_results:
            seq_len = r['seq_len']
            len_stats[seq_len]['correct_1'] += int(r['is_correct_1'])
            len_stats[seq_len]['correct_5'] += int(r['is_correct_5'])
            len_stats[seq_len]['total'] += 1
            len_stats[seq_len]['ranks'].append(r['rank'])
        
        rows = []
        for seq_len, stats in sorted(len_stats.items()):
            rows.append({
                'seq_len': seq_len,
                'n_samples': stats['total'],
                'acc@1': stats['correct_1'] / stats['total'] * 100,
                'acc@5': stats['correct_5'] / stats['total'] * 100,
                'mrr': np.mean([1/r for r in stats['ranks']]) * 100,
            })
        
        return pd.DataFrame(rows)


# =============================================================================
# Sample Preparation for Demonstration
# =============================================================================

def prepare_demo_samples(evaluator: DetailedEvaluator, test_dataset: Dataset,
                         locations_df: pd.DataFrame, n_samples: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """Prepare demonstration samples with full context."""
    from shapely import wkt
    
    # Create location lookup
    loc_coords = {}
    for _, row in locations_df.iterrows():
        try:
            pt = wkt.loads(row['center'])
            loc_coords[row['id']] = {'lng': pt.x, 'lat': pt.y}
        except:
            pass
    
    # Get positive samples (correct predictions with high confidence)
    positive_results = evaluator.get_samples_by_prediction(correct=True, n_samples=n_samples * 2)
    
    # Get negative samples (incorrect predictions with reasonable variety)
    negative_results = evaluator.get_samples_by_prediction(correct=False, n_samples=n_samples * 2)
    
    def enrich_sample(result: Dict) -> Dict:
        """Add full context to sample."""
        raw = test_dataset.get_raw_sample(result['sample_idx'])
        
        # Get coordinates for sequence
        seq_coords = []
        for loc_id in raw['X']:
            if loc_id in loc_coords:
                seq_coords.append({
                    'loc_id': int(loc_id),
                    'lat': loc_coords[loc_id]['lat'],
                    'lng': loc_coords[loc_id]['lng'],
                })
            else:
                seq_coords.append({'loc_id': int(loc_id), 'lat': None, 'lng': None})
        
        # Get target coordinates
        target_coords = loc_coords.get(raw['Y'], {'lat': None, 'lng': None})
        
        # Get prediction coordinates
        pred_coords = loc_coords.get(result['pred_label'], {'lat': None, 'lng': None})
        
        return {
            'sample_idx': int(result['sample_idx']),
            'user_id': int(raw['user_X'][0]),
            'sequence': [int(x) for x in raw['X'].tolist()],
            'sequence_coords': seq_coords,
            'target_location': int(raw['Y']),
            'target_coords': target_coords,
            'predicted_location': int(result['pred_label']),
            'predicted_coords': pred_coords,
            'prediction_confidence': float(result['pred_prob']),
            'top_5_predictions': [int(x) for x in result['top_5_preds']],
            'top_5_confidences': [float(x) for x in result['top_5_probs']],
            'is_correct': bool(result['is_correct_1']),
            'rank': int(result['rank']),
            'weekdays': [int(x) for x in raw['weekday_X'].tolist()],
            'times': [int(x) for x in (raw['start_min_X'] // 15).tolist()],  # 15-min intervals
            'durations': [int(x) for x in (raw['dur_X'] // 30).tolist()],  # 30-min buckets
            'recency': [int(x) for x in raw['diff'].tolist()],
        }
    
    # Filter samples that have valid coordinates
    positive_samples = []
    for r in positive_results:
        enriched = enrich_sample(r)
        if enriched['target_coords']['lat'] is not None:
            positive_samples.append(enriched)
            if len(positive_samples) >= n_samples:
                break
    
    negative_samples = []
    for r in negative_results:
        enriched = enrich_sample(r)
        if enriched['target_coords']['lat'] is not None:
            negative_samples.append(enriched)
            if len(negative_samples) >= n_samples:
                break
    
    return positive_samples, negative_samples


# =============================================================================
# Results Export
# =============================================================================

def export_results(output_dir: str, dataset_name: str, metrics: Dict, 
                   evaluator: DetailedEvaluator, positive_samples: List[Dict],
                   negative_samples: List[Dict]):
    """Export all results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Main metrics JSON
    metrics_path = os.path.join(output_dir, f'{dataset_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in metrics.items()}, f, indent=2)
    
    # 2. Sample results CSV
    samples_df = pd.DataFrame(evaluator.sample_results)
    samples_df.to_csv(os.path.join(output_dir, f'{dataset_name}_sample_results.csv'), index=False)
    
    # 3. User statistics CSV
    user_stats = evaluator.get_user_statistics()
    user_stats.to_csv(os.path.join(output_dir, f'{dataset_name}_user_statistics.csv'), index=False)
    
    # 4. Sequence length analysis CSV
    seq_analysis = evaluator.get_sequence_length_analysis()
    seq_analysis.to_csv(os.path.join(output_dir, f'{dataset_name}_sequence_analysis.csv'), index=False)
    
    # 5. Demo samples JSON
    demo_path = os.path.join(output_dir, f'{dataset_name}_demo_samples.json')
    with open(demo_path, 'w') as f:
        json.dump({
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
        }, f, indent=2)
    
    print(f"Results exported to {output_dir}")


def create_summary_table(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """Create summary table for all datasets."""
    rows = []
    for dataset_name, metrics in all_metrics.items():
        rows.append({
            'Dataset': dataset_name.upper(),
            'Acc@1 (%)': f"{metrics['acc@1']:.2f} ± {(metrics['acc@1_ci_upper'] - metrics['acc@1_ci_lower'])/4:.2f}",
            'Acc@5 (%)': f"{metrics['acc@5']:.2f} ± {(metrics['acc@5_ci_upper'] - metrics['acc@5_ci_lower'])/4:.2f}",
            'Acc@10 (%)': f"{metrics['acc@10']:.2f} ± {(metrics['acc@10_ci_upper'] - metrics['acc@10_ci_lower'])/4:.2f}",
            'MRR (%)': f"{metrics['mrr']:.2f} ± {(metrics['mrr_ci_upper'] - metrics['mrr_ci_lower'])/4:.2f}",
            'NDCG@10 (%)': f"{metrics['ndcg']:.2f}",
            'F1 Score': f"{metrics['f1']*100:.2f}",
            'Loss': f"{metrics['loss']:.4f}",
            'N': int(metrics['total']),
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# Main Execution
# =============================================================================

def run_experiment(dataset_key: str, output_dir: str) -> Dict:
    """Run experiment for a single dataset."""
    exp_config = EXPERIMENTS[dataset_key]
    
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_config['dataset_name']}")
    print(f"{'='*60}")
    
    # Load config
    config = load_config(exp_config['config_path'])
    config['data']['data_dir'] = exp_config['data_dir']
    config['data']['dataset_prefix'] = exp_config['dataset_prefix']
    
    # Get test dataloader
    print("Loading test data...")
    test_loader, test_dataset, info = get_test_dataloader(config)
    print(f"Test samples: {info['test_size']}")
    print(f"Locations: {info['num_locations']}, Users: {info['num_users']}")
    
    # Load model
    print("Loading model...")
    model = load_model(exp_config['checkpoint_dir'], config, info, DEVICE)
    
    # Run evaluation
    evaluator = DetailedEvaluator(model, test_loader, test_dataset, DEVICE)
    metrics = evaluator.evaluate()
    
    # Print metrics
    print(f"\n{'='*40}")
    print(f"Results for {exp_config['dataset_name']}")
    print(f"{'='*40}")
    print(f"Acc@1:  {metrics['acc@1']:.2f}% ({metrics['acc@1_ci_lower']:.2f} - {metrics['acc@1_ci_upper']:.2f})")
    print(f"Acc@5:  {metrics['acc@5']:.2f}% ({metrics['acc@5_ci_lower']:.2f} - {metrics['acc@5_ci_upper']:.2f})")
    print(f"Acc@10: {metrics['acc@10']:.2f}% ({metrics['acc@10_ci_lower']:.2f} - {metrics['acc@10_ci_upper']:.2f})")
    print(f"MRR:    {metrics['mrr']:.2f}% ({metrics['mrr_ci_lower']:.2f} - {metrics['mrr_ci_upper']:.2f})")
    print(f"NDCG:   {metrics['ndcg']:.2f}%")
    print(f"F1:     {metrics['f1']*100:.2f}%")
    print(f"Loss:   {metrics['loss']:.4f}")
    
    # Load locations for coordinate mapping
    locations_df = pd.read_csv(exp_config['locations_path'])
    
    # Prepare demo samples
    print("\nPreparing demonstration samples...")
    positive_samples, negative_samples = prepare_demo_samples(
        evaluator, test_dataset, locations_df, n_samples=10
    )
    print(f"Positive samples: {len(positive_samples)}")
    print(f"Negative samples: {len(negative_samples)}")
    
    # Export results
    export_results(output_dir, dataset_key, metrics, evaluator, 
                   positive_samples, negative_samples)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Comprehensive inference for Pointer Generator Transformer")
    parser.add_argument("--dataset", type=str, default="all", 
                        choices=['all', 'diy', 'geolife'],
                        help="Dataset to evaluate")
    parser.add_argument("--output_dir", type=str, 
                        default="/data/next_loc_clean_v2/scripts/inference/results",
                        help="Output directory for results")
    args = parser.parse_args()
    
    set_seed(SEED)
    
    print("="*60)
    print("Pointer Generator Transformer - Comprehensive Inference")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Output directory: {args.output_dir}")
    
    all_metrics = {}
    
    if args.dataset == 'all':
        datasets = ['diy', 'geolife']
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        metrics = run_experiment(dataset, args.output_dir)
        all_metrics[dataset] = metrics
    
    # Create summary table
    if len(all_metrics) > 1:
        summary_df = create_summary_table(all_metrics)
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        # Save summary table
        summary_path = os.path.join(args.output_dir, 'summary_table.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary table saved to {summary_path}")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)


if __name__ == "__main__":
    main()

"""
Experiment 1: Sequence Length Analysis
======================================
Analyzes model performance across different sequence lengths to understand
how trajectory history length impacts prediction accuracy.

This experiment evaluates:
- How prediction accuracy changes with sequence length
- Whether shorter or longer sequences are easier to predict
- The relationship between sequence length and various metrics
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.proposed.pgt import PointerGeneratorTransformer
from src.evaluation.metrics import calculate_correct_total_prediction, get_performance_dict
from sklearn.metrics import f1_score

# Configuration
SEED = 42
GEOLIFE_CHECKPOINT = "/data/next_loc_clean_v2/experiments/geolife_pointer_v45_20251229_023222/checkpoints/best.pt"
DIY_CHECKPOINT = "/data/next_loc_clean_v2/experiments/diy_pointer_v45_20251229_023930/checkpoints/best.pt"
GEOLIFE_DATA_DIR = "/data/next_loc_clean_v2/data/geolife_eps20/processed"
DIY_DATA_DIR = "/data/next_loc_clean_v2/data/diy_eps50/processed"
OUTPUT_DIR = Path(__file__).parent / "results"


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint_path, config, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = config['model']
    
    # Get data info from config
    data_dir = config['data']['data_dir']
    dataset_prefix = config['data']['dataset_prefix']
    metadata_path = os.path.join(data_dir, f"{dataset_prefix}_metadata.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Infer max_seq_len from checkpoint
    max_seq_len = checkpoint['model_state_dict']['position_bias'].shape[0]
    
    model = PointerGeneratorTransformer(
        num_locations=metadata['total_loc_num'],
        num_users=metadata['total_user_num'],
        d_model=model_cfg.get('d_model', 128),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 3),
        dim_feedforward=model_cfg.get('dim_feedforward', 256),
        dropout=model_cfg.get('dropout', 0.15),
        max_seq_len=max_seq_len,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, metadata


def load_test_data(data_dir, dataset_prefix):
    """Load test data from pickle file."""
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")
    with open(test_path, 'rb') as f:
        data = pickle.load(f)
    return data


def categorize_by_sequence_length(data):
    """Categorize samples by sequence length."""
    length_groups = defaultdict(list)
    
    for idx, sample in enumerate(data):
        seq_len = len(sample['X'])
        length_groups[seq_len].append(idx)
    
    return length_groups


def evaluate_group(model, data, indices, device):
    """Evaluate model on a subset of samples."""
    if len(indices) == 0:
        return None
    
    all_results = []
    all_true_y = []
    all_pred_y = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for idx in indices:
            sample = data[idx]
            
            # Prepare input
            x = torch.tensor(sample['X'], dtype=torch.long).unsqueeze(1).to(device)
            y = torch.tensor([sample['Y']], dtype=torch.long).to(device)
            
            x_dict = {
                'user': torch.tensor([sample['user_X'][0]], dtype=torch.long).to(device),
                'weekday': torch.tensor(sample['weekday_X'], dtype=torch.long).unsqueeze(1).to(device),
                'time': torch.tensor(sample['start_min_X'] // 15, dtype=torch.long).unsqueeze(1).to(device),
                'duration': torch.tensor(sample['dur_X'] // 30, dtype=torch.long).unsqueeze(1).to(device),
                'diff': torch.tensor(sample['diff'], dtype=torch.long).unsqueeze(1).to(device),
                'len': torch.tensor([len(sample['X'])], dtype=torch.long).to(device),
            }
            
            # Forward pass
            logits = model(x, x_dict)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            # Calculate metrics
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
    perf['loss'] = total_loss / len(indices)
    
    return perf


def run_experiment(dataset_name, checkpoint_path, data_dir, dataset_prefix, device):
    """Run sequence length analysis for a dataset."""
    print(f"\n{'='*60}")
    print(f"Running Sequence Length Analysis for {dataset_name}")
    print(f"{'='*60}")
    
    # Load config from checkpoint directory
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and data
    model, metadata = load_model(checkpoint_path, config, device)
    test_data = load_test_data(data_dir, dataset_prefix)
    
    print(f"Total test samples: {len(test_data)}")
    
    # Categorize by sequence length
    length_groups = categorize_by_sequence_length(test_data)
    
    # Get all sequence lengths and sort
    all_lengths = sorted(length_groups.keys())
    print(f"Sequence length range: {min(all_lengths)} - {max(all_lengths)}")
    
    # Create length bins
    bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 100), (101, max(all_lengths) + 1)]
    
    results = []
    
    for bin_start, bin_end in bins:
        indices = []
        for length in range(bin_start, bin_end):
            if length in length_groups:
                indices.extend(length_groups[length])
        
        if len(indices) == 0:
            continue
        
        print(f"\nEvaluating sequences with length {bin_start}-{bin_end-1} ({len(indices)} samples)...")
        perf = evaluate_group(model, test_data, indices, device)
        
        if perf:
            result = {
                'length_bin': f"{bin_start}-{bin_end-1}",
                'num_samples': len(indices),
                **perf
            }
            results.append(result)
            print(f"  Acc@1: {perf['acc@1']:.2f}%, MRR: {perf['mrr']:.2f}%")
    
    return results


def create_visualizations(geolife_results, diy_results, output_dir):
    """Create visualizations for the experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Prepare data
    geolife_df = pd.DataFrame(geolife_results)
    diy_df = pd.DataFrame(diy_results)
    
    # Figure 1: Acc@1 by sequence length
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Geolife
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(geolife_df)), geolife_df['acc@1'], color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(geolife_df)))
    ax1.set_xticklabels(geolife_df['length_bin'], rotation=45, ha='right')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Geolife: Accuracy@1 by Sequence Length')
    
    # Add value labels
    for bar, val in zip(bars1, geolife_df['acc@1']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # DIY
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(diy_df)), diy_df['acc@1'], color='darkorange', alpha=0.8)
    ax2.set_xticks(range(len(diy_df)))
    ax2.set_xticklabels(diy_df['length_bin'], rotation=45, ha='right')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Accuracy@1 (%)')
    ax2.set_title('DIY: Accuracy@1 by Sequence Length')
    
    for bar, val in zip(bars2, diy_df['acc@1']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'acc1_by_sequence_length.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Multiple metrics comparison (side by side separate bars)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['acc@1', 'acc@5', 'mrr', 'ndcg']
    titles = ['Accuracy@1', 'Accuracy@5', 'MRR', 'NDCG']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        # Plot separate bars for each dataset
        x_g = np.arange(len(geolife_df))
        x_d = np.arange(len(diy_df)) + len(geolife_df) + 1
        
        ax.bar(x_g, geolife_df[metric], label='Geolife', color='steelblue', alpha=0.8)
        ax.bar(x_d, diy_df[metric], label='DIY', color='darkorange', alpha=0.8)
        
        ax.set_xlabel('Sequence Length Bin')
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(f'{title} by Sequence Length')
        
        # Set x-ticks for both datasets
        all_ticks = list(x_g) + list(x_d)
        all_labels = list(geolife_df['length_bin']) + list(diy_df['length_bin'])
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_by_sequence_length.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Sample distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.bar(range(len(geolife_df)), geolife_df['num_samples'], color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(geolife_df)))
    ax1.set_xticklabels(geolife_df['length_bin'], rotation=45, ha='right')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Geolife: Sample Distribution by Sequence Length')
    
    ax2 = axes[1]
    ax2.bar(range(len(diy_df)), diy_df['num_samples'], color='darkorange', alpha=0.8)
    ax2.set_xticks(range(len(diy_df)))
    ax2.set_xticklabels(diy_df['length_bin'], rotation=45, ha='right')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('DIY: Sample Distribution by Sequence Length')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_distribution_by_sequence_length.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")


def save_results(geolife_results, diy_results, output_dir):
    """Save results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Save as JSON
    results = {
        'geolife': convert_to_serializable(geolife_results),
        'diy': convert_to_serializable(diy_results)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV
    geolife_df = pd.DataFrame(geolife_results)
    diy_df = pd.DataFrame(diy_results)
    
    geolife_df.to_csv(output_dir / 'geolife_results.csv', index=False)
    diy_df.to_csv(output_dir / 'diy_results.csv', index=False)
    
    # Create summary table
    summary = []
    for dataset, df in [('Geolife', geolife_df), ('DIY', diy_df)]:
        for _, row in df.iterrows():
            summary.append({
                'Dataset': dataset,
                'Sequence Length': row['length_bin'],
                'Samples': int(row['num_samples']),
                'Acc@1': f"{row['acc@1']:.2f}",
                'Acc@5': f"{row['acc@5']:.2f}",
                'Acc@10': f"{row['acc@10']:.2f}",
                'MRR': f"{row['mrr']:.2f}",
                'NDCG': f"{row['ndcg']:.2f}",
                'F1': f"{row['f1']*100:.2f}",
                'Loss': f"{row['loss']:.4f}",
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    
    # Print summary table
    print("\n" + "="*100)
    print("SEQUENCE LENGTH ANALYSIS - SUMMARY TABLE")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)


def main():
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run experiments
    geolife_results = run_experiment(
        'Geolife', 
        GEOLIFE_CHECKPOINT, 
        GEOLIFE_DATA_DIR, 
        'geolife_eps20_prev7', 
        device
    )
    
    diy_results = run_experiment(
        'DIY', 
        DIY_CHECKPOINT, 
        DIY_DATA_DIR, 
        'diy_eps50_prev7', 
        device
    )
    
    # Save results and create visualizations
    save_results(geolife_results, diy_results, OUTPUT_DIR)
    create_visualizations(geolife_results, diy_results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Experiment 1: Sequence Length Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

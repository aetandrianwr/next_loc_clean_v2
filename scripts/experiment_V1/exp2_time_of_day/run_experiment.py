"""
Experiment 2: Time-of-Day Analysis
==================================
Analyzes model performance across different time periods to understand
temporal patterns in mobility prediction.

Time periods:
- Early Morning: 00:00-06:00
- Morning: 06:00-12:00
- Afternoon: 12:00-18:00
- Evening/Night: 18:00-24:00
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

from src.models.proposed.pointer_v45 import PointerNetworkV45
from src.evaluation.metrics import calculate_correct_total_prediction, get_performance_dict
from sklearn.metrics import f1_score

# Configuration
SEED = 42
GEOLIFE_CHECKPOINT = "/data/next_loc_clean_v2/experiments/geolife_pointer_v45_20251229_023222/checkpoints/best.pt"
DIY_CHECKPOINT = "/data/next_loc_clean_v2/experiments/diy_pointer_v45_20251229_023930/checkpoints/best.pt"
GEOLIFE_DATA_DIR = "/data/next_loc_clean_v2/data/geolife_eps20/processed"
DIY_DATA_DIR = "/data/next_loc_clean_v2/data/diy_eps50/processed"
OUTPUT_DIR = Path(__file__).parent / "results"

# Time period definitions (in minutes from midnight)
TIME_PERIODS = {
    'Early Morning (00-06)': (0, 360),
    'Morning (06-12)': (360, 720),
    'Afternoon (12-18)': (720, 1080),
    'Evening (18-24)': (1080, 1440),
}


def set_seed(seed=42):
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
    
    data_dir = config['data']['data_dir']
    dataset_prefix = config['data']['dataset_prefix']
    metadata_path = os.path.join(data_dir, f"{dataset_prefix}_metadata.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Infer max_seq_len from checkpoint
    max_seq_len = checkpoint['model_state_dict']['position_bias'].shape[0]
    
    model = PointerNetworkV45(
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
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")
    with open(test_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_target_time_period(sample):
    """Get the time period of the target location (last time in sequence)."""
    # The last time in start_min_X corresponds to the last visit before prediction
    last_time = sample['start_min_X'][-1]  # in minutes from midnight
    
    for period_name, (start, end) in TIME_PERIODS.items():
        if start <= last_time < end:
            return period_name
    
    return 'Evening (18-24)'  # Default for edge cases


def categorize_by_time(data):
    """Categorize samples by time period."""
    time_groups = defaultdict(list)
    
    for idx, sample in enumerate(data):
        period = get_target_time_period(sample)
        time_groups[period].append(idx)
    
    return time_groups


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
            
            logits = model(x, x_dict)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            results, true_y, pred_y = calculate_correct_total_prediction(logits, y)
            all_results.append(results)
            all_true_y.append(true_y)
            all_pred_y.append(pred_y)
    
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
    """Run time-of-day analysis for a dataset."""
    print(f"\n{'='*60}")
    print(f"Running Time-of-Day Analysis for {dataset_name}")
    print(f"{'='*60}")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model, metadata = load_model(checkpoint_path, config, device)
    test_data = load_test_data(data_dir, dataset_prefix)
    
    print(f"Total test samples: {len(test_data)}")
    
    time_groups = categorize_by_time(test_data)
    
    results = []
    
    for period_name in TIME_PERIODS.keys():
        indices = time_groups.get(period_name, [])
        
        if len(indices) == 0:
            continue
        
        print(f"\nEvaluating {period_name} ({len(indices)} samples)...")
        perf = evaluate_group(model, test_data, indices, device)
        
        if perf:
            result = {
                'time_period': period_name,
                'num_samples': len(indices),
                **perf
            }
            results.append(result)
            print(f"  Acc@1: {perf['acc@1']:.2f}%, MRR: {perf['mrr']:.2f}%")
    
    return results


def create_visualizations(geolife_results, diy_results, output_dir):
    """Create visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    geolife_df = pd.DataFrame(geolife_results)
    diy_df = pd.DataFrame(diy_results)
    
    # Figure 1: Acc@1 by time period - Radar/Polar chart style bar
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Geolife
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(geolife_df)), geolife_df['acc@1'], color=colors[:len(geolife_df)], alpha=0.8)
    ax1.set_xticks(range(len(geolife_df)))
    ax1.set_xticklabels(geolife_df['time_period'], rotation=30, ha='right')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Geolife: Accuracy@1 by Time of Day')
    
    for bar, val in zip(bars1, geolife_df['acc@1']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # DIY
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(diy_df)), diy_df['acc@1'], color=colors[:len(diy_df)], alpha=0.8)
    ax2.set_xticks(range(len(diy_df)))
    ax2.set_xticklabels(diy_df['time_period'], rotation=30, ha='right')
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Accuracy@1 (%)')
    ax2.set_title('DIY: Accuracy@1 by Time of Day')
    
    for bar, val in zip(bars2, diy_df['acc@1']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'acc1_by_time_of_day.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: All metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['acc@1', 'acc@5', 'mrr', 'ndcg']
    titles = ['Accuracy@1', 'Accuracy@5', 'MRR', 'NDCG']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        x = np.arange(len(geolife_df))
        width = 0.35
        
        ax.bar(x - width/2, geolife_df[metric], width, label='Geolife', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, diy_df[metric], width, label='DIY', color='darkorange', alpha=0.8)
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(f'{title} by Time of Day')
        ax.set_xticks(x)
        ax.set_xticklabels(geolife_df['time_period'], rotation=30, ha='right')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_by_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Heatmap of metrics by time period
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Geolife heatmap
    metrics_cols = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg']
    geolife_heatmap = geolife_df[metrics_cols].values
    
    ax1 = axes[0]
    im1 = ax1.imshow(geolife_heatmap.T, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(geolife_df)))
    ax1.set_xticklabels(geolife_df['time_period'], rotation=45, ha='right')
    ax1.set_yticks(range(len(metrics_cols)))
    ax1.set_yticklabels(['Acc@1', 'Acc@5', 'Acc@10', 'MRR', 'NDCG'])
    ax1.set_title('Geolife: Metrics Heatmap by Time')
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations
    for i in range(len(geolife_df)):
        for j in range(len(metrics_cols)):
            ax1.text(i, j, f'{geolife_heatmap[i, j]:.1f}', ha='center', va='center', fontsize=8)
    
    # DIY heatmap
    diy_heatmap = diy_df[metrics_cols].values
    
    ax2 = axes[1]
    im2 = ax2.imshow(diy_heatmap.T, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(diy_df)))
    ax2.set_xticklabels(diy_df['time_period'], rotation=45, ha='right')
    ax2.set_yticks(range(len(metrics_cols)))
    ax2.set_yticklabels(['Acc@1', 'Acc@5', 'Acc@10', 'MRR', 'NDCG'])
    ax2.set_title('DIY: Metrics Heatmap by Time')
    plt.colorbar(im2, ax=ax2)
    
    for i in range(len(diy_df)):
        for j in range(len(metrics_cols)):
            ax2.text(i, j, f'{diy_heatmap[i, j]:.1f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap_by_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")


def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    elif hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    return obj


def save_results(geolife_results, diy_results, output_dir):
    """Save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'geolife': geolife_results,
        'diy': diy_results
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    geolife_df = pd.DataFrame(geolife_results)
    diy_df = pd.DataFrame(diy_results)
    
    geolife_df.to_csv(output_dir / 'geolife_results.csv', index=False)
    diy_df.to_csv(output_dir / 'diy_results.csv', index=False)
    
    # Summary table
    summary = []
    for dataset, df in [('Geolife', geolife_df), ('DIY', diy_df)]:
        for _, row in df.iterrows():
            summary.append({
                'Dataset': dataset,
                'Time Period': row['time_period'],
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
    
    print("\n" + "="*100)
    print("TIME-OF-DAY ANALYSIS - SUMMARY TABLE")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)


def main():
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    save_results(geolife_results, diy_results, OUTPUT_DIR)
    create_visualizations(geolife_results, diy_results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Experiment 2: Time-of-Day Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

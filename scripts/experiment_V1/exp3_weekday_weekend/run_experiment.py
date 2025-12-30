"""
Experiment 3: Weekday vs Weekend Analysis
==========================================
Compares model performance between weekdays and weekends to understand
how mobility patterns differ and affect prediction accuracy.
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

# Weekday definitions (0=Monday, 6=Sunday)
WEEKDAYS = {0, 1, 2, 3, 4}  # Monday-Friday
WEEKENDS = {5, 6}  # Saturday-Sunday

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint_path, config, device):
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


def categorize_by_day_type(data):
    """Categorize samples by weekday/weekend and individual days."""
    groups = {
        'Weekday': [],
        'Weekend': [],
    }
    day_groups = {day: [] for day in DAY_NAMES}
    
    for idx, sample in enumerate(data):
        last_weekday = sample['weekday_X'][-1]  # Last day in sequence
        
        if last_weekday in WEEKDAYS:
            groups['Weekday'].append(idx)
        else:
            groups['Weekend'].append(idx)
        
        day_groups[DAY_NAMES[last_weekday]].append(idx)
    
    return groups, day_groups


def evaluate_group(model, data, indices, device):
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
    print(f"\n{'='*60}")
    print(f"Running Weekday/Weekend Analysis for {dataset_name}")
    print(f"{'='*60}")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model, metadata = load_model(checkpoint_path, config, device)
    test_data = load_test_data(data_dir, dataset_prefix)
    
    print(f"Total test samples: {len(test_data)}")
    
    type_groups, day_groups = categorize_by_day_type(test_data)
    
    # Results for weekday/weekend
    type_results = []
    for group_name in ['Weekday', 'Weekend']:
        indices = type_groups[group_name]
        
        if len(indices) == 0:
            continue
        
        print(f"\nEvaluating {group_name} ({len(indices)} samples)...")
        perf = evaluate_group(model, test_data, indices, device)
        
        if perf:
            result = {
                'day_type': group_name,
                'num_samples': len(indices),
                **perf
            }
            type_results.append(result)
            print(f"  Acc@1: {perf['acc@1']:.2f}%, MRR: {perf['mrr']:.2f}%")
    
    # Results for individual days
    day_results = []
    for day_name in DAY_NAMES:
        indices = day_groups[day_name]
        
        if len(indices) == 0:
            continue
        
        print(f"\nEvaluating {day_name} ({len(indices)} samples)...")
        perf = evaluate_group(model, test_data, indices, device)
        
        if perf:
            result = {
                'day': day_name,
                'num_samples': len(indices),
                **perf
            }
            day_results.append(result)
            print(f"  Acc@1: {perf['acc@1']:.2f}%, MRR: {perf['mrr']:.2f}%")
    
    return type_results, day_results


def create_visualizations(geolife_type, geolife_day, diy_type, diy_day, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    geolife_type_df = pd.DataFrame(geolife_type)
    diy_type_df = pd.DataFrame(diy_type)
    geolife_day_df = pd.DataFrame(geolife_day)
    diy_day_df = pd.DataFrame(diy_day)
    
    # Figure 1: Weekday vs Weekend comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(2)
    width = 0.35
    
    ax1 = axes[0]
    ax1.bar(x - width/2, geolife_type_df['acc@1'], width, label='Geolife', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, diy_type_df['acc@1'], width, label='DIY', color='darkorange', alpha=0.8)
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Accuracy@1: Weekday vs Weekend')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Weekday', 'Weekend'])
    ax1.legend()
    
    for i, (g, d) in enumerate(zip(geolife_type_df['acc@1'], diy_type_df['acc@1'])):
        ax1.text(i - width/2, g + 0.5, f'{g:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, d + 0.5, f'{d:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[1]
    ax2.bar(x - width/2, geolife_type_df['mrr'], width, label='Geolife', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, diy_type_df['mrr'], width, label='DIY', color='darkorange', alpha=0.8)
    ax2.set_ylabel('MRR (%)')
    ax2.set_title('MRR: Weekday vs Weekend')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Weekday', 'Weekend'])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weekday_vs_weekend.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Performance by individual day
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Geolife
    ax1 = axes[0]
    colors = ['#2E86AB'] * 5 + ['#C73E1D'] * 2
    bars1 = ax1.bar(range(len(geolife_day_df)), geolife_day_df['acc@1'], color=colors, alpha=0.8)
    ax1.set_xticks(range(len(geolife_day_df)))
    ax1.set_xticklabels(geolife_day_df['day'], rotation=45, ha='right')
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Geolife: Accuracy@1 by Day of Week')
    ax1.axhline(y=geolife_day_df['acc@1'].mean(), color='red', linestyle='--', label='Mean')
    ax1.legend()
    
    for bar, val in zip(bars1, geolife_day_df['acc@1']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # DIY
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(diy_day_df)), diy_day_df['acc@1'], color=colors, alpha=0.8)
    ax2.set_xticks(range(len(diy_day_df)))
    ax2.set_xticklabels(diy_day_df['day'], rotation=45, ha='right')
    ax2.set_ylabel('Accuracy@1 (%)')
    ax2.set_title('DIY: Accuracy@1 by Day of Week')
    ax2.axhline(y=diy_day_df['acc@1'].mean(), color='red', linestyle='--', label='Mean')
    ax2.legend()
    
    for bar, val in zip(bars2, diy_day_df['acc@1']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_day.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: All metrics by day (heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    metrics_cols = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg']
    
    # Geolife
    geolife_heatmap = geolife_day_df[metrics_cols].values
    ax1 = axes[0]
    im1 = ax1.imshow(geolife_heatmap.T, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(geolife_day_df)))
    ax1.set_xticklabels(geolife_day_df['day'], rotation=45, ha='right')
    ax1.set_yticks(range(len(metrics_cols)))
    ax1.set_yticklabels(['Acc@1', 'Acc@5', 'Acc@10', 'MRR', 'NDCG'])
    ax1.set_title('Geolife: Metrics by Day of Week')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    for i in range(len(geolife_day_df)):
        for j in range(len(metrics_cols)):
            ax1.text(i, j, f'{geolife_heatmap[i, j]:.1f}', ha='center', va='center', fontsize=8)
    
    # DIY
    diy_heatmap = diy_day_df[metrics_cols].values
    ax2 = axes[1]
    im2 = ax2.imshow(diy_heatmap.T, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(diy_day_df)))
    ax2.set_xticklabels(diy_day_df['day'], rotation=45, ha='right')
    ax2.set_yticks(range(len(metrics_cols)))
    ax2.set_yticklabels(['Acc@1', 'Acc@5', 'Acc@10', 'MRR', 'NDCG'])
    ax2.set_title('DIY: Metrics by Day of Week')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    for i in range(len(diy_day_df)):
        for j in range(len(metrics_cols)):
            ax2.text(i, j, f'{diy_heatmap[i, j]:.1f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap_by_day.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Sample distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.bar(range(len(geolife_day_df)), geolife_day_df['num_samples'], 
            color=['#2E86AB'] * 5 + ['#C73E1D'] * 2, alpha=0.8)
    ax1.set_xticks(range(len(geolife_day_df)))
    ax1.set_xticklabels(geolife_day_df['day'], rotation=45, ha='right')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Geolife: Sample Distribution by Day')
    
    ax2 = axes[1]
    ax2.bar(range(len(diy_day_df)), diy_day_df['num_samples'],
            color=['#2E86AB'] * 5 + ['#C73E1D'] * 2, alpha=0.8)
    ax2.set_xticks(range(len(diy_day_df)))
    ax2.set_xticklabels(diy_day_df['day'], rotation=45, ha='right')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('DIY: Sample Distribution by Day')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_distribution_by_day.png', dpi=150, bbox_inches='tight')
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


def save_results(geolife_type, geolife_day, diy_type, diy_day, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'geolife_type': geolife_type,
        'geolife_day': geolife_day,
        'diy_type': diy_type,
        'diy_day': diy_day
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    # Save CSVs
    pd.DataFrame(geolife_type).to_csv(output_dir / 'geolife_type_results.csv', index=False)
    pd.DataFrame(diy_type).to_csv(output_dir / 'diy_type_results.csv', index=False)
    pd.DataFrame(geolife_day).to_csv(output_dir / 'geolife_day_results.csv', index=False)
    pd.DataFrame(diy_day).to_csv(output_dir / 'diy_day_results.csv', index=False)
    
    # Summary tables
    print("\n" + "="*100)
    print("WEEKDAY VS WEEKEND ANALYSIS - SUMMARY TABLE")
    print("="*100)
    
    type_summary = []
    for dataset, df in [('Geolife', pd.DataFrame(geolife_type)), ('DIY', pd.DataFrame(diy_type))]:
        for _, row in df.iterrows():
            type_summary.append({
                'Dataset': dataset,
                'Day Type': row['day_type'],
                'Samples': int(row['num_samples']),
                'Acc@1': f"{row['acc@1']:.2f}",
                'Acc@5': f"{row['acc@5']:.2f}",
                'Acc@10': f"{row['acc@10']:.2f}",
                'MRR': f"{row['mrr']:.2f}",
                'NDCG': f"{row['ndcg']:.2f}",
                'F1': f"{row['f1']*100:.2f}",
            })
    
    type_summary_df = pd.DataFrame(type_summary)
    print(type_summary_df.to_string(index=False))
    type_summary_df.to_csv(output_dir / 'summary_type_table.csv', index=False)
    
    print("\n" + "="*100)
    print("DAILY ANALYSIS - SUMMARY TABLE")
    print("="*100)
    
    day_summary = []
    for dataset, df in [('Geolife', pd.DataFrame(geolife_day)), ('DIY', pd.DataFrame(diy_day))]:
        for _, row in df.iterrows():
            day_summary.append({
                'Dataset': dataset,
                'Day': row['day'],
                'Samples': int(row['num_samples']),
                'Acc@1': f"{row['acc@1']:.2f}",
                'Acc@5': f"{row['acc@5']:.2f}",
                'MRR': f"{row['mrr']:.2f}",
            })
    
    day_summary_df = pd.DataFrame(day_summary)
    print(day_summary_df.to_string(index=False))
    day_summary_df.to_csv(output_dir / 'summary_day_table.csv', index=False)
    print("="*100)


def main():
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    geolife_type, geolife_day = run_experiment(
        'Geolife', 
        GEOLIFE_CHECKPOINT, 
        GEOLIFE_DATA_DIR, 
        'geolife_eps20_prev7', 
        device
    )
    
    diy_type, diy_day = run_experiment(
        'DIY', 
        DIY_CHECKPOINT, 
        DIY_DATA_DIR, 
        'diy_eps50_prev7', 
        device
    )
    
    save_results(geolife_type, geolife_day, diy_type, diy_day, OUTPUT_DIR)
    create_visualizations(geolife_type, geolife_day, diy_type, diy_day, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Experiment 3: Weekday/Weekend Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

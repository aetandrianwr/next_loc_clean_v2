"""
Experiment 7: Recency Analysis
==============================
Analyzes how the recency of visited locations affects prediction accuracy.
Examines whether predicting recent vs older locations is easier/harder.

This experiment evaluates:
- How "days ago" (diff) feature affects prediction
- Whether target locations visited recently are easier to predict
- Temporal decay patterns in prediction accuracy
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
from collections import defaultdict, Counter
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
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")
    with open(test_path, 'rb') as f:
        data = pickle.load(f)
    return data


def analyze_target_recency(test_data):
    """Analyze when the target location was last visited in history."""
    recency_info = []
    
    for idx, sample in enumerate(test_data):
        target = sample['Y']
        history = sample['X']
        diff_values = sample['diff']  # Days ago for each visit
        
        # Find if target is in history and when
        target_indices = [i for i, loc in enumerate(history) if loc == target]
        
        if target_indices:
            # Most recent occurrence
            most_recent_idx = target_indices[-1]
            most_recent_diff = diff_values[most_recent_idx]
            num_occurrences = len(target_indices)
        else:
            most_recent_idx = -1
            most_recent_diff = -1
            num_occurrences = 0
        
        # Average recency of all history
        avg_history_recency = np.mean(diff_values)
        
        # Last visit recency (how old is the most recent visit)
        last_visit_diff = diff_values[-1]
        
        recency_info.append({
            'idx': idx,
            'target_in_history': target in history,
            'target_most_recent_diff': most_recent_diff,
            'target_occurrences': num_occurrences,
            'avg_history_recency': avg_history_recency,
            'last_visit_diff': last_visit_diff,
            'seq_len': len(history),
        })
    
    return recency_info


def categorize_by_last_visit_recency(test_data, recency_info):
    """Group samples by recency of last visit."""
    groups = defaultdict(list)
    
    recency_bins = [
        ('Same Day (0)', 0, 1),
        ('1 Day Ago', 1, 2),
        ('2-3 Days Ago', 2, 4),
        ('4-7 Days Ago', 4, 8),
        ('> 7 Days Ago', 8, 100),
    ]
    
    for info in recency_info:
        last_diff = info['last_visit_diff']
        
        for bin_name, low, high in recency_bins:
            if low <= last_diff < high:
                groups[bin_name].append(info['idx'])
                break
    
    return groups, recency_bins


def categorize_by_target_recency(test_data, recency_info):
    """Group samples by recency of target in history."""
    groups = defaultdict(list)
    
    recency_bins = [
        ('Target: Same Day (0)', 0, 1),
        ('Target: 1 Day Ago', 1, 2),
        ('Target: 2-3 Days Ago', 2, 4),
        ('Target: 4-7 Days Ago', 4, 8),
        ('Target: > 7 Days Ago', 8, 100),
        ('Target: Not in History', -1, 0),
    ]
    
    for info in recency_info:
        target_diff = info['target_most_recent_diff']
        
        for bin_name, low, high in recency_bins:
            if low <= target_diff < high:
                groups[bin_name].append(info['idx'])
                break
    
    return groups, recency_bins


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
    print(f"Running Recency Analysis for {dataset_name}")
    print(f"{'='*60}")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model, metadata = load_model(checkpoint_path, config, device)
    test_data = load_test_data(data_dir, dataset_prefix)
    
    print(f"Total test samples: {len(test_data)}")
    
    # Analyze recency
    recency_info = analyze_target_recency(test_data)
    
    # Stats
    target_in_hist_count = sum(1 for info in recency_info if info['target_in_history'])
    print(f"Samples with target in history: {target_in_hist_count} ({target_in_hist_count/len(test_data)*100:.1f}%)")
    
    # Group by last visit recency
    print("\n--- Performance by Last Visit Recency ---")
    last_visit_groups, last_visit_bins = categorize_by_last_visit_recency(test_data, recency_info)
    
    last_visit_results = []
    for bin_name, _, _ in last_visit_bins:
        indices = last_visit_groups.get(bin_name, [])
        if len(indices) == 0:
            continue
        
        print(f"\nEvaluating {bin_name} ({len(indices)} samples)...")
        perf = evaluate_group(model, test_data, indices, device)
        
        if perf:
            result = {
                'recency_bin': bin_name,
                'num_samples': len(indices),
                **perf
            }
            last_visit_results.append(result)
            print(f"  Acc@1: {perf['acc@1']:.2f}%, MRR: {perf['mrr']:.2f}%")
    
    # Group by target recency
    print("\n--- Performance by Target Recency in History ---")
    target_groups, target_bins = categorize_by_target_recency(test_data, recency_info)
    
    target_recency_results = []
    bin_order = [
        'Target: Same Day (0)', 'Target: 1 Day Ago', 'Target: 2-3 Days Ago',
        'Target: 4-7 Days Ago', 'Target: > 7 Days Ago', 'Target: Not in History'
    ]
    
    for bin_name in bin_order:
        indices = target_groups.get(bin_name, [])
        if len(indices) == 0:
            continue
        
        print(f"\nEvaluating {bin_name} ({len(indices)} samples)...")
        perf = evaluate_group(model, test_data, indices, device)
        
        if perf:
            result = {
                'recency_bin': bin_name,
                'num_samples': len(indices),
                **perf
            }
            target_recency_results.append(result)
            print(f"  Acc@1: {perf['acc@1']:.2f}%, MRR: {perf['mrr']:.2f}%")
    
    return {
        'last_visit_results': last_visit_results,
        'target_recency_results': target_recency_results,
        'recency_info': recency_info,
    }


def create_visualizations(geolife_data, diy_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: Accuracy by Last Visit Recency
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    geolife_lv_df = pd.DataFrame(geolife_data['last_visit_results'])
    diy_lv_df = pd.DataFrame(diy_data['last_visit_results'])
    
    colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59']
    
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(geolife_lv_df)), geolife_lv_df['acc@1'], color=colors[:len(geolife_lv_df)], alpha=0.8)
    ax1.set_xticks(range(len(geolife_lv_df)))
    ax1.set_xticklabels(geolife_lv_df['recency_bin'], rotation=30, ha='right')
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Geolife: Accuracy by Last Visit Recency')
    
    for bar, val in zip(bars1, geolife_lv_df['acc@1']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(diy_lv_df)), diy_lv_df['acc@1'], color=colors[:len(diy_lv_df)], alpha=0.8)
    ax2.set_xticks(range(len(diy_lv_df)))
    ax2.set_xticklabels(diy_lv_df['recency_bin'], rotation=30, ha='right')
    ax2.set_ylabel('Accuracy@1 (%)')
    ax2.set_title('DIY: Accuracy by Last Visit Recency')
    
    for bar, val in zip(bars2, diy_lv_df['acc@1']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_last_visit_recency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Accuracy by Target Recency
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    geolife_tr_df = pd.DataFrame(geolife_data['target_recency_results'])
    diy_tr_df = pd.DataFrame(diy_data['target_recency_results'])
    
    colors2 = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']
    
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(geolife_tr_df)), geolife_tr_df['acc@1'], color=colors2[:len(geolife_tr_df)], alpha=0.8)
    ax1.set_xticks(range(len(geolife_tr_df)))
    ax1.set_xticklabels([b.replace('Target: ', '') for b in geolife_tr_df['recency_bin']], rotation=30, ha='right')
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Geolife: Accuracy by Target Recency in History')
    
    for bar, val in zip(bars1, geolife_tr_df['acc@1']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(diy_tr_df)), diy_tr_df['acc@1'], color=colors2[:len(diy_tr_df)], alpha=0.8)
    ax2.set_xticks(range(len(diy_tr_df)))
    ax2.set_xticklabels([b.replace('Target: ', '') for b in diy_tr_df['recency_bin']], rotation=30, ha='right')
    ax2.set_ylabel('Accuracy@1 (%)')
    ax2.set_title('DIY: Accuracy by Target Recency in History')
    
    for bar, val in zip(bars2, diy_tr_df['acc@1']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_target_recency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: All metrics by target recency
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['acc@1', 'acc@5', 'mrr', 'ndcg']
    titles = ['Accuracy@1', 'Accuracy@5', 'MRR', 'NDCG']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        x = np.arange(len(geolife_tr_df))
        width = 0.35
        
        ax.bar(x - width/2, geolife_tr_df[metric], width, label='Geolife', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, diy_tr_df[metric], width, label='DIY', color='darkorange', alpha=0.8)
        ax.set_xlabel('Target Recency')
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(f'{title} by Target Recency')
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace('Target: ', '') for b in geolife_tr_df['recency_bin']], rotation=30, ha='right')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_by_target_recency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Sample distribution by target recency
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.bar(range(len(geolife_tr_df)), geolife_tr_df['num_samples'], color=colors2[:len(geolife_tr_df)], alpha=0.8)
    ax1.set_xticks(range(len(geolife_tr_df)))
    ax1.set_xticklabels([b.replace('Target: ', '') for b in geolife_tr_df['recency_bin']], rotation=30, ha='right')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Geolife: Samples by Target Recency')
    
    ax2 = axes[1]
    ax2.bar(range(len(diy_tr_df)), diy_tr_df['num_samples'], color=colors2[:len(diy_tr_df)], alpha=0.8)
    ax2.set_xticks(range(len(diy_tr_df)))
    ax2.set_xticklabels([b.replace('Target: ', '') for b in diy_tr_df['recency_bin']], rotation=30, ha='right')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('DIY: Samples by Target Recency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_distribution_by_recency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Recency distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    geolife_avg_recency = [info['avg_history_recency'] for info in geolife_data['recency_info']]
    diy_avg_recency = [info['avg_history_recency'] for info in diy_data['recency_info']]
    
    ax1 = axes[0]
    ax1.hist(geolife_avg_recency, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Average History Recency (days)')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Geolife: Distribution of Average History Recency')
    
    ax2 = axes[1]
    ax2.hist(diy_avg_recency, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Average History Recency (days)')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('DIY: Distribution of Average History Recency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'history_recency_distribution.png', dpi=150, bbox_inches='tight')
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


def save_results(geolife_data, diy_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'geolife_last_visit': geolife_data['last_visit_results'],
        'geolife_target_recency': geolife_data['target_recency_results'],
        'diy_last_visit': diy_data['last_visit_results'],
        'diy_target_recency': diy_data['target_recency_results'],
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    # Save CSVs
    pd.DataFrame(geolife_data['last_visit_results']).to_csv(output_dir / 'geolife_last_visit_results.csv', index=False)
    pd.DataFrame(geolife_data['target_recency_results']).to_csv(output_dir / 'geolife_target_recency_results.csv', index=False)
    pd.DataFrame(diy_data['last_visit_results']).to_csv(output_dir / 'diy_last_visit_results.csv', index=False)
    pd.DataFrame(diy_data['target_recency_results']).to_csv(output_dir / 'diy_target_recency_results.csv', index=False)
    
    # Summary tables
    print("\n" + "="*100)
    print("RECENCY ANALYSIS - LAST VISIT RECENCY")
    print("="*100)
    
    lv_summary = []
    for dataset, results_list in [('Geolife', geolife_data['last_visit_results']), ('DIY', diy_data['last_visit_results'])]:
        for row in results_list:
            lv_summary.append({
                'Dataset': dataset,
                'Recency Bin': row['recency_bin'],
                'Samples': int(row['num_samples']),
                'Acc@1': f"{row['acc@1']:.2f}",
                'Acc@5': f"{row['acc@5']:.2f}",
                'MRR': f"{row['mrr']:.2f}",
            })
    
    lv_df = pd.DataFrame(lv_summary)
    print(lv_df.to_string(index=False))
    lv_df.to_csv(output_dir / 'summary_last_visit.csv', index=False)
    
    print("\n" + "="*100)
    print("RECENCY ANALYSIS - TARGET RECENCY IN HISTORY")
    print("="*100)
    
    tr_summary = []
    for dataset, results_list in [('Geolife', geolife_data['target_recency_results']), ('DIY', diy_data['target_recency_results'])]:
        for row in results_list:
            tr_summary.append({
                'Dataset': dataset,
                'Recency Bin': row['recency_bin'],
                'Samples': int(row['num_samples']),
                'Acc@1': f"{row['acc@1']:.2f}",
                'Acc@5': f"{row['acc@5']:.2f}",
                'MRR': f"{row['mrr']:.2f}",
            })
    
    tr_df = pd.DataFrame(tr_summary)
    print(tr_df.to_string(index=False))
    tr_df.to_csv(output_dir / 'summary_target_recency.csv', index=False)
    print("="*100)


def main():
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    geolife_data = run_experiment(
        'Geolife', 
        GEOLIFE_CHECKPOINT, 
        GEOLIFE_DATA_DIR, 
        'geolife_eps20_prev7', 
        device
    )
    
    diy_data = run_experiment(
        'DIY', 
        DIY_CHECKPOINT, 
        DIY_DATA_DIR, 
        'diy_eps50_prev7', 
        device
    )
    
    save_results(geolife_data, diy_data, OUTPUT_DIR)
    create_visualizations(geolife_data, diy_data, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Experiment 7: Recency Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

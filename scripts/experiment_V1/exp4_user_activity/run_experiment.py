"""
Experiment 4: User Activity Level Analysis
==========================================
Analyzes model performance across users with different activity levels
(low, medium, high frequency users) to understand how user behavior
patterns affect prediction accuracy.
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


def load_all_data(data_dir, dataset_prefix):
    """Load train and test data to compute user activity."""
    train_path = os.path.join(data_dir, f"{dataset_prefix}_train.pk")
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    return train_data, test_data


def compute_user_activity(train_data, test_data):
    """Compute user activity metrics from training data."""
    user_visit_counts = Counter()
    user_location_diversity = defaultdict(set)
    
    # Count from training data
    for sample in train_data:
        user = sample['user_X'][0]
        user_visit_counts[user] += 1
        for loc in sample['X']:
            user_location_diversity[user].add(loc)
        user_location_diversity[user].add(sample['Y'])
    
    # Also count from test data for complete picture
    for sample in test_data:
        user = sample['user_X'][0]
        user_visit_counts[user] += 1
    
    user_diversity = {user: len(locs) for user, locs in user_location_diversity.items()}
    
    return user_visit_counts, user_diversity


def categorize_users_by_activity(user_visit_counts, user_diversity):
    """Categorize users into activity levels using quartiles."""
    visit_counts = list(user_visit_counts.values())
    
    q25 = np.percentile(visit_counts, 25)
    q75 = np.percentile(visit_counts, 75)
    
    user_categories = {}
    for user, count in user_visit_counts.items():
        if count <= q25:
            user_categories[user] = 'Low Activity'
        elif count >= q75:
            user_categories[user] = 'High Activity'
        else:
            user_categories[user] = 'Medium Activity'
    
    return user_categories, q25, q75


def categorize_samples_by_user_activity(test_data, user_categories):
    """Group test samples by user activity level."""
    groups = defaultdict(list)
    
    for idx, sample in enumerate(test_data):
        user = sample['user_X'][0]
        if user in user_categories:
            groups[user_categories[user]].append(idx)
    
    return groups


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
    print(f"Running User Activity Analysis for {dataset_name}")
    print(f"{'='*60}")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model, metadata = load_model(checkpoint_path, config, device)
    train_data, test_data = load_all_data(data_dir, dataset_prefix)
    
    print(f"Total train samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")
    
    # Compute user activity
    user_visit_counts, user_diversity = compute_user_activity(train_data, test_data)
    user_categories, q25, q75 = categorize_users_by_activity(user_visit_counts, user_diversity)
    
    print(f"\nUser activity quartiles: Q25={q25:.0f}, Q75={q75:.0f}")
    print(f"Users by category: Low={sum(1 for c in user_categories.values() if c=='Low Activity')}, "
          f"Medium={sum(1 for c in user_categories.values() if c=='Medium Activity')}, "
          f"High={sum(1 for c in user_categories.values() if c=='High Activity')}")
    
    # Group samples by user activity
    activity_groups = categorize_samples_by_user_activity(test_data, user_categories)
    
    results = []
    activity_stats = {}
    
    for activity_level in ['Low Activity', 'Medium Activity', 'High Activity']:
        indices = activity_groups.get(activity_level, [])
        
        if len(indices) == 0:
            continue
        
        # Get users in this group
        users_in_group = set()
        for idx in indices:
            users_in_group.add(test_data[idx]['user_X'][0])
        
        avg_visits = np.mean([user_visit_counts[u] for u in users_in_group])
        avg_diversity = np.mean([user_diversity.get(u, 0) for u in users_in_group])
        
        print(f"\nEvaluating {activity_level} ({len(indices)} samples, {len(users_in_group)} users)...")
        print(f"  Avg visits: {avg_visits:.1f}, Avg location diversity: {avg_diversity:.1f}")
        
        perf = evaluate_group(model, test_data, indices, device)
        
        if perf:
            result = {
                'activity_level': activity_level,
                'num_samples': len(indices),
                'num_users': len(users_in_group),
                'avg_visits': avg_visits,
                'avg_diversity': avg_diversity,
                **perf
            }
            results.append(result)
            print(f"  Acc@1: {perf['acc@1']:.2f}%, MRR: {perf['mrr']:.2f}%")
            
            activity_stats[activity_level] = {
                'num_users': len(users_in_group),
                'avg_visits': avg_visits,
                'avg_diversity': avg_diversity,
            }
    
    return results, activity_stats, user_visit_counts, user_diversity


def create_visualizations(geolife_results, diy_results, geolife_stats, diy_stats, 
                         geolife_visits, diy_visits, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    geolife_df = pd.DataFrame(geolife_results)
    diy_df = pd.DataFrame(diy_results)
    
    # Figure 1: Accuracy by user activity level
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(3)
    width = 0.35
    
    ax1 = axes[0]
    ax1.bar(x - width/2, geolife_df['acc@1'], width, label='Geolife', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, diy_df['acc@1'], width, label='DIY', color='darkorange', alpha=0.8)
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Accuracy@1 by User Activity Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Low', 'Medium', 'High'])
    ax1.set_xlabel('User Activity Level')
    ax1.legend()
    
    for i, (g, d) in enumerate(zip(geolife_df['acc@1'], diy_df['acc@1'])):
        ax1.text(i - width/2, g + 0.5, f'{g:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, d + 0.5, f'{d:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[1]
    ax2.bar(x - width/2, geolife_df['mrr'], width, label='Geolife', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, diy_df['mrr'], width, label='DIY', color='darkorange', alpha=0.8)
    ax2.set_ylabel('MRR (%)')
    ax2.set_title('MRR by User Activity Level')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Low', 'Medium', 'High'])
    ax2.set_xlabel('User Activity Level')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_activity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: All metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['acc@1', 'acc@5', 'mrr', 'ndcg']
    titles = ['Accuracy@1', 'Accuracy@5', 'MRR', 'NDCG']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        ax.bar(x - width/2, geolife_df[metric], width, label='Geolife', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, diy_df[metric], width, label='DIY', color='darkorange', alpha=0.8)
        ax.set_xlabel('User Activity Level')
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(f'{title} by User Activity Level')
        ax.set_xticks(x)
        ax.set_xticklabels(['Low', 'Medium', 'High'])
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_by_activity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: User visit distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(list(geolife_visits.values()), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Visits')
    ax1.set_ylabel('Number of Users')
    ax1.set_title('Geolife: User Visit Distribution')
    ax1.axvline(np.percentile(list(geolife_visits.values()), 25), color='red', linestyle='--', label='Q25')
    ax1.axvline(np.percentile(list(geolife_visits.values()), 75), color='green', linestyle='--', label='Q75')
    ax1.legend()
    
    ax2 = axes[1]
    ax2.hist(list(diy_visits.values()), bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Visits')
    ax2.set_ylabel('Number of Users')
    ax2.set_title('DIY: User Visit Distribution')
    ax2.axvline(np.percentile(list(diy_visits.values()), 25), color='red', linestyle='--', label='Q25')
    ax2.axvline(np.percentile(list(diy_visits.values()), 75), color='green', linestyle='--', label='Q75')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'user_visit_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Sample distribution by activity level
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    colors = ['#1a9850', '#fdae61', '#d73027']
    ax1.bar(range(len(geolife_df)), geolife_df['num_samples'], color=colors, alpha=0.8)
    ax1.set_xticks(range(len(geolife_df)))
    ax1.set_xticklabels(geolife_df['activity_level'])
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Geolife: Samples by User Activity')
    
    ax2 = axes[1]
    ax2.bar(range(len(diy_df)), diy_df['num_samples'], color=colors, alpha=0.8)
    ax2.set_xticks(range(len(diy_df)))
    ax2.set_xticklabels(diy_df['activity_level'])
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('DIY: Samples by User Activity')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_distribution_by_activity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Accuracy vs Location Diversity
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(geolife_df['avg_diversity'], geolife_df['acc@1'], s=geolife_df['num_samples']/10, 
               label='Geolife', color='steelblue', alpha=0.7)
    ax.scatter(diy_df['avg_diversity'], diy_df['acc@1'], s=diy_df['num_samples']/10,
               label='DIY', color='darkorange', alpha=0.7)
    
    for _, row in geolife_df.iterrows():
        ax.annotate(row['activity_level'], (row['avg_diversity'], row['acc@1']), 
                   textcoords='offset points', xytext=(5, 5), fontsize=8)
    for _, row in diy_df.iterrows():
        ax.annotate(row['activity_level'], (row['avg_diversity'], row['acc@1']),
                   textcoords='offset points', xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('Average Location Diversity')
    ax.set_ylabel('Accuracy@1 (%)')
    ax.set_title('Accuracy@1 vs User Location Diversity')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_diversity.png', dpi=150, bbox_inches='tight')
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


def save_results(geolife_results, diy_results, geolife_stats, diy_stats, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'geolife': geolife_results,
        'diy': diy_results,
        'geolife_stats': geolife_stats,
        'diy_stats': diy_stats
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
                'Activity Level': row['activity_level'],
                'Users': int(row['num_users']),
                'Samples': int(row['num_samples']),
                'Avg Visits': f"{row['avg_visits']:.1f}",
                'Avg Diversity': f"{row['avg_diversity']:.1f}",
                'Acc@1': f"{row['acc@1']:.2f}",
                'Acc@5': f"{row['acc@5']:.2f}",
                'Acc@10': f"{row['acc@10']:.2f}",
                'MRR': f"{row['mrr']:.2f}",
                'NDCG': f"{row['ndcg']:.2f}",
                'F1': f"{row['f1']*100:.2f}",
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    
    print("\n" + "="*120)
    print("USER ACTIVITY ANALYSIS - SUMMARY TABLE")
    print("="*120)
    print(summary_df.to_string(index=False))
    print("="*120)


def main():
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    geolife_results, geolife_stats, geolife_visits, geolife_diversity = run_experiment(
        'Geolife', 
        GEOLIFE_CHECKPOINT, 
        GEOLIFE_DATA_DIR, 
        'geolife_eps20_prev7', 
        device
    )
    
    diy_results, diy_stats, diy_visits, diy_diversity = run_experiment(
        'DIY', 
        DIY_CHECKPOINT, 
        DIY_DATA_DIR, 
        'diy_eps50_prev7', 
        device
    )
    
    save_results(geolife_results, diy_results, geolife_stats, diy_stats, OUTPUT_DIR)
    create_visualizations(geolife_results, diy_results, geolife_stats, diy_stats,
                         geolife_visits, diy_visits, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Experiment 4: User Activity Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

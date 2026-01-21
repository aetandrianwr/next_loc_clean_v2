"""
Experiment 8: Cross-Dataset Comparison
=======================================
Comprehensive comparison between Geolife and DIY datasets to understand
dataset characteristics and their impact on prediction performance.

This experiment analyzes:
- Dataset statistics comparison
- Performance differences across various metrics
- Error pattern analysis
- Confusion matrix and error types
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
from sklearn.metrics import f1_score, confusion_matrix

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


def load_all_data(data_dir, dataset_prefix):
    """Load train, val, and test data."""
    train_path = os.path.join(data_dir, f"{dataset_prefix}_train.pk")
    val_path = os.path.join(data_dir, f"{dataset_prefix}_validation.pk")
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    return train_data, val_data, test_data


def compute_dataset_statistics(train_data, val_data, test_data, metadata):
    """Compute comprehensive dataset statistics."""
    all_data = train_data + val_data + test_data
    
    # User statistics
    user_visits = Counter()
    user_locations = defaultdict(set)
    for sample in all_data:
        user = sample['user_X'][0]
        user_visits[user] += 1
        for loc in sample['X']:
            user_locations[user].add(loc)
        user_locations[user].add(sample['Y'])
    
    user_diversity = [len(locs) for locs in user_locations.values()]
    
    # Location statistics
    location_visits = Counter()
    for sample in all_data:
        for loc in sample['X']:
            location_visits[loc] += 1
        location_visits[sample['Y']] += 1
    
    # Sequence statistics
    seq_lengths = [len(sample['X']) for sample in all_data]
    
    # Target in history rate
    target_in_hist = [sample['Y'] in sample['X'] for sample in test_data]
    
    stats = {
        'num_users': metadata['total_user_num'],
        'num_locations': metadata['total_loc_num'],
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'total_samples': len(all_data),
        'avg_visits_per_user': np.mean(list(user_visits.values())),
        'std_visits_per_user': np.std(list(user_visits.values())),
        'avg_user_diversity': np.mean(user_diversity),
        'std_user_diversity': np.std(user_diversity),
        'avg_visits_per_location': np.mean(list(location_visits.values())),
        'std_visits_per_location': np.std(list(location_visits.values())),
        'avg_seq_length': np.mean(seq_lengths),
        'std_seq_length': np.std(seq_lengths),
        'min_seq_length': np.min(seq_lengths),
        'max_seq_length': np.max(seq_lengths),
        'target_in_history_rate': np.mean(target_in_hist) * 100,
    }
    
    return stats


def evaluate_full(model, test_data, device):
    """Full evaluation with detailed predictions."""
    model.eval()
    
    all_results = []
    all_true_y = []
    all_pred_y = []
    all_top10_y = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Evaluating"):
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
            
            # Get top-10 predictions
            top10 = torch.topk(logits, k=min(10, logits.shape[-1]), dim=-1).indices[0].cpu().numpy()
            
            results, true_y, pred_y = calculate_correct_total_prediction(logits, y)
            all_results.append(results)
            all_true_y.append(true_y)
            all_pred_y.append(pred_y)
            all_top10_y.append(top10)
    
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
    perf['loss'] = total_loss / len(test_data)
    
    return {
        'metrics': perf,
        'true_y': all_true_y,
        'pred_y': np.array(all_pred_y_flat),
        'top10_y': np.array(all_top10_y),
    }


def analyze_errors(test_data, true_y, pred_y, top10_y):
    """Analyze prediction errors."""
    errors = {
        'total': len(true_y),
        'correct': np.sum(true_y == pred_y),
        'wrong': np.sum(true_y != pred_y),
        'target_in_hist_correct': 0,
        'target_in_hist_wrong': 0,
        'target_not_in_hist_correct': 0,
        'target_not_in_hist_wrong': 0,
        'wrong_but_in_top10': 0,
        'wrong_and_not_in_top10': 0,
    }
    
    for i, sample in enumerate(test_data):
        target = sample['Y']
        prediction = pred_y[i]
        top10 = top10_y[i]
        target_in_hist = target in sample['X']
        correct = (target == prediction)
        
        if target_in_hist:
            if correct:
                errors['target_in_hist_correct'] += 1
            else:
                errors['target_in_hist_wrong'] += 1
        else:
            if correct:
                errors['target_not_in_hist_correct'] += 1
            else:
                errors['target_not_in_hist_wrong'] += 1
        
        if not correct:
            if target in top10:
                errors['wrong_but_in_top10'] += 1
            else:
                errors['wrong_and_not_in_top10'] += 1
    
    return errors


def run_experiment(dataset_name, checkpoint_path, data_dir, dataset_prefix, device):
    print(f"\n{'='*60}")
    print(f"Running Cross-Dataset Analysis for {dataset_name}")
    print(f"{'='*60}")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model, metadata = load_model(checkpoint_path, config, device)
    train_data, val_data, test_data = load_all_data(data_dir, dataset_prefix)
    
    # Compute statistics
    stats = compute_dataset_statistics(train_data, val_data, test_data, metadata)
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Evaluate
    eval_results = evaluate_full(model, test_data, device)
    
    print(f"\nPerformance:")
    print(f"  Acc@1: {eval_results['metrics']['acc@1']:.2f}%")
    print(f"  Acc@5: {eval_results['metrics']['acc@5']:.2f}%")
    print(f"  Acc@10: {eval_results['metrics']['acc@10']:.2f}%")
    print(f"  MRR: {eval_results['metrics']['mrr']:.2f}%")
    print(f"  NDCG: {eval_results['metrics']['ndcg']:.2f}%")
    print(f"  F1: {eval_results['metrics']['f1']*100:.2f}%")
    
    # Error analysis
    errors = analyze_errors(test_data, eval_results['true_y'], eval_results['pred_y'], eval_results['top10_y'])
    
    print(f"\nError Analysis:")
    print(f"  Correct: {errors['correct']} ({errors['correct']/errors['total']*100:.1f}%)")
    print(f"  Wrong: {errors['wrong']} ({errors['wrong']/errors['total']*100:.1f}%)")
    print(f"  Target in history - Correct: {errors['target_in_hist_correct']}")
    print(f"  Target in history - Wrong: {errors['target_in_hist_wrong']}")
    print(f"  Target NOT in history - Correct: {errors['target_not_in_hist_correct']}")
    print(f"  Target NOT in history - Wrong: {errors['target_not_in_hist_wrong']}")
    print(f"  Wrong but in Top-10: {errors['wrong_but_in_top10']}")
    
    return {
        'stats': stats,
        'metrics': eval_results['metrics'],
        'errors': errors,
        'model_config': config['model'],
    }


def create_visualizations(geolife_data, diy_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: Dataset comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Stats comparison
    stat_keys = ['num_users', 'num_locations', 'total_samples', 
                 'avg_seq_length', 'avg_user_diversity', 'target_in_history_rate']
    stat_labels = ['Users', 'Locations', 'Samples', 'Avg Seq Len', 'Avg User Diversity', 'Target in History (%)']
    
    for ax, key, label in zip(axes.flat, stat_keys, stat_labels):
        vals = [geolife_data['stats'][key], diy_data['stats'][key]]
        bars = ax.bar(['Geolife', 'DIY'], vals, color=['steelblue', 'darkorange'], alpha=0.8)
        ax.set_ylabel(label)
        ax.set_title(label)
        
        for bar, val in zip(bars, vals):
            if isinstance(val, float):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                       f'{val}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Performance comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg']
    labels = ['Acc@1', 'Acc@5', 'Acc@10', 'MRR', 'NDCG']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    geolife_vals = [geolife_data['metrics'][m] for m in metrics]
    diy_vals = [diy_data['metrics'][m] for m in metrics]
    
    bars1 = ax.bar(x - width/2, geolife_vals, width, label='Geolife', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, diy_vals, width, label='DIY', color='darkorange', alpha=0.8)
    
    ax.set_ylabel('Performance (%)')
    ax.set_title('Performance Comparison: Geolife vs DIY')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    for bar, val in zip(bars1, geolife_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, diy_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Error analysis comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Geolife error breakdown
    ax1 = axes[0]
    geolife_errors = geolife_data['errors']
    labels1 = ['Correct\n(Target in Hist)', 'Wrong\n(Target in Hist)', 
               'Correct\n(Target NOT in Hist)', 'Wrong\n(Target NOT in Hist)']
    sizes1 = [
        geolife_errors['target_in_hist_correct'],
        geolife_errors['target_in_hist_wrong'],
        geolife_errors['target_not_in_hist_correct'],
        geolife_errors['target_not_in_hist_wrong'],
    ]
    colors1 = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']
    explode1 = (0.02, 0.02, 0.02, 0.02)
    
    ax1.pie(sizes1, explode=explode1, labels=labels1, colors=colors1, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.set_title('Geolife: Error Breakdown')
    
    # DIY error breakdown
    ax2 = axes[1]
    diy_errors = diy_data['errors']
    sizes2 = [
        diy_errors['target_in_hist_correct'],
        diy_errors['target_in_hist_wrong'],
        diy_errors['target_not_in_hist_correct'],
        diy_errors['target_not_in_hist_wrong'],
    ]
    
    ax2.pie(sizes2, explode=explode1, labels=labels1, colors=colors1, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax2.set_title('DIY: Error Breakdown')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_breakdown_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Model configuration comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    config_keys = ['d_model', 'nhead', 'num_layers', 'dim_feedforward']
    config_labels = ['d_model', 'nhead', 'num_layers', 'dim_ff']
    
    x = np.arange(len(config_keys))
    width = 0.35
    
    geolife_config = [geolife_data['model_config'].get(k, 0) for k in config_keys]
    diy_config = [diy_data['model_config'].get(k, 0) for k in config_keys]
    
    ax.bar(x - width/2, geolife_config, width, label='Geolife', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, diy_config, width, label='DIY', color='darkorange', alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('Model Configuration Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_config_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Summary radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    categories = ['Acc@1', 'Acc@5', 'MRR', 'NDCG', 'Target in Hist %']
    N = len(categories)
    
    geolife_values = [
        geolife_data['metrics']['acc@1'],
        geolife_data['metrics']['acc@5'],
        geolife_data['metrics']['mrr'],
        geolife_data['metrics']['ndcg'],
        geolife_data['stats']['target_in_history_rate'],
    ]
    diy_values = [
        diy_data['metrics']['acc@1'],
        diy_data['metrics']['acc@5'],
        diy_data['metrics']['mrr'],
        diy_data['metrics']['ndcg'],
        diy_data['stats']['target_in_history_rate'],
    ]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    geolife_values += geolife_values[:1]
    diy_values += diy_values[:1]
    
    ax.plot(angles, geolife_values, 'o-', linewidth=2, label='Geolife', color='steelblue')
    ax.fill(angles, geolife_values, alpha=0.25, color='steelblue')
    ax.plot(angles, diy_values, 'o-', linewidth=2, label='DIY', color='darkorange')
    ax.fill(angles, diy_values, alpha=0.25, color='darkorange')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Performance Radar Chart', y=1.08)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_radar.png', dpi=150, bbox_inches='tight')
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
        'geolife': {
            'stats': geolife_data['stats'],
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in geolife_data['metrics'].items()},
            'errors': geolife_data['errors'],
            'model_config': geolife_data['model_config'],
        },
        'diy': {
            'stats': diy_data['stats'],
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in diy_data['metrics'].items()},
            'errors': diy_data['errors'],
            'model_config': diy_data['model_config'],
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    # Summary tables
    print("\n" + "="*120)
    print("CROSS-DATASET COMPARISON - DATASET STATISTICS")
    print("="*120)
    
    stats_comparison = []
    stat_keys = list(geolife_data['stats'].keys())
    for key in stat_keys:
        stats_comparison.append({
            'Statistic': key,
            'Geolife': f"{geolife_data['stats'][key]:.2f}" if isinstance(geolife_data['stats'][key], float) else geolife_data['stats'][key],
            'DIY': f"{diy_data['stats'][key]:.2f}" if isinstance(diy_data['stats'][key], float) else diy_data['stats'][key],
        })
    
    stats_df = pd.DataFrame(stats_comparison)
    print(stats_df.to_string(index=False))
    stats_df.to_csv(output_dir / 'stats_comparison.csv', index=False)
    
    print("\n" + "="*120)
    print("CROSS-DATASET COMPARISON - PERFORMANCE METRICS")
    print("="*120)
    
    metrics_comparison = []
    metric_keys = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1', 'loss']
    for key in metric_keys:
        gval = geolife_data['metrics'][key]
        dval = diy_data['metrics'][key]
        if key == 'f1':
            gval *= 100
            dval *= 100
        metrics_comparison.append({
            'Metric': key.upper(),
            'Geolife': f"{gval:.2f}",
            'DIY': f"{dval:.2f}",
            'Difference': f"{dval - gval:+.2f}",
        })
    
    metrics_df = pd.DataFrame(metrics_comparison)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(output_dir / 'metrics_comparison.csv', index=False)
    
    print("\n" + "="*120)
    print("CROSS-DATASET COMPARISON - ERROR ANALYSIS")
    print("="*120)
    
    error_comparison = []
    error_keys = ['correct', 'wrong', 'target_in_hist_correct', 'target_in_hist_wrong',
                  'target_not_in_hist_correct', 'target_not_in_hist_wrong', 'wrong_but_in_top10']
    for key in error_keys:
        g_total = geolife_data['errors']['total']
        d_total = diy_data['errors']['total']
        error_comparison.append({
            'Error Type': key,
            'Geolife': f"{geolife_data['errors'][key]} ({geolife_data['errors'][key]/g_total*100:.1f}%)",
            'DIY': f"{diy_data['errors'][key]} ({diy_data['errors'][key]/d_total*100:.1f}%)",
        })
    
    error_df = pd.DataFrame(error_comparison)
    print(error_df.to_string(index=False))
    error_df.to_csv(output_dir / 'error_comparison.csv', index=False)
    print("="*120)


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
    print("Experiment 8: Cross-Dataset Comparison Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

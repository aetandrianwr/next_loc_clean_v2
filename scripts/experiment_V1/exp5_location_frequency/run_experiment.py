"""
Experiment 5: Location Frequency Analysis
==========================================
Analyzes model performance on frequent vs rare locations to understand
how location popularity affects prediction accuracy.

This experiment evaluates:
- Performance on frequently visited locations (common POIs)
- Performance on rarely visited locations (unusual destinations)
- The relationship between location frequency and prediction difficulty
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


def load_all_data(data_dir, dataset_prefix):
    """Load train and test data."""
    train_path = os.path.join(data_dir, f"{dataset_prefix}_train.pk")
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    return train_data, test_data


def compute_location_frequency(train_data):
    """Compute location visit frequency from training data."""
    location_counts = Counter()
    
    for sample in train_data:
        for loc in sample['X']:
            location_counts[loc] += 1
        location_counts[sample['Y']] += 1
    
    return location_counts


def categorize_locations_by_frequency(location_counts):
    """Categorize locations into frequency levels using percentiles."""
    counts = list(location_counts.values())
    
    p10 = np.percentile(counts, 10)
    p25 = np.percentile(counts, 25)
    p50 = np.percentile(counts, 50)
    p75 = np.percentile(counts, 75)
    p90 = np.percentile(counts, 90)
    
    location_categories = {}
    for loc, count in location_counts.items():
        if count <= p10:
            location_categories[loc] = 'Very Rare (≤P10)'
        elif count <= p25:
            location_categories[loc] = 'Rare (P10-P25)'
        elif count <= p50:
            location_categories[loc] = 'Occasional (P25-P50)'
        elif count <= p75:
            location_categories[loc] = 'Common (P50-P75)'
        elif count <= p90:
            location_categories[loc] = 'Frequent (P75-P90)'
        else:
            location_categories[loc] = 'Very Frequent (>P90)'
    
    return location_categories, {'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90}


def categorize_samples_by_target_location(test_data, location_categories, location_counts):
    """Group test samples by target location frequency."""
    groups = defaultdict(list)
    
    for idx, sample in enumerate(test_data):
        target = sample['Y']
        if target in location_categories:
            groups[location_categories[target]].append(idx)
        else:
            # New location not seen in training
            groups['Unseen Location'].append(idx)
    
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
    print(f"Running Location Frequency Analysis for {dataset_name}")
    print(f"{'='*60}")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model, metadata = load_model(checkpoint_path, config, device)
    train_data, test_data = load_all_data(data_dir, dataset_prefix)
    
    print(f"Total train samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")
    
    # Compute location frequency
    location_counts = compute_location_frequency(train_data)
    location_categories, percentiles = categorize_locations_by_frequency(location_counts)
    
    print(f"\nLocation frequency percentiles: {percentiles}")
    print(f"Unique locations: {len(location_counts)}")
    
    # Group samples by target location frequency
    freq_groups = categorize_samples_by_target_location(test_data, location_categories, location_counts)
    
    results = []
    freq_order = [
        'Very Rare (≤P10)', 'Rare (P10-P25)', 'Occasional (P25-P50)',
        'Common (P50-P75)', 'Frequent (P75-P90)', 'Very Frequent (>P90)', 'Unseen Location'
    ]
    
    for freq_level in freq_order:
        indices = freq_groups.get(freq_level, [])
        
        if len(indices) == 0:
            continue
        
        # Get average frequency of target locations
        target_freqs = [location_counts.get(test_data[idx]['Y'], 0) for idx in indices]
        avg_freq = np.mean(target_freqs)
        
        print(f"\nEvaluating {freq_level} ({len(indices)} samples, avg freq: {avg_freq:.1f})...")
        
        perf = evaluate_group(model, test_data, indices, device)
        
        if perf:
            result = {
                'frequency_level': freq_level,
                'num_samples': len(indices),
                'avg_frequency': avg_freq,
                **perf
            }
            results.append(result)
            print(f"  Acc@1: {perf['acc@1']:.2f}%, MRR: {perf['mrr']:.2f}%")
    
    return results, location_counts, percentiles


def create_visualizations(geolife_results, diy_results, geolife_counts, diy_counts, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    geolife_df = pd.DataFrame(geolife_results)
    diy_df = pd.DataFrame(diy_results)
    
    # Figure 1: Accuracy by location frequency
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
    
    # Geolife
    ax1 = axes[0]
    geolife_filtered = geolife_df[geolife_df['frequency_level'] != 'Unseen Location']
    bars1 = ax1.bar(range(len(geolife_filtered)), geolife_filtered['acc@1'], 
                    color=colors[:len(geolife_filtered)], alpha=0.8)
    ax1.set_xticks(range(len(geolife_filtered)))
    ax1.set_xticklabels(geolife_filtered['frequency_level'], rotation=45, ha='right')
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Geolife: Accuracy@1 by Target Location Frequency')
    
    for bar, val in zip(bars1, geolife_filtered['acc@1']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # DIY
    ax2 = axes[1]
    diy_filtered = diy_df[diy_df['frequency_level'] != 'Unseen Location']
    bars2 = ax2.bar(range(len(diy_filtered)), diy_filtered['acc@1'],
                    color=colors[:len(diy_filtered)], alpha=0.8)
    ax2.set_xticks(range(len(diy_filtered)))
    ax2.set_xticklabels(diy_filtered['frequency_level'], rotation=45, ha='right')
    ax2.set_ylabel('Accuracy@1 (%)')
    ax2.set_title('DIY: Accuracy@1 by Target Location Frequency')
    
    for bar, val in zip(bars2, diy_filtered['acc@1']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_location_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: All metrics by frequency
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['acc@1', 'acc@5', 'mrr', 'ndcg']
    titles = ['Accuracy@1', 'Accuracy@5', 'MRR', 'NDCG']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        x = np.arange(len(geolife_filtered))
        width = 0.35
        
        ax.bar(x - width/2, geolife_filtered[metric], width, label='Geolife', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, diy_filtered[metric], width, label='DIY', color='darkorange', alpha=0.8)
        ax.set_xlabel('Location Frequency')
        ax.set_ylabel(f'{title} (%)')
        ax.set_title(f'{title} by Target Location Frequency')
        ax.set_xticks(x)
        ax.set_xticklabels(geolife_filtered['frequency_level'], rotation=45, ha='right')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_by_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Location frequency distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(list(geolife_counts.values()), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Visit Count')
    ax1.set_ylabel('Number of Locations')
    ax1.set_title('Geolife: Location Frequency Distribution')
    ax1.set_yscale('log')
    
    ax2 = axes[1]
    ax2.hist(list(diy_counts.values()), bins=50, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Visit Count')
    ax2.set_ylabel('Number of Locations')
    ax2.set_title('DIY: Location Frequency Distribution')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'location_frequency_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Accuracy vs Average Frequency (scatter)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(geolife_filtered['avg_frequency'], geolife_filtered['acc@1'], 
               s=geolife_filtered['num_samples']/5, label='Geolife', color='steelblue', alpha=0.7)
    ax.scatter(diy_filtered['avg_frequency'], diy_filtered['acc@1'],
               s=diy_filtered['num_samples']/20, label='DIY', color='darkorange', alpha=0.7)
    
    for _, row in geolife_filtered.iterrows():
        ax.annotate(row['frequency_level'].split('(')[0].strip(), 
                   (row['avg_frequency'], row['acc@1']),
                   textcoords='offset points', xytext=(5, 5), fontsize=7)
    
    ax.set_xlabel('Average Location Frequency')
    ax.set_ylabel('Accuracy@1 (%)')
    ax.set_title('Accuracy@1 vs Average Target Location Frequency')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_avg_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Sample distribution by frequency level
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.bar(range(len(geolife_filtered)), geolife_filtered['num_samples'], 
            color=colors[:len(geolife_filtered)], alpha=0.8)
    ax1.set_xticks(range(len(geolife_filtered)))
    ax1.set_xticklabels(geolife_filtered['frequency_level'], rotation=45, ha='right')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Geolife: Samples by Target Location Frequency')
    
    ax2 = axes[1]
    ax2.bar(range(len(diy_filtered)), diy_filtered['num_samples'],
            color=colors[:len(diy_filtered)], alpha=0.8)
    ax2.set_xticks(range(len(diy_filtered)))
    ax2.set_xticklabels(diy_filtered['frequency_level'], rotation=45, ha='right')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('DIY: Samples by Target Location Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_distribution_by_frequency.png', dpi=150, bbox_inches='tight')
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


def save_results(geolife_results, diy_results, geolife_percentiles, diy_percentiles, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'geolife': geolife_results,
        'diy': diy_results,
        'geolife_percentiles': geolife_percentiles,
        'diy_percentiles': diy_percentiles
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
                'Frequency Level': row['frequency_level'],
                'Samples': int(row['num_samples']),
                'Avg Freq': f"{row['avg_frequency']:.1f}",
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
    
    print("\n" + "="*120)
    print("LOCATION FREQUENCY ANALYSIS - SUMMARY TABLE")
    print("="*120)
    print(summary_df.to_string(index=False))
    print("="*120)


def main():
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    geolife_results, geolife_counts, geolife_percentiles = run_experiment(
        'Geolife', 
        GEOLIFE_CHECKPOINT, 
        GEOLIFE_DATA_DIR, 
        'geolife_eps20_prev7', 
        device
    )
    
    diy_results, diy_counts, diy_percentiles = run_experiment(
        'DIY', 
        DIY_CHECKPOINT, 
        DIY_DATA_DIR, 
        'diy_eps50_prev7', 
        device
    )
    
    save_results(geolife_results, diy_results, geolife_percentiles, diy_percentiles, OUTPUT_DIR)
    create_visualizations(geolife_results, diy_results, geolife_counts, diy_counts, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Experiment 5: Location Frequency Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

"""
Experiment 6: Pointer vs Generator Gate Analysis
=================================================
Analyzes the pointer-generator gate behavior to understand when the model
relies on copying from history (pointer) vs generating from vocabulary.

This experiment evaluates:
- Gate values distribution
- Gate behavior by sample characteristics
- Correlation between gate values and prediction accuracy
- When pointer mechanism is most/least effective
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
import torch.nn.functional as F
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


class PointerGeneratorTransformerWithGate(PointerGeneratorTransformer):
    """Extended model that also returns gate values."""
    
    def forward_with_gate(self, x, x_dict):
        """Forward pass that returns both predictions and gate values."""
        import math
        
        x = x.T  # [batch_size, seq_len]
        batch_size, seq_len = x.shape
        device = x.device
        lengths = x_dict['len']
        
        # Embeddings
        loc_emb = self.loc_emb(x)
        user_emb = self.user_emb(x_dict['user']).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Temporal features
        time = torch.clamp(x_dict['time'].T, 0, 96)
        weekday = torch.clamp(x_dict['weekday'].T, 0, 7)
        recency = torch.clamp(x_dict['diff'].T, 0, 8)
        duration = torch.clamp(x_dict['duration'].T, 0, 99)
        
        temporal = torch.cat([
            self.time_emb(time),
            self.weekday_emb(weekday),
            self.recency_emb(recency),
            self.duration_emb(duration)
        ], dim=-1)
        
        # Position from end
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1)
        pos_emb = self.pos_from_end_emb(pos_from_end)
        
        # Combine features
        combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
        hidden = self.input_norm(self.input_proj(combined))
        hidden = hidden + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        mask = positions >= lengths.unsqueeze(1)
        encoded = self.transformer(hidden, src_key_padding_mask=mask)
        
        # Extract context
        batch_idx = torch.arange(batch_size, device=device)
        last_idx = (lengths - 1).clamp(min=0)
        context = encoded[batch_idx, last_idx]
        
        # Pointer attention
        query = self.pointer_query(context).unsqueeze(1)
        keys = self.pointer_key(encoded)
        ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
        ptr_scores = ptr_scores + self.position_bias[pos_from_end]
        ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))
        ptr_probs = F.softmax(ptr_scores, dim=-1)
        
        # Scatter pointer probabilities
        ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
        ptr_dist.scatter_add_(1, x, ptr_probs)
        
        # Generation distribution
        gen_probs = F.softmax(self.gen_head(context), dim=-1)
        
        # Gate and combine
        gate = self.ptr_gen_gate(context)
        final_probs = gate * ptr_dist + (1 - gate) * gen_probs
        
        return torch.log(final_probs + 1e-10), gate.squeeze(-1), ptr_dist, gen_probs


def analyze_gate_values(model, test_data, device):
    """Analyze gate values for all test samples."""
    model.eval()
    
    all_gate_values = []
    all_predictions_correct = []
    all_target_in_history = []
    all_seq_lengths = []
    all_unique_locs_in_seq = []
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Analyzing gate values"):
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
            
            logits, gate, ptr_dist, gen_probs = model.forward_with_gate(x, x_dict)
            
            # Get prediction
            pred = torch.argmax(logits, dim=-1).item()
            correct = (pred == sample['Y'])
            
            # Check if target is in history
            target_in_history = sample['Y'] in sample['X']
            
            all_gate_values.append(gate.item())
            all_predictions_correct.append(correct)
            all_target_in_history.append(target_in_history)
            all_seq_lengths.append(len(sample['X']))
            all_unique_locs_in_seq.append(len(set(sample['X'])))
    
    return {
        'gate_values': np.array(all_gate_values),
        'correct': np.array(all_predictions_correct),
        'target_in_history': np.array(all_target_in_history),
        'seq_lengths': np.array(all_seq_lengths),
        'unique_locs': np.array(all_unique_locs_in_seq),
    }


def evaluate_by_gate_range(model, test_data, device, gate_analysis):
    """Evaluate performance by gate value ranges."""
    gate_values = gate_analysis['gate_values']
    
    # Define gate ranges
    ranges = [
        ('Low Pointer (0-0.3)', 0.0, 0.3),
        ('Medium Pointer (0.3-0.5)', 0.3, 0.5),
        ('Balanced (0.5-0.7)', 0.5, 0.7),
        ('High Pointer (0.7-1.0)', 0.7, 1.0),
    ]
    
    results = []
    
    for range_name, low, high in ranges:
        mask = (gate_values >= low) & (gate_values < high)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        # Calculate metrics
        correct = gate_analysis['correct'][mask]
        acc1 = np.mean(correct) * 100
        
        target_in_hist = gate_analysis['target_in_history'][mask]
        target_in_hist_rate = np.mean(target_in_hist) * 100
        
        avg_gate = np.mean(gate_values[mask])
        avg_seq_len = np.mean(gate_analysis['seq_lengths'][mask])
        
        result = {
            'gate_range': range_name,
            'num_samples': len(indices),
            'acc@1': acc1,
            'avg_gate': avg_gate,
            'target_in_history_rate': target_in_hist_rate,
            'avg_seq_length': avg_seq_len,
        }
        results.append(result)
        print(f"  {range_name}: {len(indices)} samples, Acc@1={acc1:.2f}%, Target in history={target_in_hist_rate:.1f}%")
    
    return results


def run_experiment(dataset_name, checkpoint_path, data_dir, dataset_prefix, device):
    print(f"\n{'='*60}")
    print(f"Running Pointer-Generator Gate Analysis for {dataset_name}")
    print(f"{'='*60}")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.yaml')
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base model and convert to gated version
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = config['model']
    
    metadata_path = os.path.join(data_dir, f"{dataset_prefix}_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Infer max_seq_len from checkpoint
    max_seq_len = checkpoint['model_state_dict']['position_bias'].shape[0]
    
    model = PointerGeneratorTransformerWithGate(
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
    
    test_data = load_test_data(data_dir, dataset_prefix)
    print(f"Total test samples: {len(test_data)}")
    
    # Analyze gate values
    gate_analysis = analyze_gate_values(model, test_data, device)
    
    print(f"\nGate value statistics:")
    print(f"  Mean: {np.mean(gate_analysis['gate_values']):.4f}")
    print(f"  Std: {np.std(gate_analysis['gate_values']):.4f}")
    print(f"  Min: {np.min(gate_analysis['gate_values']):.4f}")
    print(f"  Max: {np.max(gate_analysis['gate_values']):.4f}")
    print(f"  Median: {np.median(gate_analysis['gate_values']):.4f}")
    
    # Evaluate by gate ranges
    print("\nPerformance by gate range:")
    range_results = evaluate_by_gate_range(model, test_data, device, gate_analysis)
    
    # Analysis: Target in history vs not
    print("\nAnalysis: Target in history vs not:")
    in_hist_mask = gate_analysis['target_in_history']
    not_in_hist_mask = ~in_hist_mask
    
    in_hist_stats = {
        'condition': 'Target in History',
        'num_samples': np.sum(in_hist_mask),
        'avg_gate': np.mean(gate_analysis['gate_values'][in_hist_mask]),
        'acc@1': np.mean(gate_analysis['correct'][in_hist_mask]) * 100,
    }
    
    not_in_hist_stats = {
        'condition': 'Target NOT in History',
        'num_samples': np.sum(not_in_hist_mask),
        'avg_gate': np.mean(gate_analysis['gate_values'][not_in_hist_mask]),
        'acc@1': np.mean(gate_analysis['correct'][not_in_hist_mask]) * 100,
    }
    
    print(f"  Target in history: {in_hist_stats['num_samples']} samples, "
          f"avg_gate={in_hist_stats['avg_gate']:.4f}, Acc@1={in_hist_stats['acc@1']:.2f}%")
    print(f"  Target NOT in history: {not_in_hist_stats['num_samples']} samples, "
          f"avg_gate={not_in_hist_stats['avg_gate']:.4f}, Acc@1={not_in_hist_stats['acc@1']:.2f}%")
    
    return {
        'gate_analysis': gate_analysis,
        'range_results': range_results,
        'in_history_stats': in_hist_stats,
        'not_in_history_stats': not_in_hist_stats,
    }


def create_visualizations(geolife_data, diy_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: Gate value distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(geolife_data['gate_analysis']['gate_values'], bins=50, color='steelblue', 
             alpha=0.7, edgecolor='black', density=True)
    ax1.axvline(np.mean(geolife_data['gate_analysis']['gate_values']), color='red', 
                linestyle='--', label=f"Mean: {np.mean(geolife_data['gate_analysis']['gate_values']):.3f}")
    ax1.axvline(0.5, color='green', linestyle=':', label='Balance (0.5)')
    ax1.set_xlabel('Gate Value (0=Generator, 1=Pointer)')
    ax1.set_ylabel('Density')
    ax1.set_title('Geolife: Pointer-Generator Gate Distribution')
    ax1.legend()
    
    ax2 = axes[1]
    ax2.hist(diy_data['gate_analysis']['gate_values'], bins=50, color='darkorange',
             alpha=0.7, edgecolor='black', density=True)
    ax2.axvline(np.mean(diy_data['gate_analysis']['gate_values']), color='red',
                linestyle='--', label=f"Mean: {np.mean(diy_data['gate_analysis']['gate_values']):.3f}")
    ax2.axvline(0.5, color='green', linestyle=':', label='Balance (0.5)')
    ax2.set_xlabel('Gate Value (0=Generator, 1=Pointer)')
    ax2.set_ylabel('Density')
    ax2.set_title('DIY: Pointer-Generator Gate Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gate_value_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Accuracy by gate range
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    geolife_range_df = pd.DataFrame(geolife_data['range_results'])
    diy_range_df = pd.DataFrame(diy_data['range_results'])
    
    ax1 = axes[0]
    colors = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4']
    bars1 = ax1.bar(range(len(geolife_range_df)), geolife_range_df['acc@1'], color=colors, alpha=0.8)
    ax1.set_xticks(range(len(geolife_range_df)))
    ax1.set_xticklabels(geolife_range_df['gate_range'], rotation=30, ha='right')
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Geolife: Accuracy by Gate Value Range')
    
    for bar, val in zip(bars1, geolife_range_df['acc@1']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(diy_range_df)), diy_range_df['acc@1'], color=colors, alpha=0.8)
    ax2.set_xticks(range(len(diy_range_df)))
    ax2.set_xticklabels(diy_range_df['gate_range'], rotation=30, ha='right')
    ax2.set_ylabel('Accuracy@1 (%)')
    ax2.set_title('DIY: Accuracy by Gate Value Range')
    
    for bar, val in zip(bars2, diy_range_df['acc@1']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_gate_range.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Gate vs Target in History
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Geolife
    ax1 = axes[0]
    gate_in_hist = geolife_data['gate_analysis']['gate_values'][geolife_data['gate_analysis']['target_in_history']]
    gate_not_in_hist = geolife_data['gate_analysis']['gate_values'][~geolife_data['gate_analysis']['target_in_history']]
    
    bp1 = ax1.boxplot([gate_in_hist, gate_not_in_hist], labels=['In History', 'Not in History'])
    ax1.set_ylabel('Gate Value')
    ax1.set_title('Geolife: Gate Value by Target Presence in History')
    
    # DIY
    ax2 = axes[1]
    gate_in_hist = diy_data['gate_analysis']['gate_values'][diy_data['gate_analysis']['target_in_history']]
    gate_not_in_hist = diy_data['gate_analysis']['gate_values'][~diy_data['gate_analysis']['target_in_history']]
    
    bp2 = ax2.boxplot([gate_in_hist, gate_not_in_hist], labels=['In History', 'Not in History'])
    ax2.set_ylabel('Gate Value')
    ax2.set_title('DIY: Gate Value by Target Presence in History')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gate_vs_target_in_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Target in history accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.35
    
    geolife_vals = [geolife_data['in_history_stats']['acc@1'], geolife_data['not_in_history_stats']['acc@1']]
    diy_vals = [diy_data['in_history_stats']['acc@1'], diy_data['not_in_history_stats']['acc@1']]
    
    ax.bar(x - width/2, geolife_vals, width, label='Geolife', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, diy_vals, width, label='DIY', color='darkorange', alpha=0.8)
    
    ax.set_ylabel('Accuracy@1 (%)')
    ax.set_title('Accuracy: Target in History vs Not')
    ax.set_xticks(x)
    ax.set_xticklabels(['Target in History', 'Target NOT in History'])
    ax.legend()
    
    for i, (g, d) in enumerate(zip(geolife_vals, diy_vals)):
        ax.text(i - width/2, g + 0.5, f'{g:.1f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, d + 0.5, f'{d:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_target_in_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Gate value vs Sequence Length (scatter)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.scatter(geolife_data['gate_analysis']['seq_lengths'], 
                geolife_data['gate_analysis']['gate_values'],
                c=geolife_data['gate_analysis']['correct'].astype(int),
                cmap='RdYlGn', alpha=0.3, s=10)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Gate Value')
    ax1.set_title('Geolife: Gate Value vs Sequence Length\n(Green=Correct, Red=Wrong)')
    
    ax2 = axes[1]
    ax2.scatter(diy_data['gate_analysis']['seq_lengths'],
                diy_data['gate_analysis']['gate_values'],
                c=diy_data['gate_analysis']['correct'].astype(int),
                cmap='RdYlGn', alpha=0.3, s=10)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Gate Value')
    ax2.set_title('DIY: Gate Value vs Sequence Length\n(Green=Correct, Red=Wrong)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gate_vs_sequence_length.png', dpi=150, bbox_inches='tight')
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
    
    # Convert numpy arrays to lists for JSON serialization
    results = {
        'geolife': {
            'range_results': geolife_data['range_results'],
            'in_history_stats': geolife_data['in_history_stats'],
            'not_in_history_stats': geolife_data['not_in_history_stats'],
            'gate_stats': {
                'mean': float(np.mean(geolife_data['gate_analysis']['gate_values'])),
                'std': float(np.std(geolife_data['gate_analysis']['gate_values'])),
                'min': float(np.min(geolife_data['gate_analysis']['gate_values'])),
                'max': float(np.max(geolife_data['gate_analysis']['gate_values'])),
                'median': float(np.median(geolife_data['gate_analysis']['gate_values'])),
            }
        },
        'diy': {
            'range_results': diy_data['range_results'],
            'in_history_stats': diy_data['in_history_stats'],
            'not_in_history_stats': diy_data['not_in_history_stats'],
            'gate_stats': {
                'mean': float(np.mean(diy_data['gate_analysis']['gate_values'])),
                'std': float(np.std(diy_data['gate_analysis']['gate_values'])),
                'min': float(np.min(diy_data['gate_analysis']['gate_values'])),
                'max': float(np.max(diy_data['gate_analysis']['gate_values'])),
                'median': float(np.median(diy_data['gate_analysis']['gate_values'])),
            }
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    # Save range results
    pd.DataFrame(geolife_data['range_results']).to_csv(output_dir / 'geolife_range_results.csv', index=False)
    pd.DataFrame(diy_data['range_results']).to_csv(output_dir / 'diy_range_results.csv', index=False)
    
    # Summary table
    print("\n" + "="*100)
    print("POINTER-GENERATOR GATE ANALYSIS - SUMMARY")
    print("="*100)
    
    print("\nGate Value Statistics:")
    print(f"  Geolife: Mean={results['geolife']['gate_stats']['mean']:.4f}, "
          f"Std={results['geolife']['gate_stats']['std']:.4f}, "
          f"Median={results['geolife']['gate_stats']['median']:.4f}")
    print(f"  DIY:     Mean={results['diy']['gate_stats']['mean']:.4f}, "
          f"Std={results['diy']['gate_stats']['std']:.4f}, "
          f"Median={results['diy']['gate_stats']['median']:.4f}")
    
    print("\nPerformance by Gate Range:")
    geolife_range_df = pd.DataFrame(geolife_data['range_results'])
    diy_range_df = pd.DataFrame(diy_data['range_results'])
    print("\nGeolife:")
    print(geolife_range_df.to_string(index=False))
    print("\nDIY:")
    print(diy_range_df.to_string(index=False))
    
    print("\nTarget in History Analysis:")
    print(f"  Geolife - In History: Acc@1={geolife_data['in_history_stats']['acc@1']:.2f}%, "
          f"Gate={geolife_data['in_history_stats']['avg_gate']:.4f}")
    print(f"  Geolife - NOT in History: Acc@1={geolife_data['not_in_history_stats']['acc@1']:.2f}%, "
          f"Gate={geolife_data['not_in_history_stats']['avg_gate']:.4f}")
    print(f"  DIY - In History: Acc@1={diy_data['in_history_stats']['acc@1']:.2f}%, "
          f"Gate={diy_data['in_history_stats']['avg_gate']:.4f}")
    print(f"  DIY - NOT in History: Acc@1={diy_data['not_in_history_stats']['acc@1']:.2f}%, "
          f"Gate={diy_data['not_in_history_stats']['avg_gate']:.4f}")
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
    print("Experiment 6: Pointer-Generator Gate Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

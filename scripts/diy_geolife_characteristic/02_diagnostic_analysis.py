"""
Model-Based Diagnostic Analysis: Pointer Mechanism Behavior.

This script conducts diagnostic analysis using the trained models to understand
WHY the pointer mechanism has different impact on DIY vs GeoLife datasets.

Key Experiments:
1. Gate Value Analysis - How does the model weight pointer vs generation?
2. Pointer Attention Analysis - Where does the pointer focus?
3. Per-Sample Prediction Analysis - When does pointer help vs hurt?
4. Stratified Performance by Copy-Ability

This script uses trained checkpoints without retraining.
"""

import os
import sys
import pickle
import json
import yaml
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.proposed.pointer_v45 import PointerNetworkV45
from src.training.train_pointer_v45 import NextLocationDataset, collate_fn, set_seed
from src.evaluation.metrics import calculate_correct_total_prediction

# Set style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset and checkpoint paths
DIY_CONFIG = BASE_DIR / "scripts/sci_hyperparam_tuning/configs/pointer_v45_diy_trial09.yaml"
GEOLIFE_CONFIG = BASE_DIR / "scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml"
DIY_CHECKPOINT = BASE_DIR / "experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt"
GEOLIFE_CHECKPOINT = BASE_DIR / "experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt"

SEED = 42


class PointerNetworkV45WithDiagnostics(PointerNetworkV45):
    """
    Extended PointerNetworkV45 that returns diagnostic information.
    """
    
    def forward_with_diagnostics(self, x: torch.Tensor, x_dict: dict):
        """
        Forward pass with diagnostic outputs.
        
        Returns:
            log_probs: [batch_size, num_locations]
            diagnostics: dict with gate values, pointer attention, etc.
        """
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
        ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / (self.d_model ** 0.5)
        ptr_scores_with_bias = ptr_scores + self.position_bias[pos_from_end]
        ptr_scores_masked = ptr_scores_with_bias.masked_fill(mask, float('-inf'))
        ptr_probs = F.softmax(ptr_scores_masked, dim=-1)
        
        # Scatter to vocabulary
        ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
        ptr_dist.scatter_add_(1, x, ptr_probs)
        
        # Generation distribution
        gen_logits = self.gen_head(context)
        gen_probs = F.softmax(gen_logits, dim=-1)
        
        # Gate
        gate = self.ptr_gen_gate(context)
        
        # Final distribution
        final_probs = gate * ptr_dist + (1 - gate) * gen_probs
        log_probs = torch.log(final_probs + 1e-10)
        
        # Diagnostics
        diagnostics = {
            'gate_values': gate.squeeze(-1),  # [batch_size]
            'ptr_attention': ptr_probs,  # [batch_size, seq_len]
            'ptr_dist': ptr_dist,  # [batch_size, num_locations]
            'gen_probs': gen_probs,  # [batch_size, num_locations]
            'input_locations': x,  # [batch_size, seq_len]
            'seq_lengths': lengths,  # [batch_size]
        }
        
        return log_probs, diagnostics


def load_model_with_diagnostics(checkpoint_path, config_path, device):
    """Load model with diagnostic capabilities."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint first to get correct max_seq_len
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get max_seq_len from checkpoint
    max_seq_len = checkpoint['model_state_dict']['position_bias'].shape[0]
    
    # Load dataset to get num_locations and num_users
    data_dir = BASE_DIR / config['data']['data_dir']
    dataset_prefix = config['data']['dataset_prefix']
    test_path = data_dir / f"{dataset_prefix}_test.pk"
    train_path = data_dir / f"{dataset_prefix}_train.pk"
    
    train_ds = NextLocationDataset(str(train_path), build_user_history=False)
    test_ds = NextLocationDataset(str(test_path), build_user_history=False)
    
    info = {
        'num_locations': max(train_ds.num_locations, test_ds.num_locations),
        'num_users': max(train_ds.num_users, test_ds.num_users),
        'max_seq_len': max_seq_len,
    }
    
    # Create model with exact max_seq_len from checkpoint
    model_cfg = config['model']
    model = PointerNetworkV45WithDiagnostics(
        num_locations=info['num_locations'],
        num_users=info['num_users'],
        d_model=model_cfg.get('d_model', 128),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 3),
        dim_feedforward=model_cfg.get('dim_feedforward', 256),
        dropout=model_cfg.get('dropout', 0.15),
        max_seq_len=max_seq_len,  # Use exact value from checkpoint
    )
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    test_loader = DataLoader(
        test_ds, 
        batch_size=64, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    return model, test_loader, info, config


@torch.no_grad()
def collect_diagnostics(model, loader, device, name):
    """Collect diagnostic information from model predictions."""
    print(f"\nCollecting diagnostics for {name}...")
    
    all_diagnostics = {
        'gate_values': [],
        'target_in_history': [],
        'target_in_last_k': {1: [], 3: [], 5: []},
        'ptr_correct': [],
        'gen_correct': [],
        'final_correct': [],
        'ptr_rank': [],
        'gen_rank': [],
        'final_rank': [],
        'target_position_from_end': [],
        'seq_lengths': [],
        'targets': [],
        'ptr_prob_on_target': [],
        'gen_prob_on_target': [],
        'final_prob_on_target': [],
    }
    
    for x, y, x_dict in tqdm(loader, desc=f"Processing {name}"):
        x = x.to(device)
        y = y.to(device)
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        
        log_probs, diag = model.forward_with_diagnostics(x, x_dict)
        
        batch_size = y.shape[0]
        
        for i in range(batch_size):
            target = y[i].item()
            input_seq = diag['input_locations'][i].cpu().numpy()
            seq_len = diag['seq_lengths'][i].item()
            valid_seq = input_seq[:seq_len]
            
            # Gate value
            all_diagnostics['gate_values'].append(diag['gate_values'][i].item())
            
            # Target in history
            target_in_hist = target in valid_seq
            all_diagnostics['target_in_history'].append(target_in_hist)
            
            # Target in last k
            for k in [1, 3, 5]:
                all_diagnostics['target_in_last_k'][k].append(target in valid_seq[-k:])
            
            # Target position from end (if in history)
            if target_in_hist:
                positions = [seq_len - j for j, loc in enumerate(valid_seq) if loc == target]
                all_diagnostics['target_position_from_end'].append(min(positions))
            else:
                all_diagnostics['target_position_from_end'].append(-1)
            
            # Predictions
            ptr_dist = diag['ptr_dist'][i]
            gen_probs = diag['gen_probs'][i]
            final_probs = torch.exp(log_probs[i])
            
            # Top-1 predictions
            ptr_pred = ptr_dist.argmax().item()
            gen_pred = gen_probs.argmax().item()
            final_pred = final_probs.argmax().item()
            
            all_diagnostics['ptr_correct'].append(ptr_pred == target)
            all_diagnostics['gen_correct'].append(gen_pred == target)
            all_diagnostics['final_correct'].append(final_pred == target)
            
            # Ranks (lower is better)
            ptr_rank = (ptr_dist > ptr_dist[target]).sum().item() + 1
            gen_rank = (gen_probs > gen_probs[target]).sum().item() + 1
            final_rank = (final_probs > final_probs[target]).sum().item() + 1
            
            all_diagnostics['ptr_rank'].append(ptr_rank)
            all_diagnostics['gen_rank'].append(gen_rank)
            all_diagnostics['final_rank'].append(final_rank)
            
            # Probabilities on target
            all_diagnostics['ptr_prob_on_target'].append(ptr_dist[target].item())
            all_diagnostics['gen_prob_on_target'].append(gen_probs[target].item())
            all_diagnostics['final_prob_on_target'].append(final_probs[target].item())
            
            all_diagnostics['seq_lengths'].append(seq_len)
            all_diagnostics['targets'].append(target)
    
    return all_diagnostics


def analyze_gate_behavior(diy_diag, geolife_diag, output_dir):
    """Analyze gate behavior differences between datasets."""
    print("\nAnalyzing gate behavior...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Gate value distribution
    axes[0].hist(diy_diag['gate_values'], bins=50, alpha=0.7, label='DIY', color='#3498db', density=True)
    axes[0].hist(geolife_diag['gate_values'], bins=50, alpha=0.7, label='GeoLife', color='#e74c3c', density=True)
    axes[0].axvline(np.mean(diy_diag['gate_values']), color='#3498db', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(geolife_diag['gate_values']), color='#e74c3c', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Gate Value (Higher = More Pointer)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('(a) Gate Value Distribution')
    axes[0].legend()
    
    # Gate value by target-in-history
    diy_gate_in = [g for g, t in zip(diy_diag['gate_values'], diy_diag['target_in_history']) if t]
    diy_gate_out = [g for g, t in zip(diy_diag['gate_values'], diy_diag['target_in_history']) if not t]
    geo_gate_in = [g for g, t in zip(geolife_diag['gate_values'], geolife_diag['target_in_history']) if t]
    geo_gate_out = [g for g, t in zip(geolife_diag['gate_values'], geolife_diag['target_in_history']) if not t]
    
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width/2, [np.mean(diy_gate_in), np.mean(diy_gate_out)], width, label='DIY', color='#3498db')
    axes[1].bar(x + width/2, [np.mean(geo_gate_in), np.mean(geo_gate_out)], width, label='GeoLife', color='#e74c3c')
    axes[1].set_ylabel('Mean Gate Value')
    axes[1].set_title('(b) Gate by Target-in-History')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Target in History', 'Target NOT in History'])
    axes[1].legend()
    
    # Gate value vs position of target
    valid_diy = [(g, p) for g, p in zip(diy_diag['gate_values'], diy_diag['target_position_from_end']) if p > 0]
    valid_geo = [(g, p) for g, p in zip(geolife_diag['gate_values'], geolife_diag['target_position_from_end']) if p > 0]
    
    if valid_diy:
        axes[2].scatter([p for _, p in valid_diy], [g for g, _ in valid_diy], alpha=0.3, label='DIY', color='#3498db', s=10)
    if valid_geo:
        axes[2].scatter([p for _, p in valid_geo], [g for g, _ in valid_geo], alpha=0.3, label='GeoLife', color='#e74c3c', s=10)
    axes[2].set_xlabel('Target Position from End')
    axes[2].set_ylabel('Gate Value')
    axes[2].set_title('(c) Gate vs Target Position')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_gate_analysis.png')
    plt.savefig(output_dir / 'fig5_gate_analysis.pdf')
    plt.close()
    
    # Summary statistics
    results = {
        'DIY': {
            'mean_gate': np.mean(diy_diag['gate_values']),
            'std_gate': np.std(diy_diag['gate_values']),
            'mean_gate_when_target_in_hist': np.mean(diy_gate_in) if diy_gate_in else 0,
            'mean_gate_when_target_out_hist': np.mean(diy_gate_out) if diy_gate_out else 0,
        },
        'GeoLife': {
            'mean_gate': np.mean(geolife_diag['gate_values']),
            'std_gate': np.std(geolife_diag['gate_values']),
            'mean_gate_when_target_in_hist': np.mean(geo_gate_in) if geo_gate_in else 0,
            'mean_gate_when_target_out_hist': np.mean(geo_gate_out) if geo_gate_out else 0,
        }
    }
    
    return results


def analyze_ptr_vs_gen_performance(diy_diag, geolife_diag, output_dir):
    """Analyze pointer vs generation head performance."""
    print("\nAnalyzing pointer vs generation performance...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Accuracy comparison
    for idx, (diag, name, color) in enumerate([(diy_diag, 'DIY', '#3498db'), (geolife_diag, 'GeoLife', '#e74c3c')]):
        # Overall accuracy
        ptr_acc = np.mean(diag['ptr_correct']) * 100
        gen_acc = np.mean(diag['gen_correct']) * 100
        final_acc = np.mean(diag['final_correct']) * 100
        
        # When target in history
        in_hist_mask = diag['target_in_history']
        ptr_acc_in = np.mean([c for c, m in zip(diag['ptr_correct'], in_hist_mask) if m]) * 100
        gen_acc_in = np.mean([c for c, m in zip(diag['gen_correct'], in_hist_mask) if m]) * 100
        final_acc_in = np.mean([c for c, m in zip(diag['final_correct'], in_hist_mask) if m]) * 100
        
        # When target NOT in history
        ptr_acc_out = np.mean([c for c, m in zip(diag['ptr_correct'], in_hist_mask) if not m]) * 100 if any(not m for m in in_hist_mask) else 0
        gen_acc_out = np.mean([c for c, m in zip(diag['gen_correct'], in_hist_mask) if not m]) * 100 if any(not m for m in in_hist_mask) else 0
        final_acc_out = np.mean([c for c, m in zip(diag['final_correct'], in_hist_mask) if not m]) * 100 if any(not m for m in in_hist_mask) else 0
        
        x = np.arange(3)
        width = 0.25
        
        axes[0, idx].bar(x - width, [ptr_acc_in, ptr_acc_out, ptr_acc], width, label='Pointer', color='#2ecc71')
        axes[0, idx].bar(x, [gen_acc_in, gen_acc_out, gen_acc], width, label='Generation', color='#9b59b6')
        axes[0, idx].bar(x + width, [final_acc_in, final_acc_out, final_acc], width, label='Combined', color='#e67e22')
        axes[0, idx].set_ylabel('Accuracy (%)')
        axes[0, idx].set_title(f'{name} - Accuracy by Component')
        axes[0, idx].set_xticks(x)
        axes[0, idx].set_xticklabels(['Target in Hist', 'Target NOT in Hist', 'Overall'])
        axes[0, idx].legend()
        axes[0, idx].set_ylim(0, 100)
    
    # Direct comparison
    categories = ['Ptr (in hist)', 'Ptr (out hist)', 'Gen (in hist)', 'Gen (out hist)', 'Final (overall)']
    
    in_hist_diy = diy_diag['target_in_history']
    in_hist_geo = geolife_diag['target_in_history']
    
    diy_vals = [
        np.mean([c for c, m in zip(diy_diag['ptr_correct'], in_hist_diy) if m]) * 100,
        np.mean([c for c, m in zip(diy_diag['ptr_correct'], in_hist_diy) if not m]) * 100 if any(not m for m in in_hist_diy) else 0,
        np.mean([c for c, m in zip(diy_diag['gen_correct'], in_hist_diy) if m]) * 100,
        np.mean([c for c, m in zip(diy_diag['gen_correct'], in_hist_diy) if not m]) * 100 if any(not m for m in in_hist_diy) else 0,
        np.mean(diy_diag['final_correct']) * 100,
    ]
    
    geo_vals = [
        np.mean([c for c, m in zip(geolife_diag['ptr_correct'], in_hist_geo) if m]) * 100,
        np.mean([c for c, m in zip(geolife_diag['ptr_correct'], in_hist_geo) if not m]) * 100 if any(not m for m in in_hist_geo) else 0,
        np.mean([c for c, m in zip(geolife_diag['gen_correct'], in_hist_geo) if m]) * 100,
        np.mean([c for c, m in zip(geolife_diag['gen_correct'], in_hist_geo) if not m]) * 100 if any(not m for m in in_hist_geo) else 0,
        np.mean(geolife_diag['final_correct']) * 100,
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    axes[0, 2].bar(x - width/2, diy_vals, width, label='DIY', color='#3498db')
    axes[0, 2].bar(x + width/2, geo_vals, width, label='GeoLife', color='#e74c3c')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Direct Comparison')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(categories, rotation=45, ha='right')
    axes[0, 2].legend()
    
    # Row 2: Rank analysis
    # MRR comparison
    diy_mrr_ptr = np.mean([1/r for r in diy_diag['ptr_rank']]) * 100
    diy_mrr_gen = np.mean([1/r for r in diy_diag['gen_rank']]) * 100
    diy_mrr_final = np.mean([1/r for r in diy_diag['final_rank']]) * 100
    
    geo_mrr_ptr = np.mean([1/r for r in geolife_diag['ptr_rank']]) * 100
    geo_mrr_gen = np.mean([1/r for r in geolife_diag['gen_rank']]) * 100
    geo_mrr_final = np.mean([1/r for r in geolife_diag['final_rank']]) * 100
    
    x = np.arange(3)
    width = 0.35
    axes[1, 0].bar(x - width/2, [diy_mrr_ptr, diy_mrr_gen, diy_mrr_final], width, label='DIY', color='#3498db')
    axes[1, 0].bar(x + width/2, [geo_mrr_ptr, geo_mrr_gen, geo_mrr_final], width, label='GeoLife', color='#e74c3c')
    axes[1, 0].set_ylabel('MRR (%)')
    axes[1, 0].set_title('MRR by Component')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Pointer', 'Generation', 'Combined'])
    axes[1, 0].legend()
    
    # Probability on target
    axes[1, 1].boxplot([diy_diag['ptr_prob_on_target'], diy_diag['gen_prob_on_target']], 
                       positions=[0.8, 1.2], widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor='#3498db', alpha=0.7))
    axes[1, 1].boxplot([geolife_diag['ptr_prob_on_target'], geolife_diag['gen_prob_on_target']], 
                       positions=[1.8, 2.2], widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor='#e74c3c', alpha=0.7))
    axes[1, 1].set_ylabel('Probability on Target')
    axes[1, 1].set_title('Probability Assignment to Target')
    axes[1, 1].set_xticks([1, 2])
    axes[1, 1].set_xticklabels(['DIY', 'GeoLife'])
    
    # Pointer advantage
    diy_ptr_advantage = np.array(diy_diag['ptr_prob_on_target']) - np.array(diy_diag['gen_prob_on_target'])
    geo_ptr_advantage = np.array(geolife_diag['ptr_prob_on_target']) - np.array(geolife_diag['gen_prob_on_target'])
    
    axes[1, 2].hist(diy_ptr_advantage, bins=50, alpha=0.7, label='DIY', color='#3498db', density=True)
    axes[1, 2].hist(geo_ptr_advantage, bins=50, alpha=0.7, label='GeoLife', color='#e74c3c', density=True)
    axes[1, 2].axvline(0, color='black', linestyle='--')
    axes[1, 2].axvline(np.mean(diy_ptr_advantage), color='#3498db', linestyle='--', linewidth=2)
    axes[1, 2].axvline(np.mean(geo_ptr_advantage), color='#e74c3c', linestyle='--', linewidth=2)
    axes[1, 2].set_xlabel('Pointer - Generation Probability')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Pointer Advantage Distribution')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_ptr_vs_gen.png')
    plt.savefig(output_dir / 'fig6_ptr_vs_gen.pdf')
    plt.close()
    
    # Summary
    results = {
        'DIY': {
            'ptr_acc_overall': np.mean(diy_diag['ptr_correct']) * 100,
            'gen_acc_overall': np.mean(diy_diag['gen_correct']) * 100,
            'final_acc_overall': np.mean(diy_diag['final_correct']) * 100,
            'mrr_ptr': diy_mrr_ptr,
            'mrr_gen': diy_mrr_gen,
            'mrr_final': diy_mrr_final,
            'mean_ptr_advantage': np.mean(diy_ptr_advantage),
        },
        'GeoLife': {
            'ptr_acc_overall': np.mean(geolife_diag['ptr_correct']) * 100,
            'gen_acc_overall': np.mean(geolife_diag['gen_correct']) * 100,
            'final_acc_overall': np.mean(geolife_diag['final_correct']) * 100,
            'mrr_ptr': geo_mrr_ptr,
            'mrr_gen': geo_mrr_gen,
            'mrr_final': geo_mrr_final,
            'mean_ptr_advantage': np.mean(geo_ptr_advantage),
        }
    }
    
    return results


def analyze_vocabulary_effect(diy_diag, geolife_diag, output_dir):
    """Analyze how vocabulary size affects generation head performance."""
    print("\nAnalyzing vocabulary effect...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Target frequency analysis
    diy_targets = Counter(diy_diag['targets'])
    geo_targets = Counter(geolife_diag['targets'])
    
    # Accuracy by target frequency
    diy_freq_acc = defaultdict(list)
    for target, correct in zip(diy_diag['targets'], diy_diag['gen_correct']):
        freq = diy_targets[target]
        diy_freq_acc[freq].append(correct)
    
    geo_freq_acc = defaultdict(list)
    for target, correct in zip(geolife_diag['targets'], geolife_diag['gen_correct']):
        freq = geo_targets[target]
        geo_freq_acc[freq].append(correct)
    
    # Plot
    diy_freqs = sorted(diy_freq_acc.keys())
    diy_accs = [np.mean(diy_freq_acc[f]) * 100 for f in diy_freqs]
    
    geo_freqs = sorted(geo_freq_acc.keys())
    geo_accs = [np.mean(geo_freq_acc[f]) * 100 for f in geo_freqs]
    
    axes[0].scatter(diy_freqs, diy_accs, alpha=0.5, label='DIY', color='#3498db', s=20)
    axes[0].scatter(geo_freqs, geo_accs, alpha=0.5, label='GeoLife', color='#e74c3c', s=20)
    axes[0].set_xlabel('Target Frequency in Test Set')
    axes[0].set_ylabel('Generation Head Accuracy (%)')
    axes[0].set_title('(a) Gen. Accuracy vs Target Frequency')
    axes[0].legend()
    axes[0].set_xscale('log')
    
    # Number of unique targets
    axes[1].bar(['DIY', 'GeoLife'], [len(diy_targets), len(geo_targets)], color=['#3498db', '#e74c3c'])
    axes[1].set_ylabel('Number of Unique Targets')
    axes[1].set_title('(b) Target Vocabulary Size')
    for i, v in enumerate([len(diy_targets), len(geo_targets)]):
        axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_vocabulary_effect.png')
    plt.savefig(output_dir / 'fig7_vocabulary_effect.pdf')
    plt.close()
    
    return {
        'DIY_unique_targets': len(diy_targets),
        'GeoLife_unique_targets': len(geo_targets),
    }


def create_summary_table(gate_results, perf_results, vocab_results, diy_diag, geolife_diag, output_dir):
    """Create comprehensive summary table."""
    
    summary_data = []
    
    # Gate Analysis
    summary_data.append({
        'Category': 'Gate Behavior',
        'Metric': 'Mean Gate Value',
        'DIY': f"{gate_results['DIY']['mean_gate']:.4f}",
        'GeoLife': f"{gate_results['GeoLife']['mean_gate']:.4f}",
        'Interpretation': 'Higher = More Pointer Reliance'
    })
    summary_data.append({
        'Category': 'Gate Behavior',
        'Metric': 'Gate (Target in History)',
        'DIY': f"{gate_results['DIY']['mean_gate_when_target_in_hist']:.4f}",
        'GeoLife': f"{gate_results['GeoLife']['mean_gate_when_target_in_hist']:.4f}",
        'Interpretation': 'Model should increase gate when target is in history'
    })
    
    # Performance Analysis
    summary_data.append({
        'Category': 'Component Accuracy',
        'Metric': 'Pointer-Only Acc@1 (%)',
        'DIY': f"{perf_results['DIY']['ptr_acc_overall']:.2f}",
        'GeoLife': f"{perf_results['GeoLife']['ptr_acc_overall']:.2f}",
        'Interpretation': 'Pointer head performance alone'
    })
    summary_data.append({
        'Category': 'Component Accuracy',
        'Metric': 'Generation-Only Acc@1 (%)',
        'DIY': f"{perf_results['DIY']['gen_acc_overall']:.2f}",
        'GeoLife': f"{perf_results['GeoLife']['gen_acc_overall']:.2f}",
        'Interpretation': 'Generation head performance alone'
    })
    summary_data.append({
        'Category': 'Component Accuracy',
        'Metric': 'Combined Acc@1 (%)',
        'DIY': f"{perf_results['DIY']['final_acc_overall']:.2f}",
        'GeoLife': f"{perf_results['GeoLife']['final_acc_overall']:.2f}",
        'Interpretation': 'Final combined performance'
    })
    summary_data.append({
        'Category': 'Component Accuracy',
        'Metric': 'Pointer Advantage',
        'DIY': f"{perf_results['DIY']['mean_ptr_advantage']:.4f}",
        'GeoLife': f"{perf_results['GeoLife']['mean_ptr_advantage']:.4f}",
        'Interpretation': 'Avg(P_ptr - P_gen) on target'
    })
    
    # MRR
    summary_data.append({
        'Category': 'MRR Analysis',
        'Metric': 'Pointer MRR (%)',
        'DIY': f"{perf_results['DIY']['mrr_ptr']:.2f}",
        'GeoLife': f"{perf_results['GeoLife']['mrr_ptr']:.2f}",
        'Interpretation': 'Mean reciprocal rank of pointer'
    })
    summary_data.append({
        'Category': 'MRR Analysis',
        'Metric': 'Generation MRR (%)',
        'DIY': f"{perf_results['DIY']['mrr_gen']:.2f}",
        'GeoLife': f"{perf_results['GeoLife']['mrr_gen']:.2f}",
        'Interpretation': 'Mean reciprocal rank of generation'
    })
    
    # Vocabulary
    summary_data.append({
        'Category': 'Vocabulary',
        'Metric': 'Unique Targets in Test',
        'DIY': f"{vocab_results['DIY_unique_targets']}",
        'GeoLife': f"{vocab_results['GeoLife_unique_targets']}",
        'Interpretation': 'More targets = harder for generation'
    })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / 'diagnostic_analysis_results.csv', index=False)
    
    # Markdown
    with open(output_dir / 'diagnostic_analysis_results.md', 'w') as f:
        f.write("# Diagnostic Analysis Results\n\n")
        f.write("## Model-Based Analysis Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Key Insights\n\n")
        
        ptr_diff = perf_results['GeoLife']['ptr_acc_overall'] - perf_results['DIY']['ptr_acc_overall']
        gen_diff = perf_results['GeoLife']['gen_acc_overall'] - perf_results['DIY']['gen_acc_overall']
        
        f.write(f"### 1. Pointer Performance Difference\n")
        f.write(f"- GeoLife pointer accuracy: {perf_results['GeoLife']['ptr_acc_overall']:.2f}%\n")
        f.write(f"- DIY pointer accuracy: {perf_results['DIY']['ptr_acc_overall']:.2f}%\n")
        f.write(f"- Difference: {ptr_diff:.2f}%\n\n")
        
        f.write(f"### 2. Generation Performance Difference\n")
        f.write(f"- GeoLife generation accuracy: {perf_results['GeoLife']['gen_acc_overall']:.2f}%\n")
        f.write(f"- DIY generation accuracy: {perf_results['DIY']['gen_acc_overall']:.2f}%\n")
        f.write(f"- Difference: {gen_diff:.2f}%\n\n")
        
        f.write(f"### 3. Critical Insight: Why Pointer Matters More for GeoLife\n")
        f.write(f"The generation head performs significantly worse on GeoLife ({perf_results['GeoLife']['gen_acc_overall']:.2f}%) ")
        f.write(f"compared to DIY ({perf_results['DIY']['gen_acc_overall']:.2f}%). ")
        f.write(f"This makes the pointer mechanism relatively more important for GeoLife.\n")
    
    return df


def main():
    print("=" * 70)
    print("DIAGNOSTIC ANALYSIS: Pointer Mechanism Behavior")
    print("=" * 70)
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load models
    print("\nLoading DIY model...")
    diy_model, diy_loader, diy_info, diy_config = load_model_with_diagnostics(
        DIY_CHECKPOINT, DIY_CONFIG, device
    )
    print(f"  Locations: {diy_info['num_locations']}, Users: {diy_info['num_users']}")
    
    print("\nLoading GeoLife model...")
    geo_model, geo_loader, geo_info, geo_config = load_model_with_diagnostics(
        GEOLIFE_CHECKPOINT, GEOLIFE_CONFIG, device
    )
    print(f"  Locations: {geo_info['num_locations']}, Users: {geo_info['num_users']}")
    
    # Collect diagnostics
    diy_diag = collect_diagnostics(diy_model, diy_loader, device, "DIY")
    geo_diag = collect_diagnostics(geo_model, geo_loader, device, "GeoLife")
    
    # Analyze
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    gate_results = analyze_gate_behavior(diy_diag, geo_diag, OUTPUT_DIR)
    perf_results = analyze_ptr_vs_gen_performance(diy_diag, geo_diag, OUTPUT_DIR)
    vocab_results = analyze_vocabulary_effect(diy_diag, geo_diag, OUTPUT_DIR)
    
    # Summary
    df = create_summary_table(gate_results, perf_results, vocab_results, diy_diag, geo_diag, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    print(f"\n1. GATE VALUES:")
    print(f"   DIY Mean Gate:     {gate_results['DIY']['mean_gate']:.4f}")
    print(f"   GeoLife Mean Gate: {gate_results['GeoLife']['mean_gate']:.4f}")
    
    print(f"\n2. POINTER ACCURACY:")
    print(f"   DIY Pointer Acc:     {perf_results['DIY']['ptr_acc_overall']:.2f}%")
    print(f"   GeoLife Pointer Acc: {perf_results['GeoLife']['ptr_acc_overall']:.2f}%")
    
    print(f"\n3. GENERATION ACCURACY:")
    print(f"   DIY Generation Acc:     {perf_results['DIY']['gen_acc_overall']:.2f}%")
    print(f"   GeoLife Generation Acc: {perf_results['GeoLife']['gen_acc_overall']:.2f}%")
    
    print(f"\n4. POINTER ADVANTAGE:")
    print(f"   DIY Mean Advantage:     {perf_results['DIY']['mean_ptr_advantage']:.4f}")
    print(f"   GeoLife Mean Advantage: {perf_results['GeoLife']['mean_ptr_advantage']:.4f}")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Save diagnostics for further analysis
    diag_summary = {
        'diy': {
            'n_samples': len(diy_diag['gate_values']),
            'mean_gate': float(np.mean(diy_diag['gate_values'])),
            'target_in_history_rate': float(np.mean(diy_diag['target_in_history'])) * 100,
            'ptr_acc': float(np.mean(diy_diag['ptr_correct'])) * 100,
            'gen_acc': float(np.mean(diy_diag['gen_correct'])) * 100,
            'final_acc': float(np.mean(diy_diag['final_correct'])) * 100,
        },
        'geolife': {
            'n_samples': len(geo_diag['gate_values']),
            'mean_gate': float(np.mean(geo_diag['gate_values'])),
            'target_in_history_rate': float(np.mean(geo_diag['target_in_history'])) * 100,
            'ptr_acc': float(np.mean(geo_diag['ptr_correct'])) * 100,
            'gen_acc': float(np.mean(geo_diag['gen_correct'])) * 100,
            'final_acc': float(np.mean(geo_diag['final_correct'])) * 100,
        }
    }
    
    with open(OUTPUT_DIR / 'diagnostic_summary.json', 'w') as f:
        json.dump(diag_summary, f, indent=2)


if __name__ == "__main__":
    main()

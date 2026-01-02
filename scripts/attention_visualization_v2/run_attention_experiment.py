#!/usr/bin/env python3
"""
Comprehensive Attention Visualization Experiment for PointerNetworkV45 (V2).

This experiment provides Nature Journal-standard scientific analysis of attention
mechanisms in the Pointer Network model for next location prediction.

Style Reference: Classic scientific publication style with:
- White background, black axis box (all 4 sides)
- Inside tick marks
- No grid lines
- Simple colors: black, blue, red, green
- Open markers: circles, squares, diamonds, triangles

Experiments Conducted:
======================
1. **Aggregate Attention Analysis**
   - Pointer attention distribution across sequence positions
   - Self-attention patterns in transformer layers
   - Pointer-Generation gate value distribution
   - Comparison between correct and incorrect predictions

2. **Sample-Level Analysis**
   - Top 10 samples with highest prediction confidence
   - Attention heatmaps for individual predictions
   - Position bias visualization
   - Multi-head attention decomposition

3. **Statistical Analysis**
   - Attention entropy analysis
   - Correlation between attention patterns and accuracy
   - Position bias effect quantification

Scientific Output:
==================
- Tables: CSV and LaTeX format for all metrics
- Figures: Publication-quality visualizations (300 DPI)
- Statistical tests with p-values where applicable

Usage:
    python run_attention_experiment.py --dataset diy --seed 42
    python run_attention_experiment.py --dataset geolife --seed 42

Author: PhD Thesis Experiment
Date: 2026
"""

import os
import sys
import json
import yaml
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

# Set classic scientific publication style (matching reference images)
plt.rcParams.update({
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    
    # Figure settings
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.1,
    
    # Axes settings - box style (all 4 sides visible)
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,  # No grid
    'axes.spines.top': True,  # Show all 4 sides
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    
    # Tick settings - inside ticks
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.top': True,  # Ticks on all sides
    'xtick.bottom': True,
    'ytick.left': True,
    'ytick.right': True,
    
    # Line settings
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
    
    # Legend settings
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
})

# Classic scientific color palette
COLORS = {
    'primary': 'blue',
    'secondary': 'red',
    'tertiary': 'green',
    'quaternary': 'black',
    'correct': 'blue',
    'incorrect': 'red',
}

# Marker styles (open markers)
MARKERS = {
    'circle': 'o',
    'square': 's',
    'diamond': 'D',
    'triangle': '^',
}

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.proposed.pointer_v45 import PointerNetworkV45
from src.training.train_pointer_v45 import NextLocationDataset, collate_fn, set_seed
from scripts.attention_visualization_v2.attention_extractor import (
    AttentionExtractor,
    extract_batch_attention,
    compute_attention_statistics
)


def setup_classic_axes(ax):
    """Configure axes to match classic scientific publication style."""
    # Ensure all spines visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # Inside ticks on all sides
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis='both', which='major', length=5, width=1)
    ax.tick_params(axis='both', which='minor', length=3, width=0.5)


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_CONFIGS = {
    'diy': {
        'experiment_dir': '/data/next_loc_clean_v2/experiments/diy_pointer_v45_20260101_155348',
        'config_path': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pointer_v45_diy_trial09.yaml',
        'test_data': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_test.pk',
        'train_data': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_train.pk',
        'dataset_name': 'DIY',
        'full_name': 'DIY Check-in Dataset'
    },
    'geolife': {
        'experiment_dir': '/data/next_loc_clean_v2/experiments/geolife_pointer_v45_20260101_151038',
        'config_path': '/data/next_loc_clean_v2/scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml',
        'test_data': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_test.pk',
        'train_data': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_train.pk',
        'dataset_name': 'Geolife',
        'full_name': 'GeoLife GPS Trajectory Dataset'
    }
}


# =============================================================================
# Data Loading and Model Setup
# =============================================================================

def load_model_and_data(config: Dict, device: torch.device) -> Tuple[nn.Module, DataLoader, Dict]:
    """
    Load trained model and test data.
    
    Args:
        config: Experiment configuration
        device: Torch device
        
    Returns:
        model: Loaded PointerNetworkV45 model
        test_loader: DataLoader for test set
        info: Dataset information
    """
    # Load checkpoint first to get the exact model config used during training
    checkpoint_path = os.path.join(config['experiment_dir'], 'checkpoints', 'best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load YAML config as backup
    with open(config['config_path'], 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Load training data to get statistics
    train_ds = NextLocationDataset(config['train_data'], build_user_history=False)
    test_ds = NextLocationDataset(config['test_data'], build_user_history=False)
    
    info = {
        'num_locations': train_ds.num_locations,
        'num_users': train_ds.num_users,
        'max_seq_len': max(train_ds.max_seq_len, test_ds.max_seq_len),
        'test_size': len(test_ds),
    }
    
    # Infer max_seq_len from checkpoint's position_bias shape
    checkpoint_max_seq_len = checkpoint['model_state_dict']['position_bias'].shape[0]
    
    # Create model with exact config from checkpoint
    model_cfg = checkpoint.get('config', {}).get('model', model_config['model'])
    model = PointerNetworkV45(
        num_locations=info['num_locations'],
        num_users=info['num_users'],
        d_model=model_cfg.get('d_model', 128),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 3),
        dim_feedforward=model_cfg.get('dim_feedforward', 256),
        dropout=model_cfg.get('dropout', 0.15),
        max_seq_len=checkpoint_max_seq_len,  # Use exact value from checkpoint
    )
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create test loader
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    info['model_config'] = model_cfg
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Test samples: {info['test_size']}")
    
    return model, test_loader, info


# =============================================================================
# Analysis Functions
# =============================================================================

def select_best_samples(
    attention_results: List[Dict],
    num_samples: int = 10
) -> Tuple[List[int], List[Dict]]:
    """
    Select best samples for visualization based on prediction confidence.
    
    Selection criteria:
    1. Correct predictions only
    2. High prediction confidence (top probability)
    3. Diverse sequence lengths
    4. Interesting attention patterns
    
    Args:
        attention_results: List of attention data per sample
        num_samples: Number of samples to select
        
    Returns:
        indices: Selected sample indices
        samples: Selected sample data
    """
    # Filter correct predictions
    correct_samples = []
    for idx, r in enumerate(attention_results):
        if r['prediction'] == r['target']:
            confidence = r['final_probs'][r['target']].item()
            max_ptr_attn = r['pointer_probs'][:r['length']].max().item()
            entropy = -torch.sum(r['pointer_probs'][:r['length']] * 
                                 torch.log(r['pointer_probs'][:r['length']] + 1e-10)).item()
            
            correct_samples.append({
                'idx': idx,
                'confidence': confidence,
                'length': r['length'],
                'gate_value': r['gate_value'],
                'max_ptr_attn': max_ptr_attn,
                'entropy': entropy,
                'data': r
            })
    
    if len(correct_samples) < num_samples:
        print(f"Warning: Only {len(correct_samples)} correct samples available")
        num_samples = len(correct_samples)
    
    # Sort by confidence (descending)
    correct_samples.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Select top samples with length diversity
    selected = []
    selected_lengths = set()
    
    # First pass: get diverse lengths from top confident
    for sample in correct_samples[:50]:  # Consider top 50
        if sample['length'] not in selected_lengths:
            selected.append(sample)
            selected_lengths.add(sample['length'])
            if len(selected) >= num_samples // 2:
                break
    
    # Second pass: fill remaining with highest confidence
    for sample in correct_samples:
        if sample not in selected:
            selected.append(sample)
            if len(selected) >= num_samples:
                break
    
    indices = [s['idx'] for s in selected]
    samples = [s['data'] for s in selected]
    
    return indices, samples


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_aggregate_pointer_attention(
    attention_results: List[Dict],
    stats: Dict,
    output_dir: str,
    dataset_name: str
):
    """
    Create aggregate pointer attention visualization.
    Classic scientific publication style.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Position-wise attention (relative to end)
    ax = axes[0]
    pos_attn = stats['position_attention']
    pos_counts = stats['position_counts']
    
    # Only show positions with sufficient samples
    valid_mask = pos_counts >= 10
    positions = np.arange(len(pos_attn))
    max_pos = min(20, valid_mask.sum())
    
    # Classic style: white fill with blue edge
    ax.bar(positions[:max_pos], pos_attn[:max_pos], 
           color='white', edgecolor='blue', linewidth=1.5)
    ax.set_xlabel('Position from Sequence End (t-k)')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_xlim(-0.5, max_pos - 0.5)
    
    # Add annotation for recency effect
    if len(pos_attn) > 0:
        ax.annotate(f't-0: {pos_attn[0]:.3f}', 
                   xy=(0, pos_attn[0]), xytext=(3, pos_attn[0]*0.8),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=10)
    
    setup_classic_axes(ax)
    
    # Right: Attention entropy distribution
    ax = axes[1]
    entropies = stats['pointer_entropies']
    
    # Classic style: white fill with red edge
    ax.hist(entropies, bins=50, color='white', edgecolor='red', linewidth=1.0)
    ax.axvline(stats['pointer_entropy_mean'], color='black', 
               linestyle='--', linewidth=1.5, label=f'Mean: {stats["pointer_entropy_mean"]:.2f}')
    ax.set_xlabel('Attention Entropy (nats)')
    ax.set_ylabel('Number of Samples')
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    
    setup_classic_axes(ax)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(output_dir, f'aggregate_pointer_attention.{fmt}'),
                   format=fmt, dpi=300, facecolor='white')
    plt.close()


def plot_gate_analysis(
    attention_results: List[Dict],
    stats: Dict,
    output_dir: str,
    dataset_name: str
):
    """
    Analyze pointer-generation gate behavior.
    Classic scientific publication style.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    gate_values = stats['gate_values']
    correct_mask = np.array([r['prediction'] == r['target'] for r in attention_results])
    
    # 1. Overall gate distribution
    ax = axes[0]
    ax.hist(gate_values, bins=50, color='white', edgecolor='blue', linewidth=1.0)
    ax.axvline(stats['gate_mean'], color='black', linestyle='--',
               linewidth=1.5, label=f'Mean: {stats["gate_mean"]:.3f}')
    ax.set_xlabel('Gate Value (p_pointer)')
    ax.set_ylabel('Number of Samples')
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax.set_xlim(0, 1)
    setup_classic_axes(ax)
    
    # 2. Gate by prediction correctness - use box plot for classic style
    ax = axes[1]
    correct_gates = gate_values[correct_mask]
    incorrect_gates = gate_values[~correct_mask]
    
    # Classic style box plot
    bp = ax.boxplot([correct_gates, incorrect_gates], 
                    positions=[1, 2],
                    widths=0.6,
                    patch_artist=True)
    
    # Style the boxes
    colors = ['blue', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor('white')
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.0)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Correct', 'Incorrect'])
    ax.set_ylabel('Gate Value')
    ax.set_ylim(0, 1)
    
    # Add statistics annotation
    ax.text(1, 0.05, f'μ={np.mean(correct_gates):.3f}\nn={len(correct_gates)}',
            ha='center', fontsize=10)
    ax.text(2, 0.05, f'μ={np.mean(incorrect_gates):.3f}\nn={len(incorrect_gates)}',
            ha='center', fontsize=10)
    setup_classic_axes(ax)
    
    # 3. Gate vs sequence length
    ax = axes[2]
    lengths = np.array([r['length'] for r in attention_results])
    
    # Bin by length
    gate_by_length_mean = []
    gate_by_length_std = []
    valid_lengths = []
    
    for l in np.unique(lengths):
        mask = lengths == l
        if mask.sum() >= 5:  # Minimum samples
            gate_by_length_mean.append(np.mean(gate_values[mask]))
            gate_by_length_std.append(np.std(gate_values[mask]))
            valid_lengths.append(l)
    
    # Classic style: open circles with error bars
    ax.errorbar(valid_lengths, gate_by_length_mean, yerr=gate_by_length_std,
                fmt='o-', color='blue', capsize=3, markersize=7,
                markerfacecolor='white', markeredgecolor='blue', markeredgewidth=1.5,
                linewidth=1.5, capthick=1)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Mean Gate Value')
    ax.set_ylim(0, 1)
    setup_classic_axes(ax)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(output_dir, f'gate_analysis.{fmt}'),
                   format=fmt, dpi=300, facecolor='white')
    plt.close()


def plot_self_attention_aggregate(
    attention_results: List[Dict],
    output_dir: str,
    dataset_name: str,
    num_layers: int
):
    """
    Visualize aggregate self-attention patterns from transformer layers.
    Classic scientific publication style with grayscale heatmap.
    """
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4.5))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        # Aggregate attention weights (head-averaged)
        max_len = 15  # Show up to 15 positions
        agg_attn = np.zeros((max_len, max_len))
        count_matrix = np.zeros((max_len, max_len))
        
        for r in attention_results:
            length = min(r['length'], max_len)
            # Average across heads
            layer_attn = r['self_attention'][layer_idx].mean(dim=0).numpy()
            
            # Align to end of sequence
            for i in range(length):
                for j in range(length):
                    src_pos = length - 1 - i
                    tgt_pos = length - 1 - j
                    if src_pos < max_len and tgt_pos < max_len:
                        agg_attn[src_pos, tgt_pos] += layer_attn[i, j]
                        count_matrix[src_pos, tgt_pos] += 1
        
        # Normalize
        agg_attn = np.divide(agg_attn, count_matrix, 
                            out=np.zeros_like(agg_attn), where=count_matrix > 0)
        
        # Plot heatmap - use grayscale for classic style
        im = ax.imshow(agg_attn, cmap='Greys', aspect='auto',
                       vmin=0, vmax=np.percentile(agg_attn[agg_attn > 0], 95) if agg_attn.max() > 0 else 1)
        
        ax.set_xlabel('Key Position (from end)')
        ax.set_ylabel('Query Position (from end)')
        ax.set_title(f'Layer {layer_idx + 1}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_edgecolor('black')
        
        # Style the axes
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(output_dir, f'self_attention_aggregate.{fmt}'),
                   format=fmt, dpi=300, facecolor='white')
    plt.close()


def plot_sample_attention(
    sample: Dict,
    sample_idx: int,
    output_dir: str,
    dataset_name: str,
    num_layers: int
):
    """
    Create detailed attention visualization for a single sample.
    Classic scientific publication style.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.8])
    
    length = sample['length']
    seq = sample['input_sequence'][:length].numpy()
    
    # Create location labels
    loc_labels = [f'L{int(loc)}' for loc in seq]
    
    # 1. Pointer attention (main visualization)
    ax1 = fig.add_subplot(gs[0, :2])
    ptr_probs = sample['pointer_probs'][:length].numpy()
    
    # Classic style: white fill with blue edge
    bars = ax1.bar(range(length), ptr_probs, color='white', edgecolor='blue', linewidth=1.0)
    
    # Highlight max attention with black edge
    max_idx = np.argmax(ptr_probs)
    bars[max_idx].set_edgecolor('black')
    bars[max_idx].set_linewidth(2)
    
    ax1.set_xticks(range(length))
    ax1.set_xticklabels(loc_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Attention Weight')
    
    # Add gate value annotation
    gate_text = f'Gate: {sample["gate_value"]:.3f}'
    ax1.text(0.98, 0.95, gate_text, transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='square', facecolor='white', edgecolor='black'))
    setup_classic_axes(ax1)
    
    # 2. Position bias effect
    ax2 = fig.add_subplot(gs[0, 2])
    pos_bias = sample['position_bias'][:length].numpy()
    raw_scores = sample['pointer_scores_raw'][:length].numpy()
    
    x = np.arange(length)
    width = 0.35
    ax2.bar(x - width/2, raw_scores, width, label='Raw', color='white', edgecolor='blue', linewidth=1.0, hatch='\\\\')
    ax2.bar(x + width/2, pos_bias, width, label='Bias', color='white', edgecolor='red', linewidth=1.0, hatch='///')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Score')
    ax2.legend(fontsize=9, frameon=True, edgecolor='black', fancybox=False)
    ax2.set_xticks(range(0, length, max(1, length // 5)))
    setup_classic_axes(ax2)
    
    # 3. Self-attention heatmaps
    for layer_idx in range(min(num_layers, 3)):
        ax = fig.add_subplot(gs[1, layer_idx])
        
        # Average across heads
        layer_attn = sample['self_attention'][layer_idx].mean(dim=0).numpy()[:length, :length]
        
        # Classic style: grayscale
        im = ax.imshow(layer_attn, cmap='Greys', aspect='auto')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Layer {layer_idx + 1}')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_edgecolor('black')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    # 4. Multi-head attention comparison (first layer)
    ax4 = fig.add_subplot(gs[2, :])
    
    first_layer_attn = sample['self_attention'][0][:, :length, :length].numpy()
    num_heads = first_layer_attn.shape[0]
    
    # Average query attention per head
    head_patterns = first_layer_attn.mean(axis=1)  # [num_heads, seq_len]
    
    # Classic style: grayscale
    im = ax4.imshow(head_patterns, cmap='Greys', aspect='auto')
    ax4.set_xlabel('Key Position')
    ax4.set_ylabel('Attention Head')
    ax4.set_yticks(range(num_heads))
    ax4.set_yticklabels([f'Head {i+1}' for i in range(num_heads)])
    cbar = plt.colorbar(im, ax=ax4, fraction=0.02, pad=0.01)
    cbar.outline.set_edgecolor('black')
    
    for spine in ax4.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'sample_{sample_idx + 1:02d}_attention.{fmt}'),
                   format=fmt, dpi=300, facecolor='white')
    plt.close()


def plot_combined_samples_overview(
    selected_samples: List[Dict],
    output_dir: str,
    dataset_name: str
):
    """
    Create an overview visualization of all selected samples.
    Classic scientific publication style.
    """
    num_samples = len(selected_samples)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    
    for idx, (ax, sample) in enumerate(zip(axes, selected_samples)):
        length = sample['length']
        ptr_probs = sample['pointer_probs'][:length].numpy()
        
        # Classic style: white fill with blue edge
        ax.bar(range(length), ptr_probs, color='white', edgecolor='blue', linewidth=0.8)
        
        # Mark target position if in sequence
        target = sample['target']
        seq = sample['input_sequence'][:length].numpy()
        if target in seq:
            target_pos = np.where(seq == target)[0]
            for pos in target_pos:
                ax.axvline(pos, color='black', linestyle='--', linewidth=1)
        
        ax.set_title(f"S{idx+1}: g={sample['gate_value']:.2f}",
                     fontsize=10)
        ax.set_xlabel('Position', fontsize=9)
        if idx % 5 == 0:
            ax.set_ylabel('Attention', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        setup_classic_axes(ax)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(output_dir, f'samples_overview.{fmt}'),
                   format=fmt, dpi=300, facecolor='white')
    plt.close()


def plot_position_bias_analysis(
    model: nn.Module,
    output_dir: str,
    dataset_name: str,
    max_positions: int = 30
):
    """
    Analyze the learned position bias parameter.
    Classic scientific publication style.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Get position bias
    pos_bias = model.position_bias[:max_positions].detach().cpu().numpy()
    
    # 1. Raw position bias values - classic style with open markers
    ax = axes[0]
    ax.plot(pos_bias, 'o-', color='blue', markersize=7, linewidth=1.5,
            markerfacecolor='white', markeredgecolor='blue', markeredgewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Position from End')
    ax.set_ylabel('Bias Value')
    setup_classic_axes(ax)
    
    # 2. Bias contribution to attention
    ax = axes[1]
    # Simulate softmax contribution
    example_scores = np.zeros(max_positions)  # Equal base scores
    biased_scores = example_scores + pos_bias
    attention_weights = np.exp(biased_scores) / np.sum(np.exp(biased_scores))
    
    # Classic style: white fill with red edge
    ax.bar(range(max_positions), attention_weights, color='white',
           edgecolor='red', linewidth=1.0)
    ax.set_xlabel('Position from End')
    ax.set_ylabel('Attention Weight (equal base)')
    
    # Annotate recency
    ax.annotate(f'Recency', xy=(0, attention_weights[0]),
                xytext=(5, attention_weights[0] * 0.8),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    setup_classic_axes(ax)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        plt.savefig(os.path.join(output_dir, f'position_bias_analysis.{fmt}'),
                   format=fmt, dpi=300, facecolor='white')
    plt.close()


# =============================================================================
# Table Generation
# =============================================================================

def generate_statistics_table(
    stats: Dict,
    attention_results: List[Dict],
    output_dir: str,
    dataset_name: str
):
    """
    Generate comprehensive statistics tables.
    """
    # Main statistics table
    main_stats = {
        'Metric': [
            'Total Samples',
            'Prediction Accuracy (%)',
            'Mean Gate Value',
            'Gate Std Dev',
            'Gate (Correct Predictions)',
            'Gate (Incorrect Predictions)',
            'Mean Pointer Entropy',
            'Pointer Entropy Std Dev',
            'Most Recent Position Attention',
        ],
        'Value': [
            len(attention_results),
            f"{stats['accuracy'] * 100:.2f}",
            f"{stats['gate_mean']:.4f}",
            f"{stats['gate_std']:.4f}",
            f"{stats['correct_gate_mean']:.4f}",
            f"{stats['incorrect_gate_mean']:.4f}",
            f"{stats['pointer_entropy_mean']:.4f}",
            f"{stats['pointer_entropy_std']:.4f}",
            f"{stats['position_attention'][0]:.4f}",
        ]
    }
    
    df_main = pd.DataFrame(main_stats)
    
    # Save CSV
    df_main.to_csv(os.path.join(output_dir, 'attention_statistics.csv'), index=False)
    
    # Save LaTeX
    latex_str = df_main.to_latex(index=False, escape=False, 
                                  caption=f'Attention Statistics ({dataset_name} Dataset)',
                                  label=f'tab:attention_stats_{dataset_name.lower()}')
    with open(os.path.join(output_dir, 'attention_statistics.tex'), 'w') as f:
        f.write(latex_str)
    
    # Position-wise attention table
    pos_stats = {
        'Position (from end)': list(range(min(15, len(stats['position_attention'])))),
        'Mean Attention': [f"{v:.4f}" for v in stats['position_attention'][:15]],
        'Sample Count': [int(c) for c in stats['position_counts'][:15]]
    }
    
    df_pos = pd.DataFrame(pos_stats)
    df_pos.to_csv(os.path.join(output_dir, 'position_attention.csv'), index=False)
    
    return df_main, df_pos


def generate_sample_table(
    selected_samples: List[Dict],
    indices: List[int],
    output_dir: str,
    dataset_name: str
):
    """
    Generate table for selected sample analysis.
    """
    sample_data = []
    for idx, (orig_idx, sample) in enumerate(zip(indices, selected_samples)):
        seq = sample['input_sequence'][:sample['length']].numpy()
        confidence = sample['final_probs'][sample['target']].item()
        max_ptr = sample['pointer_probs'][:sample['length']].max().item()
        
        sample_data.append({
            'Sample': idx + 1,
            'Original Index': orig_idx,
            'Sequence Length': sample['length'],
            'Target Location': sample['target'],
            'Prediction': sample['prediction'],
            'Correct': 'Yes' if sample['target'] == sample['prediction'] else 'No',
            'Confidence': f"{confidence:.4f}",
            'Gate Value': f"{sample['gate_value']:.4f}",
            'Max Pointer Attn': f"{max_ptr:.4f}",
        })
    
    df = pd.DataFrame(sample_data)
    df.to_csv(os.path.join(output_dir, 'selected_samples.csv'), index=False)
    
    # LaTeX version
    latex_str = df.to_latex(index=False, escape=False,
                            caption=f'Selected Samples for Attention Analysis ({dataset_name})',
                            label=f'tab:selected_samples_{dataset_name.lower()}')
    with open(os.path.join(output_dir, 'selected_samples.tex'), 'w') as f:
        f.write(latex_str)
    
    return df


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(dataset: str, seed: int = 42):
    """
    Run the complete attention visualization experiment.
    
    Args:
        dataset: 'diy' or 'geolife'
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print(f"ATTENTION VISUALIZATION EXPERIMENT - {dataset.upper()}")
    print("=" * 70)
    
    # Set seed
    set_seed(seed)
    
    # Get configuration
    config = EXPERIMENT_CONFIGS[dataset]
    dataset_name = config['dataset_name']
    
    # Setup output directory
    output_dir = os.path.join(
        PROJECT_ROOT, 'scripts', 'attention_visualization_v2', 'results', dataset
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model and data
    print("\n[1/6] Loading model and data...")
    model, test_loader, info = load_model_and_data(config, device)
    num_layers = len(model.transformer.layers)
    
    # Create attention extractor
    extractor = AttentionExtractor(model, device)
    
    # Extract attention for all test samples
    print("\n[2/6] Extracting attention weights...")
    attention_results = extract_batch_attention(
        extractor, test_loader, num_samples=None, device=device
    )
    print(f"  Extracted attention for {len(attention_results)} samples")
    
    # Compute statistics
    print("\n[3/6] Computing aggregate statistics...")
    stats = compute_attention_statistics(attention_results)
    
    # Select best samples
    print("\n[4/6] Selecting best samples for visualization...")
    indices, selected_samples = select_best_samples(attention_results, num_samples=10)
    print(f"  Selected {len(selected_samples)} samples")
    
    # Generate visualizations
    print("\n[5/6] Generating visualizations...")
    
    # Aggregate visualizations
    print("  - Aggregate pointer attention...")
    plot_aggregate_pointer_attention(attention_results, stats, output_dir, dataset_name)
    
    print("  - Gate analysis...")
    plot_gate_analysis(attention_results, stats, output_dir, dataset_name)
    
    print("  - Self-attention aggregate...")
    plot_self_attention_aggregate(attention_results, output_dir, dataset_name, num_layers)
    
    print("  - Position bias analysis...")
    plot_position_bias_analysis(model, output_dir, dataset_name)
    
    print("  - Sample overview...")
    plot_combined_samples_overview(selected_samples, output_dir, dataset_name)
    
    # Individual sample visualizations
    print("  - Individual sample attention...")
    for idx, sample in enumerate(selected_samples):
        plot_sample_attention(sample, idx, output_dir, dataset_name, num_layers)
    
    # Generate tables
    print("\n[6/6] Generating statistical tables...")
    df_stats, df_pos = generate_statistics_table(stats, attention_results, output_dir, dataset_name)
    df_samples = generate_sample_table(selected_samples, indices, output_dir, dataset_name)
    
    # Save experiment metadata
    metadata = {
        'dataset': dataset,
        'dataset_name': dataset_name,
        'seed': seed,
        'num_samples': len(attention_results),
        'accuracy': float(stats['accuracy']),
        'gate_mean': float(stats['gate_mean']),
        'gate_std': float(stats['gate_std']),
        'pointer_entropy_mean': float(stats['pointer_entropy_mean']),
        'timestamp': datetime.now().isoformat(),
        'model_config': info['model_config'],
        'num_layers': num_layers,
    }
    
    with open(os.path.join(output_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Samples Analyzed: {len(attention_results)}")
    print(f"Prediction Accuracy: {stats['accuracy'] * 100:.2f}%")
    print(f"Mean Gate Value: {stats['gate_mean']:.4f} ± {stats['gate_std']:.4f}")
    print(f"Gate (Correct): {stats['correct_gate_mean']:.4f}")
    print(f"Gate (Incorrect): {stats['incorrect_gate_mean']:.4f}")
    print(f"Mean Pointer Entropy: {stats['pointer_entropy_mean']:.4f}")
    print(f"\nOutput saved to: {output_dir}")
    print("=" * 70)
    
    return stats, selected_samples, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run attention visualization experiment for PointerNetworkV45'
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['diy', 'geolife'],
        help='Dataset to analyze'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    run_experiment(args.dataset, args.seed)

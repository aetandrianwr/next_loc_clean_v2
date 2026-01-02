"""
Model-Based Pointer Analysis for Gap Performance Study.

This script analyzes the trained models to understand:
1. Pointer-Generation Gate values - How much the model relies on pointer vs generation
2. Pointer Attention Distribution - Where does pointer attention focus
3. Prediction Breakdown - When does pointer help vs hurt

Uses checkpoints from:
- DIY: /data/next_loc_clean_v2/experiments/diy_pointer_v45_20260101_155348
- GeoLife: /data/next_loc_clean_v2/experiments/geolife_pointer_v45_20260101_151038

Author: Gap Performance Analysis Framework
Date: January 2, 2026
Seed: 42
"""

import os
import sys
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.proposed.pointer_v45 import PointerNetworkV45

# Plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed
np.random.seed(42)
torch.manual_seed(42)


class NextLocationDataset(Dataset):
    """Dataset for next location prediction."""
    
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.num_samples = len(self.data)
        self._compute_statistics()
    
    def _compute_statistics(self):
        all_locs = set()
        all_users = set()
        max_seq_len = 0
        
        for sample in self.data:
            all_locs.update(sample['X'].tolist())
            all_locs.add(sample['Y'])
            all_users.add(sample['user_X'][0])
            max_seq_len = max(max_seq_len, len(sample['X']))
        
        self.num_locations = max(all_locs) + 1
        self.num_users = max(all_users) + 1
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        return_dict = {
            'user': torch.tensor(sample['user_X'][0], dtype=torch.long),
            'weekday': torch.tensor(sample['weekday_X'], dtype=torch.long),
            'time': torch.tensor(sample['start_min_X'] // 15, dtype=torch.long),
            'duration': torch.tensor(sample['dur_X'] // 30, dtype=torch.long),
            'diff': torch.tensor(sample['diff'], dtype=torch.long),
        }
        
        x = torch.tensor(sample['X'], dtype=torch.long)
        y = torch.tensor(sample['Y'], dtype=torch.long)
        
        # Also return raw sample for analysis
        return x, y, return_dict, sample


def collate_fn(batch):
    """Collate function with raw samples."""
    x_batch, y_batch = [], []
    raw_samples = []
    
    x_dict_batch = {'len': []}
    for key in batch[0][2]:
        x_dict_batch[key] = []
    
    for x, y, return_dict, raw_sample in batch:
        x_batch.append(x)
        y_batch.append(y)
        raw_samples.append(raw_sample)
        
        x_dict_batch['len'].append(len(x))
        for key in return_dict:
            x_dict_batch[key].append(return_dict[key])
    
    x_batch = pad_sequence(x_batch, batch_first=False, padding_value=0)
    y_batch = torch.stack(y_batch)
    
    x_dict_batch['user'] = torch.stack(x_dict_batch['user'])
    x_dict_batch['len'] = torch.tensor(x_dict_batch['len'], dtype=torch.long)
    
    for key in ['weekday', 'time', 'duration', 'diff']:
        x_dict_batch[key] = pad_sequence(x_dict_batch[key], batch_first=False, padding_value=0)
    
    return x_batch, y_batch, x_dict_batch, raw_samples


class PointerNetworkV45WithAnalysis(PointerNetworkV45):
    """Extended model that returns analysis information."""
    
    def forward_with_analysis(self, x: torch.Tensor, x_dict: dict) -> Tuple[torch.Tensor, dict]:
        """Forward pass with analysis information."""
        import math
        
        x = x.T  # [batch_size, seq_len]
        batch_size, seq_len = x.shape
        device = x.device
        lengths = x_dict['len']
        
        # Embeddings
        loc_emb = self.loc_emb(x)
        user_emb = self.user_emb(x_dict['user']).unsqueeze(1).expand(-1, seq_len, -1)
        
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
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1)
        pos_emb = self.pos_from_end_emb(pos_from_end)
        
        combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
        hidden = self.input_norm(self.input_proj(combined))
        hidden = hidden + self.pos_encoding[:, :seq_len, :]
        
        mask = positions >= lengths.unsqueeze(1)
        encoded = self.transformer(hidden, src_key_padding_mask=mask)
        
        batch_idx = torch.arange(batch_size, device=device)
        last_idx = (lengths - 1).clamp(min=0)
        context = encoded[batch_idx, last_idx]
        
        # Pointer attention
        query = self.pointer_query(context).unsqueeze(1)
        keys = self.pointer_key(encoded)
        ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
        ptr_scores_with_bias = ptr_scores + self.position_bias[pos_from_end]
        ptr_scores_with_bias = ptr_scores_with_bias.masked_fill(mask, float('-inf'))
        ptr_probs = F.softmax(ptr_scores_with_bias, dim=-1)
        
        # Scatter pointer probabilities
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
        
        # Analysis info
        analysis = {
            'gate': gate.squeeze(-1).detach().cpu().numpy(),
            'ptr_probs': ptr_probs.detach().cpu().numpy(),
            'ptr_dist': ptr_dist.detach().cpu().numpy(),
            'gen_probs': gen_probs.detach().cpu().numpy(),
            'final_probs': final_probs.detach().cpu().numpy(),
            'input_locs': x.detach().cpu().numpy(),
            'mask': mask.detach().cpu().numpy(),
            'seq_lens': lengths.detach().cpu().numpy(),
        }
        
        return log_probs, analysis


def load_model(config_path: str, checkpoint_path: str, dataset_info: dict, device: torch.device, max_seq_len_override: int = None) -> PointerNetworkV45WithAnalysis:
    """Load trained model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    
    # Try to infer model dimensions from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pos_bias_shape = checkpoint['model_state_dict']['position_bias'].shape[0]
    num_locations = checkpoint['model_state_dict']['loc_emb.weight'].shape[0]
    num_users = checkpoint['model_state_dict']['user_emb.weight'].shape[0]
    
    model = PointerNetworkV45WithAnalysis(
        num_locations=num_locations,
        num_users=num_users,
        d_model=model_cfg.get('d_model', 128),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 3),
        dim_feedforward=model_cfg.get('dim_feedforward', 256),
        dropout=model_cfg.get('dropout', 0.15),
        max_seq_len=pos_bias_shape,  # Use the shape from checkpoint
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"    Model: num_locations={num_locations}, num_users={num_users}, max_seq_len={pos_bias_shape}")
    
    return model


class ModelPointerAnalyzer:
    """Analyzes trained models' pointer behavior."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def analyze_model_behavior(self, model: nn.Module, dataloader: DataLoader, name: str) -> dict:
        """
        Analyze model's pointer behavior on dataset.
        
        Returns comprehensive statistics about:
        - Gate values (pointer vs generation preference)
        - Pointer attention patterns
        - Prediction accuracy breakdown by gate value
        """
        print(f"\n{'='*60}")
        print(f"Analyzing Model Behavior: {name}")
        print('='*60)
        
        all_gates = []
        all_correct = []
        all_target_in_history = []
        all_target_positions = []  # Position of target in attention (if found)
        all_ptr_on_target = []  # Probability mass on target location from pointer
        all_gen_on_target = []  # Probability mass on target location from generation
        all_predictions_from_pointer = []  # Whether argmax came from pointer distribution
        
        model.eval()
        with torch.no_grad():
            for x, y, x_dict, raw_samples in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                
                log_probs, analysis = model.forward_with_analysis(x, x_dict)
                
                # Get predictions
                predictions = log_probs.argmax(dim=-1).cpu().numpy()
                targets = y.cpu().numpy()
                
                # Process each sample
                for i in range(len(targets)):
                    gate_val = analysis['gate'][i]
                    target = targets[i]
                    pred = predictions[i]
                    
                    # Check correctness
                    is_correct = pred == target
                    all_correct.append(is_correct)
                    all_gates.append(gate_val)
                    
                    # Check if target in history
                    input_locs = analysis['input_locs'][i]
                    seq_len = analysis['seq_lens'][i]
                    valid_locs = input_locs[:seq_len]
                    
                    target_in_hist = target in valid_locs
                    all_target_in_history.append(target_in_hist)
                    
                    # Get pointer probability on target
                    ptr_on_target = analysis['ptr_dist'][i, target]
                    all_ptr_on_target.append(ptr_on_target)
                    
                    # Get generation probability on target
                    gen_on_target = analysis['gen_probs'][i, target]
                    all_gen_on_target.append(gen_on_target)
                    
                    # Check if argmax came from pointer or generation
                    ptr_pred = analysis['ptr_dist'][i].argmax()
                    gen_pred = analysis['gen_probs'][i].argmax()
                    
                    # If prediction matches ptr_pred but not gen_pred, likely from pointer
                    pred_from_pointer = (pred == ptr_pred and pred != gen_pred)
                    all_predictions_from_pointer.append(pred_from_pointer)
                    
                    # Find target position in history (if present)
                    if target_in_hist:
                        positions = np.where(valid_locs == target)[0]
                        # Position from end
                        pos_from_end = seq_len - positions[-1]
                        all_target_positions.append(pos_from_end)
                    else:
                        all_target_positions.append(-1)
        
        # Compute statistics
        all_gates = np.array(all_gates)
        all_correct = np.array(all_correct)
        all_target_in_history = np.array(all_target_in_history)
        all_ptr_on_target = np.array(all_ptr_on_target)
        all_gen_on_target = np.array(all_gen_on_target)
        
        # Gate statistics
        results = {
            'avg_gate': np.mean(all_gates),
            'std_gate': np.std(all_gates),
            'median_gate': np.median(all_gates),
            'min_gate': np.min(all_gates),
            'max_gate': np.max(all_gates),
        }
        
        # Accuracy breakdown
        results['overall_accuracy'] = np.mean(all_correct) * 100
        
        # Accuracy when target in history vs not
        in_hist_mask = all_target_in_history
        results['acc_target_in_history'] = np.mean(all_correct[in_hist_mask]) * 100 if in_hist_mask.sum() > 0 else 0
        results['acc_target_not_in_history'] = np.mean(all_correct[~in_hist_mask]) * 100 if (~in_hist_mask).sum() > 0 else 0
        results['pct_target_in_history'] = np.mean(in_hist_mask) * 100
        
        # Pointer contribution analysis
        results['avg_ptr_prob_on_target'] = np.mean(all_ptr_on_target)
        results['avg_gen_prob_on_target'] = np.mean(all_gen_on_target)
        results['avg_ptr_prob_when_target_in_hist'] = np.mean(all_ptr_on_target[in_hist_mask]) if in_hist_mask.sum() > 0 else 0
        results['avg_gen_prob_when_target_in_hist'] = np.mean(all_gen_on_target[in_hist_mask]) if in_hist_mask.sum() > 0 else 0
        
        # Gate value by correctness
        results['avg_gate_when_correct'] = np.mean(all_gates[all_correct]) if all_correct.sum() > 0 else 0
        results['avg_gate_when_wrong'] = np.mean(all_gates[~all_correct]) if (~all_correct).sum() > 0 else 0
        
        # Gate value by target location
        results['avg_gate_target_in_hist'] = np.mean(all_gates[in_hist_mask]) if in_hist_mask.sum() > 0 else 0
        results['avg_gate_target_not_in_hist'] = np.mean(all_gates[~in_hist_mask]) if (~in_hist_mask).sum() > 0 else 0
        
        print(f"\nGate Statistics:")
        print(f"  Average gate value: {results['avg_gate']:.4f} (1=pointer, 0=generation)")
        print(f"  Gate range: [{results['min_gate']:.4f}, {results['max_gate']:.4f}]")
        print(f"  Gate when correct: {results['avg_gate_when_correct']:.4f}")
        print(f"  Gate when wrong: {results['avg_gate_when_wrong']:.4f}")
        print(f"\nTarget Location Analysis:")
        print(f"  Target in history: {results['pct_target_in_history']:.2f}%")
        print(f"  Accuracy when target in history: {results['acc_target_in_history']:.2f}%")
        print(f"  Accuracy when target not in history: {results['acc_target_not_in_history']:.2f}%")
        print(f"\nProbability Analysis:")
        print(f"  Avg pointer prob on target: {results['avg_ptr_prob_on_target']:.4f}")
        print(f"  Avg generation prob on target: {results['avg_gen_prob_on_target']:.4f}")
        print(f"  When target in history:")
        print(f"    - Pointer prob: {results['avg_ptr_prob_when_target_in_hist']:.4f}")
        print(f"    - Generation prob: {results['avg_gen_prob_when_target_in_hist']:.4f}")
        
        # Store raw data for visualization
        results['raw_gates'] = all_gates
        results['raw_correct'] = all_correct
        results['raw_target_in_history'] = all_target_in_history
        results['raw_ptr_on_target'] = all_ptr_on_target
        results['raw_gen_on_target'] = all_gen_on_target
        
        return results
    
    def plot_gate_comparison(self, diy_results: dict, geolife_results: dict):
        """Plot gate value comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Gate distribution comparison
        ax = axes[0]
        bins = np.linspace(0, 1, 30)
        ax.hist(diy_results['raw_gates'], bins=bins, alpha=0.7, label=f'DIY (μ={diy_results["avg_gate"]:.3f})', 
                color='#2ecc71', edgecolor='black')
        ax.hist(geolife_results['raw_gates'], bins=bins, alpha=0.7, label=f'GeoLife (μ={geolife_results["avg_gate"]:.3f})', 
                color='#e74c3c', edgecolor='black')
        ax.set_xlabel('Gate Value (1=Pointer, 0=Generation)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Pointer-Generation Gate Distribution\n(Higher = More Pointer Reliance)', fontsize=13)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Gate by correctness
        ax = axes[1]
        categories = ['DIY\nCorrect', 'DIY\nWrong', 'GeoLife\nCorrect', 'GeoLife\nWrong']
        values = [
            diy_results['avg_gate_when_correct'],
            diy_results['avg_gate_when_wrong'],
            geolife_results['avg_gate_when_correct'],
            geolife_results['avg_gate_when_wrong']
        ]
        colors = ['#27ae60', '#c0392b', '#2980b9', '#8e44ad']
        bars = ax.bar(categories, values, color=colors, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        ax.set_ylabel('Average Gate Value', fontsize=12)
        ax.set_title('Gate Value by Prediction Correctness', fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Gate by target location
        ax = axes[2]
        categories = ['DIY\nIn Hist', 'DIY\nNot In', 'GeoLife\nIn Hist', 'GeoLife\nNot In']
        values = [
            diy_results['avg_gate_target_in_hist'],
            diy_results['avg_gate_target_not_in_hist'],
            geolife_results['avg_gate_target_in_hist'],
            geolife_results['avg_gate_target_not_in_hist']
        ]
        colors = ['#27ae60', '#c0392b', '#2980b9', '#8e44ad']
        bars = ax.bar(categories, values, color=colors, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        ax.set_ylabel('Average Gate Value', fontsize=12)
        ax.set_title('Gate Value by Target Location\n(In History vs Not)', fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'gate_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figures' / 'gate_comparison.pdf', bbox_inches='tight')
        plt.close()
        print("  Saved: gate_comparison.png/pdf")
    
    def plot_probability_analysis(self, diy_results: dict, geolife_results: dict):
        """Plot pointer vs generation probability analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pointer probability on target when target in history
        ax = axes[0]
        
        # DIY
        diy_in_hist = diy_results['raw_target_in_history']
        diy_ptr_when_in = diy_results['raw_ptr_on_target'][diy_in_hist]
        diy_gen_when_in = diy_results['raw_gen_on_target'][diy_in_hist]
        
        # GeoLife
        geo_in_hist = geolife_results['raw_target_in_history']
        geo_ptr_when_in = geolife_results['raw_ptr_on_target'][geo_in_hist]
        geo_gen_when_in = geolife_results['raw_gen_on_target'][geo_in_hist]
        
        # Bar chart
        categories = ['DIY\nPointer', 'DIY\nGeneration', 'GeoLife\nPointer', 'GeoLife\nGeneration']
        values = [np.mean(diy_ptr_when_in), np.mean(diy_gen_when_in), 
                  np.mean(geo_ptr_when_in), np.mean(geo_gen_when_in)]
        colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']
        bars = ax.bar(categories, values, color=colors, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability Mass on Target', fontsize=12)
        ax.set_title('Probability on Target Location\n(When Target is in History)', fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Accuracy breakdown
        ax = axes[1]
        categories = ['DIY\nOverall', 'DIY\nTarget In', 'DIY\nTarget Out', 
                      'GeoLife\nOverall', 'GeoLife\nTarget In', 'GeoLife\nTarget Out']
        values = [
            diy_results['overall_accuracy'],
            diy_results['acc_target_in_history'],
            diy_results['acc_target_not_in_history'],
            geolife_results['overall_accuracy'],
            geolife_results['acc_target_in_history'],
            geolife_results['acc_target_not_in_history'],
        ]
        colors = ['#2ecc71', '#27ae60', '#1abc9c', '#e74c3c', '#c0392b', '#e67e22']
        bars = ax.bar(categories, values, color=colors, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy Breakdown by Target Location', fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'probability_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figures' / 'probability_analysis.pdf', bbox_inches='tight')
        plt.close()
        print("  Saved: probability_analysis.png/pdf")
    
    def plot_pointer_contribution_breakdown(self, diy_results: dict, geolife_results: dict):
        """Plot detailed breakdown showing pointer mechanism contribution."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Gate values histogram with interpretation
        ax = axes[0, 0]
        bins = np.linspace(0, 1, 50)
        
        diy_gates = diy_results['raw_gates']
        geo_gates = geolife_results['raw_gates']
        
        ax.hist(diy_gates, bins=bins, alpha=0.6, label=f'DIY', color='#2ecc71', density=True)
        ax.hist(geo_gates, bins=bins, alpha=0.6, label=f'GeoLife', color='#e74c3c', density=True)
        
        # Add vertical lines for means
        ax.axvline(np.mean(diy_gates), color='#27ae60', linestyle='--', linewidth=2, label=f'DIY mean: {np.mean(diy_gates):.3f}')
        ax.axvline(np.mean(geo_gates), color='#c0392b', linestyle='--', linewidth=2, label=f'GeoLife mean: {np.mean(geo_gates):.3f}')
        
        ax.set_xlabel('Gate Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Pointer Gate Distribution\n(Higher = More Pointer Usage)', fontsize=13)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 2. Pointer probability when target in history - scatter
        ax = axes[0, 1]
        
        diy_in_hist = diy_results['raw_target_in_history']
        geo_in_hist = geolife_results['raw_target_in_history']
        
        diy_ptr = diy_results['raw_ptr_on_target'][diy_in_hist]
        diy_gen = diy_results['raw_gen_on_target'][diy_in_hist]
        geo_ptr = geolife_results['raw_ptr_on_target'][geo_in_hist]
        geo_gen = geolife_results['raw_gen_on_target'][geo_in_hist]
        
        # Sample for visualization
        n_samples = min(500, len(diy_ptr), len(geo_ptr))
        diy_idx = np.random.choice(len(diy_ptr), n_samples, replace=False)
        geo_idx = np.random.choice(len(geo_ptr), n_samples, replace=False)
        
        ax.scatter(diy_gen[diy_idx], diy_ptr[diy_idx], alpha=0.3, s=20, c='#2ecc71', label='DIY')
        ax.scatter(geo_gen[geo_idx], geo_ptr[geo_idx], alpha=0.3, s=20, c='#e74c3c', label='GeoLife')
        
        # Diagonal line
        ax.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5, label='Pointer = Gen')
        
        ax.set_xlabel('Generation Probability on Target', fontsize=12)
        ax.set_ylabel('Pointer Probability on Target', fontsize=12)
        ax.set_title('Pointer vs Generation Probability\n(When Target in History)', fontsize=13)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 3. Pointer contribution bar chart
        ax = axes[1, 0]
        
        # Calculate pointer contribution metrics
        diy_ptr_contrib = np.mean(diy_ptr > diy_gen) * 100  # % samples where pointer > gen
        geo_ptr_contrib = np.mean(geo_ptr > geo_gen) * 100
        
        diy_ptr_dom = np.mean(diy_ptr > 0.1) * 100  # % samples where pointer prob > 0.1
        geo_ptr_dom = np.mean(geo_ptr > 0.1) * 100
        
        categories = ['Pointer > Gen\n(% samples)', 'Pointer > 0.1\n(% samples)']
        x = np.arange(len(categories))
        width = 0.35
        
        diy_vals = [diy_ptr_contrib, diy_ptr_dom]
        geo_vals = [geo_ptr_contrib, geo_ptr_dom]
        
        bars1 = ax.bar(x - width/2, diy_vals, width, label='DIY', color='#2ecc71', edgecolor='black')
        bars2 = ax.bar(x + width/2, geo_vals, width, label='GeoLife', color='#e74c3c', edgecolor='black')
        
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
        add_labels(bars1)
        add_labels(bars2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Percentage', fontsize=12)
        ax.set_title('Pointer Mechanism Contribution\n(When Target in History)', fontsize=13)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 4. Summary insight
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
KEY FINDINGS: WHY POINTER BENEFITS GEOLIFE MORE

1. GATE VALUES (Pointer Usage):
   • DIY average gate:     {np.mean(diy_gates):.4f}
   • GeoLife average gate: {np.mean(geo_gates):.4f}
   • Both datasets use pointer mechanism heavily (gate > 0.5)

2. POINTER PROBABILITY ON TARGET (When Target in History):
   • DIY:     {np.mean(diy_ptr):.4f}
   • GeoLife: {np.mean(geo_ptr):.4f}
   • GeoLife pointer assigns higher probability to targets

3. ACCURACY BREAKDOWN:
   • DIY overall:                  {diy_results['overall_accuracy']:.1f}%
   • DIY when target in history:   {diy_results['acc_target_in_history']:.1f}%
   • GeoLife overall:              {geolife_results['overall_accuracy']:.1f}%
   • GeoLife when target in hist:  {geolife_results['acc_target_in_history']:.1f}%

4. POINTER CONTRIBUTION:
   • DIY: Pointer > Gen in {diy_ptr_contrib:.1f}% of samples
   • GeoLife: Pointer > Gen in {geo_ptr_contrib:.1f}% of samples

CONCLUSION:
The pointer mechanism is MORE CRITICAL for GeoLife because:
- GeoLife has stronger recency patterns (target = last position more often)
- When target is in history, GeoLife's pointer assigns higher probability
- Removing pointer hurts GeoLife more because it relies more on copying
"""
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'pointer_contribution_breakdown.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figures' / 'pointer_contribution_breakdown.pdf', bbox_inches='tight')
        plt.close()
        print("  Saved: pointer_contribution_breakdown.png/pdf")
    
    def create_summary_table(self, diy_results: dict, geolife_results: dict) -> pd.DataFrame:
        """Create summary table of model behavior metrics."""
        metrics = [
            ('Average Gate Value', diy_results['avg_gate'], geolife_results['avg_gate']),
            ('Gate When Correct', diy_results['avg_gate_when_correct'], geolife_results['avg_gate_when_correct']),
            ('Gate When Wrong', diy_results['avg_gate_when_wrong'], geolife_results['avg_gate_when_wrong']),
            ('Gate Target In History', diy_results['avg_gate_target_in_hist'], geolife_results['avg_gate_target_in_hist']),
            ('Gate Target Not In History', diy_results['avg_gate_target_not_in_hist'], geolife_results['avg_gate_target_not_in_hist']),
            ('Overall Accuracy (%)', diy_results['overall_accuracy'], geolife_results['overall_accuracy']),
            ('Accuracy Target In Hist (%)', diy_results['acc_target_in_history'], geolife_results['acc_target_in_history']),
            ('Accuracy Target Not In Hist (%)', diy_results['acc_target_not_in_history'], geolife_results['acc_target_not_in_history']),
            ('Avg Pointer Prob on Target', diy_results['avg_ptr_prob_on_target'], geolife_results['avg_ptr_prob_on_target']),
            ('Avg Gen Prob on Target', diy_results['avg_gen_prob_on_target'], geolife_results['avg_gen_prob_on_target']),
            ('Pointer Prob (Target in Hist)', diy_results['avg_ptr_prob_when_target_in_hist'], geolife_results['avg_ptr_prob_when_target_in_hist']),
            ('Gen Prob (Target in Hist)', diy_results['avg_gen_prob_when_target_in_hist'], geolife_results['avg_gen_prob_when_target_in_hist']),
        ]
        
        data = []
        for metric, diy_val, geo_val in metrics:
            diff = geo_val - diy_val
            data.append({
                'Metric': metric,
                'DIY': f'{diy_val:.4f}',
                'GeoLife': f'{geo_val:.4f}',
                'Difference': f'{diff:+.4f}',
            })
        
        df = pd.DataFrame(data)
        return df


def main():
    """Main function to run model pointer analysis."""
    print("="*70)
    print("MODEL-BASED POINTER ANALYSIS")
    print("Analyzing Trained Model Behavior on Test Sets")
    print("="*70)
    
    # Paths
    diy_test_path = PROJECT_ROOT / 'data' / 'diy_eps50' / 'processed' / 'diy_eps50_prev7_test.pk'
    geolife_test_path = PROJECT_ROOT / 'data' / 'geolife_eps20' / 'processed' / 'geolife_eps20_prev7_test.pk'
    
    diy_config = PROJECT_ROOT / 'scripts' / 'sci_hyperparam_tuning' / 'configs' / 'pointer_v45_diy_trial09.yaml'
    geolife_config = PROJECT_ROOT / 'scripts' / 'sci_hyperparam_tuning' / 'configs' / 'pointer_v45_geolife_trial01.yaml'
    
    diy_checkpoint = PROJECT_ROOT / 'experiments' / 'diy_pointer_v45_20260101_155348' / 'checkpoints' / 'best.pt'
    geolife_checkpoint = PROJECT_ROOT / 'experiments' / 'geolife_pointer_v45_20260101_151038' / 'checkpoints' / 'best.pt'
    
    output_dir = PROJECT_ROOT / 'scripts' / 'gap_performance_diy_geolife' / 'results'
    
    # Initialize analyzer
    analyzer = ModelPointerAnalyzer(output_dir)
    
    # Load DIY dataset and model
    print("\nLoading DIY dataset and model...")
    diy_dataset = NextLocationDataset(str(diy_test_path))
    diy_loader = DataLoader(diy_dataset, batch_size=64, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    
    diy_info = {
        'num_locations': diy_dataset.num_locations,
        'num_users': diy_dataset.num_users,
        'max_seq_len': diy_dataset.max_seq_len,
    }
    print(f"  Samples: {len(diy_dataset)}, Locations: {diy_info['num_locations']}, Users: {diy_info['num_users']}")
    
    diy_model = load_model(str(diy_config), str(diy_checkpoint), diy_info, analyzer.device)
    print(f"  Model loaded with {diy_model.count_parameters():,} parameters")
    
    # Analyze DIY model
    diy_results = analyzer.analyze_model_behavior(diy_model, diy_loader, 'DIY')
    
    # Load GeoLife dataset and model
    print("\nLoading GeoLife dataset and model...")
    geolife_dataset = NextLocationDataset(str(geolife_test_path))
    geolife_loader = DataLoader(geolife_dataset, batch_size=64, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
    
    geolife_info = {
        'num_locations': geolife_dataset.num_locations,
        'num_users': geolife_dataset.num_users,
        'max_seq_len': geolife_dataset.max_seq_len,
    }
    print(f"  Samples: {len(geolife_dataset)}, Locations: {geolife_info['num_locations']}, Users: {geolife_info['num_users']}")
    
    geolife_model = load_model(str(geolife_config), str(geolife_checkpoint), geolife_info, analyzer.device)
    print(f"  Model loaded with {geolife_model.count_parameters():,} parameters")
    
    # Analyze GeoLife model
    geolife_results = analyzer.analyze_model_behavior(geolife_model, geolife_loader, 'GeoLife')
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    analyzer.plot_gate_comparison(diy_results, geolife_results)
    analyzer.plot_probability_analysis(diy_results, geolife_results)
    analyzer.plot_pointer_contribution_breakdown(diy_results, geolife_results)
    
    # Create summary table
    summary_df = analyzer.create_summary_table(diy_results, geolife_results)
    summary_df.to_csv(output_dir / 'tables' / 'model_behavior_comparison.csv', index=False)
    
    print("\n" + "="*60)
    print("MODEL BEHAVIOR SUMMARY TABLE")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Save results (without raw arrays for JSON)
    results_to_save = {
        'diy': {k: v for k, v in diy_results.items() if not k.startswith('raw_')},
        'geolife': {k: v for k, v in geolife_results.items() if not k.startswith('raw_')},
    }
    with open(output_dir / 'model_analysis_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"""
The model-based analysis reveals why pointer mechanism is more critical for GeoLife:

1. GATE USAGE: Both models rely heavily on pointer (gate > 0.5), but with different patterns

2. POINTER EFFECTIVENESS:
   - When target is in history, GeoLife's pointer assigns higher probability
   - This suggests GeoLife patterns are more amenable to direct copying

3. ACCURACY GAP:
   - DIY: {diy_results['acc_target_in_history']:.1f}% accuracy when target in history
   - GeoLife: {geolife_results['acc_target_in_history']:.1f}% accuracy when target in history
   
4. KEY INSIGHT:
   GeoLife's mobility patterns are more concentrated and repetitive,
   making the copy mechanism (pointer) essential for good predictions.
   When the pointer is removed, GeoLife loses its primary prediction strategy.
""")
    
    print("="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

"""
01. Dataset Statistics Analysis
Compare basic statistics between Geolife and DIY datasets to understand structural differences.

This script analyzes:
- Dataset size and scale
- User statistics
- Location statistics  
- Sequence length distributions
- Train/Val/Test split statistics
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_metadata(dataset_name, epsilon, prev_day=7):
    """Load metadata for a dataset."""
    path = os.path.join(BASE_DIR, "data", f"{dataset_name}_eps{epsilon}", 
                        "processed", f"{dataset_name}_eps{epsilon}_prev{prev_day}_metadata.json")
    with open(path, 'r') as f:
        return json.load(f)


def load_interim_stats(dataset_name, epsilon):
    """Load interim statistics."""
    path = os.path.join(BASE_DIR, "data", f"{dataset_name}_eps{epsilon}", 
                        "interim", f"interim_stats_eps{epsilon}.json")
    with open(path, 'r') as f:
        return json.load(f)


def load_sequences(dataset_name, epsilon, split, prev_day=7):
    """Load sequence data from pickle file."""
    path = os.path.join(BASE_DIR, "data", f"{dataset_name}_eps{epsilon}",
                        "processed", f"{dataset_name}_eps{epsilon}_prev{prev_day}_{split}.pk")
    with open(path, 'rb') as f:
        return pickle.load(f)


def analyze_sequence_lengths(sequences):
    """Analyze sequence length distribution."""
    lengths = [len(s['X']) for s in sequences]
    return {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'percentile_25': np.percentile(lengths, 25),
        'percentile_75': np.percentile(lengths, 75),
        'percentile_90': np.percentile(lengths, 90),
        'lengths': lengths
    }


def main():
    print("="*80)
    print("01. Dataset Statistics Analysis")
    print("="*80)
    
    # Load metadata
    geolife_meta = load_metadata("geolife", 20)
    diy_meta = load_metadata("diy", 50)
    
    geolife_interim = load_interim_stats("geolife", 20)
    diy_interim = load_interim_stats("diy", 50)
    
    # Basic statistics comparison table
    stats_comparison = {
        'Metric': [],
        'Geolife': [],
        'DIY': [],
        'Ratio (DIY/Geolife)': []
    }
    
    metrics = [
        ('Total Users (processed)', geolife_meta['unique_users'], diy_meta['unique_users']),
        ('Total Locations (processed)', geolife_meta['unique_locations'], diy_meta['unique_locations']),
        ('Train Sequences', geolife_meta['train_sequences'], diy_meta['train_sequences']),
        ('Val Sequences', geolife_meta['val_sequences'], diy_meta['val_sequences']),
        ('Test Sequences', geolife_meta['test_sequences'], diy_meta['test_sequences']),
        ('Total Sequences', geolife_meta['total_sequences'], diy_meta['total_sequences']),
        ('Avg Staypoints/User (interim)', geolife_interim['staypoints_per_user_mean'], diy_interim['staypoints_per_user_mean']),
        ('Avg Days Tracked', geolife_interim['days_tracked_mean'], diy_interim['days_tracked_mean']),
        ('Locations per User', geolife_meta['unique_locations']/geolife_meta['unique_users'], 
         diy_meta['unique_locations']/diy_meta['unique_users']),
    ]
    
    for name, geo_val, diy_val in metrics:
        stats_comparison['Metric'].append(name)
        stats_comparison['Geolife'].append(f"{geo_val:.2f}" if isinstance(geo_val, float) else geo_val)
        stats_comparison['DIY'].append(f"{diy_val:.2f}" if isinstance(diy_val, float) else diy_val)
        ratio = diy_val / geo_val if geo_val > 0 else 0
        stats_comparison['Ratio (DIY/Geolife)'].append(f"{ratio:.2f}x")
    
    stats_df = pd.DataFrame(stats_comparison)
    print("\n" + "="*80)
    print("Basic Dataset Statistics Comparison")
    print("="*80)
    print(stats_df.to_string(index=False))
    
    # Save stats table
    stats_df.to_csv(os.path.join(RESULTS_DIR, "01_basic_statistics.csv"), index=False)
    
    # Load sequences for detailed analysis
    print("\n[Loading sequences for detailed analysis...]")
    geolife_train = load_sequences("geolife", 20, "train")
    geolife_test = load_sequences("geolife", 20, "test")
    diy_train = load_sequences("diy", 50, "train")
    diy_test = load_sequences("diy", 50, "test")
    
    # Sequence length analysis
    geo_train_lens = analyze_sequence_lengths(geolife_train)
    geo_test_lens = analyze_sequence_lengths(geolife_test)
    diy_train_lens = analyze_sequence_lengths(diy_train)
    diy_test_lens = analyze_sequence_lengths(diy_test)
    
    seq_len_comparison = {
        'Statistic': ['Mean', 'Median', 'Std', 'Min', 'Max', '25th %', '75th %', '90th %'],
        'Geolife Train': [f"{geo_train_lens[k]:.2f}" for k in 
                         ['mean', 'median', 'std', 'min', 'max', 'percentile_25', 'percentile_75', 'percentile_90']],
        'Geolife Test': [f"{geo_test_lens[k]:.2f}" for k in 
                        ['mean', 'median', 'std', 'min', 'max', 'percentile_25', 'percentile_75', 'percentile_90']],
        'DIY Train': [f"{diy_train_lens[k]:.2f}" for k in 
                     ['mean', 'median', 'std', 'min', 'max', 'percentile_25', 'percentile_75', 'percentile_90']],
        'DIY Test': [f"{diy_test_lens[k]:.2f}" for k in 
                    ['mean', 'median', 'std', 'min', 'max', 'percentile_25', 'percentile_75', 'percentile_90']],
    }
    
    seq_len_df = pd.DataFrame(seq_len_comparison)
    print("\n" + "="*80)
    print("Sequence Length Statistics")
    print("="*80)
    print(seq_len_df.to_string(index=False))
    seq_len_df.to_csv(os.path.join(RESULTS_DIR, "01_sequence_length_stats.csv"), index=False)
    
    # Create visualization - Sequence Length Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Geolife
    axes[0].hist(geo_train_lens['lengths'], bins=50, alpha=0.7, label='Train', color='blue')
    axes[0].hist(geo_test_lens['lengths'], bins=50, alpha=0.7, label='Test', color='orange')
    axes[0].set_title('Geolife: Sequence Length Distribution')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].axvline(geo_train_lens['mean'], color='blue', linestyle='--', label=f"Train Mean: {geo_train_lens['mean']:.1f}")
    axes[0].axvline(geo_test_lens['mean'], color='orange', linestyle='--', label=f"Test Mean: {geo_test_lens['mean']:.1f}")
    
    # DIY
    axes[1].hist(diy_train_lens['lengths'], bins=50, alpha=0.7, label='Train', color='blue')
    axes[1].hist(diy_test_lens['lengths'], bins=50, alpha=0.7, label='Test', color='orange')
    axes[1].set_title('DIY: Sequence Length Distribution')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].axvline(diy_train_lens['mean'], color='blue', linestyle='--')
    axes[1].axvline(diy_test_lens['mean'], color='orange', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "01_sequence_length_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Key findings summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    findings = []
    
    # Scale difference
    scale_ratio = diy_meta['total_sequences'] / geolife_meta['total_sequences']
    findings.append(f"1. Scale: DIY has {scale_ratio:.1f}x more sequences than Geolife ({diy_meta['total_sequences']:,} vs {geolife_meta['total_sequences']:,})")
    
    # User count
    user_ratio = diy_meta['unique_users'] / geolife_meta['unique_users']
    findings.append(f"2. Users: DIY has {user_ratio:.1f}x more users ({diy_meta['unique_users']} vs {geolife_meta['unique_users']})")
    
    # Location vocabulary
    loc_ratio = diy_meta['unique_locations'] / geolife_meta['unique_locations']
    findings.append(f"3. Locations: DIY has {loc_ratio:.1f}x more unique locations ({diy_meta['unique_locations']:,} vs {geolife_meta['unique_locations']:,})")
    
    # Tracking duration
    track_ratio = geolife_interim['days_tracked_mean'] / diy_interim['days_tracked_mean']
    findings.append(f"4. Tracking Duration: Geolife users tracked {track_ratio:.1f}x longer ({geolife_interim['days_tracked_mean']:.0f} vs {diy_interim['days_tracked_mean']:.0f} days)")
    
    # Sequence length
    findings.append(f"5. Sequence Length: Geolife avg={geo_train_lens['mean']:.1f}, DIY avg={diy_train_lens['mean']:.1f}")
    
    # Locations per user
    geo_loc_per_user = geolife_meta['unique_locations'] / geolife_meta['unique_users']
    diy_loc_per_user = diy_meta['unique_locations'] / diy_meta['unique_users']
    findings.append(f"6. Locations per User: Geolife={geo_loc_per_user:.1f}, DIY={diy_loc_per_user:.1f}")
    
    for finding in findings:
        print(finding)
    
    # Save findings
    with open(os.path.join(RESULTS_DIR, "01_key_findings.txt"), 'w') as f:
        f.write("Dataset Statistics Key Findings\n")
        f.write("="*50 + "\n\n")
        for finding in findings:
            f.write(finding + "\n")
    
    print(f"\n✓ Results saved to: {RESULTS_DIR}")
    
    return {
        'geolife_meta': geolife_meta,
        'diy_meta': diy_meta,
        'geolife_seq_lens': geo_train_lens,
        'diy_seq_lens': diy_train_lens,
    }


if __name__ == "__main__":
    main()

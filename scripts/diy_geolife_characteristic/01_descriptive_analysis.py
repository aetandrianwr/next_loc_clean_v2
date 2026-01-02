"""
Descriptive Analysis of DIY and GeoLife Test Datasets.

This script performs comprehensive descriptive statistics to understand the 
fundamental differences between DIY and GeoLife datasets that may explain
the differential impact of the pointer mechanism (46.7% vs 8.3% drop).

Key Analysis:
1. Target-in-History Rate (Copy Applicability)
2. Location Repetition Patterns
3. Vocabulary Utilization
4. User Mobility Diversity
5. Sequence Characteristics
6. Temporal Patterns

Output:
- Detailed statistics tables
- Visualizations comparing both datasets
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for Nature Journal quality plots
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DIY_TEST_PATH = BASE_DIR / "data/diy_eps50/processed/diy_eps50_prev7_test.pk"
GEOLIFE_TEST_PATH = BASE_DIR / "data/geolife_eps20/processed/geolife_eps20_prev7_test.pk"
DIY_TRAIN_PATH = BASE_DIR / "data/diy_eps50/processed/diy_eps50_prev7_train.pk"
GEOLIFE_TRAIN_PATH = BASE_DIR / "data/geolife_eps20/processed/geolife_eps20_prev7_train.pk"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset(path):
    """Load pickle dataset."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def analyze_target_in_history(data, name):
    """
    Analyze target-in-history rate - key metric for pointer mechanism applicability.
    
    This measures how often the target location appears in the input sequence.
    If target is in history, pointer mechanism can directly copy it.
    """
    total = len(data)
    target_in_history = 0
    target_in_last_k = {1: 0, 3: 0, 5: 0, 7: 0}
    position_of_target = []  # Position from end where target appears
    target_frequency_in_history = []  # How many times target appears in history
    
    for sample in data:
        X = sample['X']
        Y = sample['Y']
        
        if Y in X:
            target_in_history += 1
            # Find positions where Y appears (from end)
            positions = [len(X) - i for i, loc in enumerate(X) if loc == Y]
            position_of_target.extend(positions)
            target_frequency_in_history.append(sum(1 for loc in X if loc == Y))
            
            # Check last k positions
            for k in target_in_last_k:
                if Y in X[-k:]:
                    target_in_last_k[k] += 1
        else:
            target_frequency_in_history.append(0)
    
    rate = target_in_history / total * 100
    
    results = {
        'dataset': name,
        'total_samples': total,
        'target_in_history': target_in_history,
        'target_in_history_rate': rate,
        'target_in_last_1': target_in_last_k[1] / total * 100,
        'target_in_last_3': target_in_last_k[3] / total * 100,
        'target_in_last_5': target_in_last_k[5] / total * 100,
        'target_in_last_7': target_in_last_k[7] / total * 100,
        'position_of_target': position_of_target,
        'target_frequency_in_history': target_frequency_in_history,
    }
    
    return results


def analyze_repetition_patterns(data, name):
    """
    Analyze location repetition patterns in sequences.
    
    Higher repetition = more benefit from pointer mechanism.
    """
    unique_ratios = []  # unique_locs / seq_len
    repetition_rates = []  # (total - unique) / total
    consecutive_repetitions = []  # Number of consecutive same locations
    
    for sample in data:
        X = sample['X']
        seq_len = len(X)
        unique_locs = len(set(X))
        
        unique_ratios.append(unique_locs / seq_len)
        repetition_rates.append((seq_len - unique_locs) / seq_len)
        
        # Count consecutive repetitions
        consec_count = sum(1 for i in range(1, len(X)) if X[i] == X[i-1])
        consecutive_repetitions.append(consec_count / max(1, seq_len - 1))
    
    results = {
        'dataset': name,
        'avg_unique_ratio': np.mean(unique_ratios),
        'std_unique_ratio': np.std(unique_ratios),
        'avg_repetition_rate': np.mean(repetition_rates),
        'std_repetition_rate': np.std(repetition_rates),
        'avg_consecutive_repetition': np.mean(consecutive_repetitions),
        'unique_ratios': unique_ratios,
        'repetition_rates': repetition_rates,
    }
    
    return results


def analyze_vocabulary_utilization(data, train_data, name):
    """
    Analyze vocabulary utilization patterns.
    
    - How many unique locations are used?
    - How concentrated is the location distribution?
    """
    all_locs_seq = []
    all_targets = []
    
    for sample in data:
        all_locs_seq.extend(sample['X'].tolist())
        all_targets.append(sample['Y'])
    
    # Train vocabulary
    train_locs = []
    for sample in train_data:
        train_locs.extend(sample['X'].tolist())
        train_locs.append(sample['Y'])
    
    seq_counter = Counter(all_locs_seq)
    target_counter = Counter(all_targets)
    train_counter = Counter(train_locs)
    
    # Calculate concentration metrics
    total_seq = sum(seq_counter.values())
    total_target = sum(target_counter.values())
    
    # Top-k coverage
    seq_sorted = sorted(seq_counter.values(), reverse=True)
    target_sorted = sorted(target_counter.values(), reverse=True)
    
    top_10_seq_coverage = sum(seq_sorted[:10]) / total_seq * 100
    top_50_seq_coverage = sum(seq_sorted[:50]) / total_seq * 100
    top_10_target_coverage = sum(target_sorted[:10]) / total_target * 100
    top_50_target_coverage = sum(target_sorted[:50]) / total_target * 100
    
    results = {
        'dataset': name,
        'unique_locs_in_sequences': len(seq_counter),
        'unique_targets': len(target_counter),
        'train_vocabulary_size': len(train_counter),
        'top_10_seq_coverage': top_10_seq_coverage,
        'top_50_seq_coverage': top_50_seq_coverage,
        'top_10_target_coverage': top_10_target_coverage,
        'top_50_target_coverage': top_50_target_coverage,
        'seq_distribution_entropy': -sum((c/total_seq) * np.log(c/total_seq + 1e-10) for c in seq_counter.values()),
        'target_distribution_entropy': -sum((c/total_target) * np.log(c/total_target + 1e-10) for c in target_counter.values()),
    }
    
    return results


def analyze_user_patterns(data, name):
    """
    Analyze user-level mobility patterns.
    
    - How diverse are individual user patterns?
    - Do users revisit the same locations?
    """
    user_data = defaultdict(list)
    
    for sample in data:
        user_id = sample['user_X'][0]
        user_data[user_id].append({
            'X': sample['X'],
            'Y': sample['Y'],
        })
    
    user_stats = []
    
    for user_id, samples in user_data.items():
        all_locs = []
        all_targets = []
        for s in samples:
            all_locs.extend(s['X'].tolist())
            all_targets.append(s['Y'])
        
        unique_locs = len(set(all_locs))
        total_locs = len(all_locs)
        target_revisit_rate = sum(1 for t in all_targets if t in all_locs) / len(all_targets)
        
        user_stats.append({
            'user_id': user_id,
            'num_samples': len(samples),
            'unique_locations': unique_locs,
            'total_visits': total_locs,
            'location_diversity': unique_locs / total_locs if total_locs > 0 else 0,
            'target_revisit_rate': target_revisit_rate,
        })
    
    df_users = pd.DataFrame(user_stats)
    
    results = {
        'dataset': name,
        'num_users': len(user_data),
        'avg_samples_per_user': df_users['num_samples'].mean(),
        'std_samples_per_user': df_users['num_samples'].std(),
        'avg_unique_locs_per_user': df_users['unique_locations'].mean(),
        'avg_location_diversity': df_users['location_diversity'].mean(),
        'avg_target_revisit_rate': df_users['target_revisit_rate'].mean(),
        'user_stats_df': df_users,
    }
    
    return results


def analyze_sequence_characteristics(data, name):
    """
    Analyze sequence length and structure.
    """
    seq_lengths = [len(sample['X']) for sample in data]
    
    results = {
        'dataset': name,
        'avg_seq_length': np.mean(seq_lengths),
        'std_seq_length': np.std(seq_lengths),
        'min_seq_length': np.min(seq_lengths),
        'max_seq_length': np.max(seq_lengths),
        'median_seq_length': np.median(seq_lengths),
        'seq_lengths': seq_lengths,
    }
    
    return results


def analyze_temporal_patterns(data, name):
    """
    Analyze temporal patterns in mobility data.
    """
    time_of_day = []
    weekdays = []
    durations = []
    recencies = []
    
    for sample in data:
        time_of_day.extend(sample['start_min_X'].tolist())
        weekdays.extend(sample['weekday_X'].tolist())
        durations.extend(sample['dur_X'].tolist())
        recencies.extend(sample['diff'].tolist())
    
    # Time buckets (morning, afternoon, evening, night)
    time_buckets = {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0}
    for t in time_of_day:
        hour = t // 60
        if 6 <= hour < 12:
            time_buckets['morning'] += 1
        elif 12 <= hour < 18:
            time_buckets['afternoon'] += 1
        elif 18 <= hour < 22:
            time_buckets['evening'] += 1
        else:
            time_buckets['night'] += 1
    
    total_time = sum(time_buckets.values())
    
    results = {
        'dataset': name,
        'morning_ratio': time_buckets['morning'] / total_time * 100,
        'afternoon_ratio': time_buckets['afternoon'] / total_time * 100,
        'evening_ratio': time_buckets['evening'] / total_time * 100,
        'night_ratio': time_buckets['night'] / total_time * 100,
        'avg_duration_min': np.mean(durations),
        'avg_recency_days': np.mean(recencies),
        'weekday_distribution': Counter(weekdays),
    }
    
    return results


def create_visualizations(diy_results, geolife_results, output_dir):
    """Create comprehensive visualizations."""
    
    # Figure 1: Target-in-History Rate Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1a. Overall target-in-history rate
    datasets = ['DIY', 'GeoLife']
    rates = [diy_results['target_in_history']['target_in_history_rate'],
             geolife_results['target_in_history']['target_in_history_rate']]
    colors = ['#3498db', '#e74c3c']
    
    axes[0].bar(datasets, rates, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Target-in-History Rate (%)')
    axes[0].set_title('(a) Copy Applicability Rate')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(rates):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 1b. Target position distribution
    diy_pos = diy_results['target_in_history']['position_of_target']
    geolife_pos = geolife_results['target_in_history']['position_of_target']
    
    axes[1].hist(diy_pos, bins=30, alpha=0.7, label='DIY', color='#3498db', density=True)
    axes[1].hist(geolife_pos, bins=30, alpha=0.7, label='GeoLife', color='#e74c3c', density=True)
    axes[1].set_xlabel('Position from End')
    axes[1].set_ylabel('Density')
    axes[1].set_title('(b) Target Position Distribution')
    axes[1].legend()
    
    # 1c. Target-in-last-k comparison
    k_values = [1, 3, 5, 7]
    diy_last_k = [diy_results['target_in_history'][f'target_in_last_{k}'] for k in k_values]
    geolife_last_k = [geolife_results['target_in_history'][f'target_in_last_{k}'] for k in k_values]
    
    x = np.arange(len(k_values))
    width = 0.35
    axes[2].bar(x - width/2, diy_last_k, width, label='DIY', color='#3498db', edgecolor='black')
    axes[2].bar(x + width/2, geolife_last_k, width, label='GeoLife', color='#e74c3c', edgecolor='black')
    axes[2].set_xlabel('Last k positions')
    axes[2].set_ylabel('Rate (%)')
    axes[2].set_title('(c) Target in Recent Positions')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'Last {k}' for k in k_values])
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_target_in_history.png')
    plt.savefig(output_dir / 'fig1_target_in_history.pdf')
    plt.close()
    
    # Figure 2: Repetition Patterns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2a. Repetition rate distribution
    diy_rep = diy_results['repetition']['repetition_rates']
    geolife_rep = geolife_results['repetition']['repetition_rates']
    
    axes[0].hist(diy_rep, bins=30, alpha=0.7, label='DIY', color='#3498db', density=True)
    axes[0].hist(geolife_rep, bins=30, alpha=0.7, label='GeoLife', color='#e74c3c', density=True)
    axes[0].set_xlabel('Repetition Rate')
    axes[0].set_ylabel('Density')
    axes[0].set_title('(a) Location Repetition Rate Distribution')
    axes[0].legend()
    axes[0].axvline(np.mean(diy_rep), color='#3498db', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(geolife_rep), color='#e74c3c', linestyle='--', linewidth=2)
    
    # 2b. Average repetition comparison
    metrics = ['Avg Repetition\nRate', 'Avg Consecutive\nRepetition']
    diy_vals = [diy_results['repetition']['avg_repetition_rate'],
                diy_results['repetition']['avg_consecutive_repetition']]
    geolife_vals = [geolife_results['repetition']['avg_repetition_rate'],
                   geolife_results['repetition']['avg_consecutive_repetition']]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[1].bar(x - width/2, diy_vals, width, label='DIY', color='#3498db', edgecolor='black')
    axes[1].bar(x + width/2, geolife_vals, width, label='GeoLife', color='#e74c3c', edgecolor='black')
    axes[1].set_ylabel('Rate')
    axes[1].set_title('(b) Average Repetition Metrics')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    
    # 2c. Unique ratio distribution
    diy_unique = diy_results['repetition']['unique_ratios']
    geolife_unique = geolife_results['repetition']['unique_ratios']
    
    axes[2].boxplot([diy_unique, geolife_unique], labels=['DIY', 'GeoLife'])
    axes[2].set_ylabel('Unique Location Ratio')
    axes[2].set_title('(c) Sequence Diversity (Unique/Total)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_repetition_patterns.png')
    plt.savefig(output_dir / 'fig2_repetition_patterns.pdf')
    plt.close()
    
    # Figure 3: Vocabulary and User Patterns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 3a. Vocabulary metrics
    metrics = ['Unique\nLocations', 'Top-10\nCoverage (%)', 'Top-50\nCoverage (%)']
    diy_vocab = [diy_results['vocabulary']['unique_locs_in_sequences'],
                 diy_results['vocabulary']['top_10_seq_coverage'],
                 diy_results['vocabulary']['top_50_seq_coverage']]
    geolife_vocab = [geolife_results['vocabulary']['unique_locs_in_sequences'],
                    geolife_results['vocabulary']['top_10_seq_coverage'],
                    geolife_results['vocabulary']['top_50_seq_coverage']]
    
    # Normalize for visualization
    x = np.arange(3)
    width = 0.35
    ax1 = axes[0]
    ax1.bar(x[0] - width/2, diy_vocab[0], width, label='DIY', color='#3498db', edgecolor='black')
    ax1.bar(x[0] + width/2, geolife_vocab[0], width, label='GeoLife', color='#e74c3c', edgecolor='black')
    ax1.set_ylabel('Count / Percentage')
    ax1.set_title('(a) Vocabulary Size')
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Unique Locations'])
    ax1.legend()
    
    # 3b. Coverage comparison
    x = np.arange(2)
    axes[1].bar(x - width/2, diy_vocab[1:], width, label='DIY', color='#3498db', edgecolor='black')
    axes[1].bar(x + width/2, geolife_vocab[1:], width, label='GeoLife', color='#e74c3c', edgecolor='black')
    axes[1].set_ylabel('Coverage (%)')
    axes[1].set_title('(b) Location Concentration')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Top-10', 'Top-50'])
    axes[1].legend()
    
    # 3c. User target revisit rate
    diy_user_revisit = diy_results['user_patterns']['user_stats_df']['target_revisit_rate']
    geolife_user_revisit = geolife_results['user_patterns']['user_stats_df']['target_revisit_rate']
    
    axes[2].boxplot([diy_user_revisit, geolife_user_revisit], labels=['DIY', 'GeoLife'])
    axes[2].set_ylabel('Target Revisit Rate')
    axes[2].set_title('(c) User Target Revisit Rate')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_vocabulary_user_patterns.png')
    plt.savefig(output_dir / 'fig3_vocabulary_user_patterns.pdf')
    plt.close()
    
    # Figure 4: Summary comparison radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Target-in-History\nRate', 'Repetition\nRate', 'Top-10\nCoverage',
                  'User Revisit\nRate', 'Avg Seq\nLength', 'Location\nConcentration']
    
    # Normalize values to 0-1 range for comparison
    diy_values = [
        diy_results['target_in_history']['target_in_history_rate'] / 100,
        diy_results['repetition']['avg_repetition_rate'],
        diy_results['vocabulary']['top_10_seq_coverage'] / 100,
        diy_results['user_patterns']['avg_target_revisit_rate'],
        min(diy_results['sequence']['avg_seq_length'] / 50, 1),  # Normalize by 50
        1 - diy_results['vocabulary']['unique_locs_in_sequences'] / 500,  # Inverse of diversity
    ]
    
    geolife_values = [
        geolife_results['target_in_history']['target_in_history_rate'] / 100,
        geolife_results['repetition']['avg_repetition_rate'],
        geolife_results['vocabulary']['top_10_seq_coverage'] / 100,
        geolife_results['user_patterns']['avg_target_revisit_rate'],
        min(geolife_results['sequence']['avg_seq_length'] / 50, 1),
        1 - geolife_results['vocabulary']['unique_locs_in_sequences'] / 500,
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    diy_values += diy_values[:1]
    geolife_values += geolife_values[:1]
    angles += angles[:1]
    
    ax.plot(angles, diy_values, 'o-', linewidth=2, label='DIY', color='#3498db')
    ax.fill(angles, diy_values, alpha=0.25, color='#3498db')
    ax.plot(angles, geolife_values, 'o-', linewidth=2, label='GeoLife', color='#e74c3c')
    ax.fill(angles, geolife_values, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Dataset Characteristics Comparison\n(Higher = More Pointer-Friendly)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_radar_comparison.png')
    plt.savefig(output_dir / 'fig4_radar_comparison.pdf')
    plt.close()


def save_results_to_table(diy_results, geolife_results, output_dir):
    """Save comprehensive results to CSV and markdown tables."""
    
    # Create summary table
    summary_data = []
    
    # Target-in-History Analysis
    summary_data.append({
        'Category': 'Copy Applicability',
        'Metric': 'Target-in-History Rate (%)',
        'DIY': f"{diy_results['target_in_history']['target_in_history_rate']:.2f}",
        'GeoLife': f"{geolife_results['target_in_history']['target_in_history_rate']:.2f}",
        'Difference': f"{geolife_results['target_in_history']['target_in_history_rate'] - diy_results['target_in_history']['target_in_history_rate']:.2f}",
    })
    summary_data.append({
        'Category': 'Copy Applicability',
        'Metric': 'Target in Last-1 (%)',
        'DIY': f"{diy_results['target_in_history']['target_in_last_1']:.2f}",
        'GeoLife': f"{geolife_results['target_in_history']['target_in_last_1']:.2f}",
        'Difference': f"{geolife_results['target_in_history']['target_in_last_1'] - diy_results['target_in_history']['target_in_last_1']:.2f}",
    })
    summary_data.append({
        'Category': 'Copy Applicability',
        'Metric': 'Target in Last-3 (%)',
        'DIY': f"{diy_results['target_in_history']['target_in_last_3']:.2f}",
        'GeoLife': f"{geolife_results['target_in_history']['target_in_last_3']:.2f}",
        'Difference': f"{geolife_results['target_in_history']['target_in_last_3'] - diy_results['target_in_history']['target_in_last_3']:.2f}",
    })
    
    # Repetition Analysis
    summary_data.append({
        'Category': 'Repetition Patterns',
        'Metric': 'Avg Repetition Rate',
        'DIY': f"{diy_results['repetition']['avg_repetition_rate']:.4f}",
        'GeoLife': f"{geolife_results['repetition']['avg_repetition_rate']:.4f}",
        'Difference': f"{geolife_results['repetition']['avg_repetition_rate'] - diy_results['repetition']['avg_repetition_rate']:.4f}",
    })
    summary_data.append({
        'Category': 'Repetition Patterns',
        'Metric': 'Avg Consecutive Repetition',
        'DIY': f"{diy_results['repetition']['avg_consecutive_repetition']:.4f}",
        'GeoLife': f"{geolife_results['repetition']['avg_consecutive_repetition']:.4f}",
        'Difference': f"{geolife_results['repetition']['avg_consecutive_repetition'] - diy_results['repetition']['avg_consecutive_repetition']:.4f}",
    })
    
    # Vocabulary Analysis
    summary_data.append({
        'Category': 'Vocabulary',
        'Metric': 'Unique Locations in Test',
        'DIY': f"{diy_results['vocabulary']['unique_locs_in_sequences']}",
        'GeoLife': f"{geolife_results['vocabulary']['unique_locs_in_sequences']}",
        'Difference': f"{geolife_results['vocabulary']['unique_locs_in_sequences'] - diy_results['vocabulary']['unique_locs_in_sequences']}",
    })
    summary_data.append({
        'Category': 'Vocabulary',
        'Metric': 'Top-10 Location Coverage (%)',
        'DIY': f"{diy_results['vocabulary']['top_10_seq_coverage']:.2f}",
        'GeoLife': f"{geolife_results['vocabulary']['top_10_seq_coverage']:.2f}",
        'Difference': f"{geolife_results['vocabulary']['top_10_seq_coverage'] - diy_results['vocabulary']['top_10_seq_coverage']:.2f}",
    })
    summary_data.append({
        'Category': 'Vocabulary',
        'Metric': 'Target Distribution Entropy',
        'DIY': f"{diy_results['vocabulary']['target_distribution_entropy']:.4f}",
        'GeoLife': f"{geolife_results['vocabulary']['target_distribution_entropy']:.4f}",
        'Difference': f"{geolife_results['vocabulary']['target_distribution_entropy'] - diy_results['vocabulary']['target_distribution_entropy']:.4f}",
    })
    
    # User Patterns
    summary_data.append({
        'Category': 'User Patterns',
        'Metric': 'Number of Users',
        'DIY': f"{diy_results['user_patterns']['num_users']}",
        'GeoLife': f"{geolife_results['user_patterns']['num_users']}",
        'Difference': f"{geolife_results['user_patterns']['num_users'] - diy_results['user_patterns']['num_users']}",
    })
    summary_data.append({
        'Category': 'User Patterns',
        'Metric': 'Avg Target Revisit Rate',
        'DIY': f"{diy_results['user_patterns']['avg_target_revisit_rate']:.4f}",
        'GeoLife': f"{geolife_results['user_patterns']['avg_target_revisit_rate']:.4f}",
        'Difference': f"{geolife_results['user_patterns']['avg_target_revisit_rate'] - diy_results['user_patterns']['avg_target_revisit_rate']:.4f}",
    })
    
    # Sequence Characteristics
    summary_data.append({
        'Category': 'Sequence',
        'Metric': 'Avg Sequence Length',
        'DIY': f"{diy_results['sequence']['avg_seq_length']:.2f}",
        'GeoLife': f"{geolife_results['sequence']['avg_seq_length']:.2f}",
        'Difference': f"{geolife_results['sequence']['avg_seq_length'] - diy_results['sequence']['avg_seq_length']:.2f}",
    })
    summary_data.append({
        'Category': 'Sequence',
        'Metric': 'Total Test Samples',
        'DIY': f"{diy_results['target_in_history']['total_samples']}",
        'GeoLife': f"{geolife_results['target_in_history']['total_samples']}",
        'Difference': f"{geolife_results['target_in_history']['total_samples'] - diy_results['target_in_history']['total_samples']}",
    })
    
    df = pd.DataFrame(summary_data)
    
    # Save as CSV
    df.to_csv(output_dir / 'descriptive_analysis_results.csv', index=False)
    
    # Save as markdown
    markdown_table = df.to_markdown(index=False)
    with open(output_dir / 'descriptive_analysis_results.md', 'w') as f:
        f.write("# Descriptive Analysis Results\n\n")
        f.write("## Dataset Characteristics Comparison\n\n")
        f.write(markdown_table)
        f.write("\n\n")
        
        # Add interpretation
        f.write("## Key Findings\n\n")
        f.write(f"### 1. Target-in-History Rate (Copy Applicability)\n")
        f.write(f"- **GeoLife**: {geolife_results['target_in_history']['target_in_history_rate']:.2f}% of targets appear in history\n")
        f.write(f"- **DIY**: {diy_results['target_in_history']['target_in_history_rate']:.2f}% of targets appear in history\n")
        f.write(f"- **Interpretation**: GeoLife has significantly higher copy applicability, meaning the pointer mechanism has more opportunity to be useful.\n\n")
        
        f.write(f"### 2. Repetition Patterns\n")
        f.write(f"- **GeoLife Repetition Rate**: {geolife_results['repetition']['avg_repetition_rate']:.4f}\n")
        f.write(f"- **DIY Repetition Rate**: {diy_results['repetition']['avg_repetition_rate']:.4f}\n")
        f.write(f"- **Interpretation**: Higher repetition in GeoLife sequences means more opportunities for the pointer to select from repeated locations.\n\n")
    
    return df


def main():
    print("=" * 70)
    print("DESCRIPTIVE ANALYSIS: DIY vs GeoLife Dataset Characteristics")
    print("=" * 70)
    
    # Load datasets
    print("\nLoading datasets...")
    diy_test = load_dataset(DIY_TEST_PATH)
    geolife_test = load_dataset(GEOLIFE_TEST_PATH)
    diy_train = load_dataset(DIY_TRAIN_PATH)
    geolife_train = load_dataset(GEOLIFE_TRAIN_PATH)
    
    print(f"  DIY Test: {len(diy_test)} samples")
    print(f"  GeoLife Test: {len(geolife_test)} samples")
    
    # Run analyses
    print("\nRunning analyses...")
    
    diy_results = {}
    geolife_results = {}
    
    print("  1. Target-in-History Analysis...")
    diy_results['target_in_history'] = analyze_target_in_history(diy_test, 'DIY')
    geolife_results['target_in_history'] = analyze_target_in_history(geolife_test, 'GeoLife')
    
    print("  2. Repetition Pattern Analysis...")
    diy_results['repetition'] = analyze_repetition_patterns(diy_test, 'DIY')
    geolife_results['repetition'] = analyze_repetition_patterns(geolife_test, 'GeoLife')
    
    print("  3. Vocabulary Utilization Analysis...")
    diy_results['vocabulary'] = analyze_vocabulary_utilization(diy_test, diy_train, 'DIY')
    geolife_results['vocabulary'] = analyze_vocabulary_utilization(geolife_test, geolife_train, 'GeoLife')
    
    print("  4. User Pattern Analysis...")
    diy_results['user_patterns'] = analyze_user_patterns(diy_test, 'DIY')
    geolife_results['user_patterns'] = analyze_user_patterns(geolife_test, 'GeoLife')
    
    print("  5. Sequence Characteristics...")
    diy_results['sequence'] = analyze_sequence_characteristics(diy_test, 'DIY')
    geolife_results['sequence'] = analyze_sequence_characteristics(geolife_test, 'GeoLife')
    
    print("  6. Temporal Pattern Analysis...")
    diy_results['temporal'] = analyze_temporal_patterns(diy_test, 'DIY')
    geolife_results['temporal'] = analyze_temporal_patterns(geolife_test, 'GeoLife')
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(diy_results, geolife_results, OUTPUT_DIR)
    
    # Save results
    print("\nSaving results...")
    df = save_results_to_table(diy_results, geolife_results, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)
    
    print(f"\n1. TARGET-IN-HISTORY RATE (Copy Applicability):")
    print(f"   GeoLife: {geolife_results['target_in_history']['target_in_history_rate']:.2f}%")
    print(f"   DIY:     {diy_results['target_in_history']['target_in_history_rate']:.2f}%")
    print(f"   Δ:       {geolife_results['target_in_history']['target_in_history_rate'] - diy_results['target_in_history']['target_in_history_rate']:.2f}%")
    
    print(f"\n2. REPETITION RATE:")
    print(f"   GeoLife: {geolife_results['repetition']['avg_repetition_rate']:.4f}")
    print(f"   DIY:     {diy_results['repetition']['avg_repetition_rate']:.4f}")
    
    print(f"\n3. USER TARGET REVISIT RATE:")
    print(f"   GeoLife: {geolife_results['user_patterns']['avg_target_revisit_rate']:.4f}")
    print(f"   DIY:     {diy_results['user_patterns']['avg_target_revisit_rate']:.4f}")
    
    print(f"\n4. SEQUENCE LENGTH:")
    print(f"   GeoLife: {geolife_results['sequence']['avg_seq_length']:.2f}")
    print(f"   DIY:     {diy_results['sequence']['avg_seq_length']:.2f}")
    
    print(f"\n5. VOCABULARY SIZE:")
    print(f"   GeoLife: {geolife_results['vocabulary']['unique_locs_in_sequences']} unique locations")
    print(f"   DIY:     {diy_results['vocabulary']['unique_locs_in_sequences']} unique locations")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Save full results as JSON
    results_for_json = {
        'diy': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (pd.DataFrame, list)) or len(str(vv)) < 1000} 
                for k, v in diy_results.items()},
        'geolife': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (pd.DataFrame, list)) or len(str(vv)) < 1000} 
                   for k, v in geolife_results.items()},
    }
    
    # Remove non-serializable items
    for dataset in ['diy', 'geolife']:
        for category in results_for_json[dataset]:
            for key in list(results_for_json[dataset][category].keys()):
                val = results_for_json[dataset][category][key]
                if isinstance(val, (pd.DataFrame, Counter, defaultdict)):
                    del results_for_json[dataset][category][key]
                elif isinstance(val, list) and len(val) > 100:
                    results_for_json[dataset][category][key] = f"[List of {len(val)} items]"
    
    with open(OUTPUT_DIR / 'descriptive_analysis_full.json', 'w') as f:
        json.dump(results_for_json, f, indent=2, default=str)


if __name__ == "__main__":
    main()

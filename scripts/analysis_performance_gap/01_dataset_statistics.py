"""
Dataset Statistics Analysis for Performance Gap Investigation.

This script analyzes the statistical differences between Geolife and DIY datasets
to help explain why PGT shows larger improvement over MHSA baseline on Geolife
(+20.78pp) compared to DIY (+3.71pp).

Key hypotheses to investigate:
1. Location vocabulary size and distribution
2. Sequence length distributions
3. Location revisitation patterns (pointer mechanism benefit)
4. User behavior diversity
5. Temporal patterns
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_dataset(data_dir, prefix, split="train"):
    """Load dataset from pickle file."""
    path = os.path.join(data_dir, f"{prefix}_{split}.pk")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_metadata(data_dir, prefix):
    """Load metadata JSON."""
    path = os.path.join(data_dir, f"{prefix}_metadata.json")
    with open(path, "r") as f:
        return json.load(f)


def analyze_sequence_lengths(data, name):
    """Analyze sequence length distribution."""
    lengths = [len(sample['X']) for sample in data]
    
    stats = {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'median': np.median(lengths),
        'p25': np.percentile(lengths, 25),
        'p75': np.percentile(lengths, 75),
        'p90': np.percentile(lengths, 90),
        'p95': np.percentile(lengths, 95),
    }
    
    print(f"\n=== Sequence Length Analysis: {name} ===")
    print(f"  Mean:   {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Range:  [{stats['min']}, {stats['max']}]")
    print(f"  IQR:    [{stats['p25']:.0f}, {stats['p75']:.0f}]")
    print(f"  P90/P95: {stats['p90']:.0f} / {stats['p95']:.0f}")
    
    return stats, lengths


def analyze_location_distribution(data, name):
    """Analyze location vocabulary usage and distribution."""
    all_locs = []
    target_locs = []
    
    for sample in data:
        all_locs.extend(sample['X'].tolist())
        target_locs.append(sample['Y'])
    
    loc_counter = Counter(all_locs)
    target_counter = Counter(target_locs)
    
    # Calculate statistics
    num_unique = len(loc_counter)
    total_visits = len(all_locs)
    
    # Frequency distribution
    freq_values = list(loc_counter.values())
    
    # How concentrated is the distribution?
    sorted_freqs = sorted(freq_values, reverse=True)
    cumsum = np.cumsum(sorted_freqs) / sum(sorted_freqs)
    
    # Find how many locations cover X% of visits
    top_10_percent = int(num_unique * 0.1)
    top_20_percent = int(num_unique * 0.2)
    coverage_top10 = sum(sorted_freqs[:top_10_percent]) / total_visits * 100
    coverage_top20 = sum(sorted_freqs[:top_20_percent]) / total_visits * 100
    
    stats = {
        'num_unique': num_unique,
        'total_visits': total_visits,
        'freq_mean': np.mean(freq_values),
        'freq_std': np.std(freq_values),
        'freq_max': np.max(freq_values),
        'freq_min': np.min(freq_values),
        'coverage_top10_percent': coverage_top10,
        'coverage_top20_percent': coverage_top20,
    }
    
    print(f"\n=== Location Distribution Analysis: {name} ===")
    print(f"  Unique Locations: {num_unique}")
    print(f"  Total Visits:     {total_visits}")
    print(f"  Avg Visits/Loc:   {stats['freq_mean']:.2f} ± {stats['freq_std']:.2f}")
    print(f"  Most Visited:     {stats['freq_max']} times")
    print(f"  Top 10% locs cover: {coverage_top10:.1f}% of visits")
    print(f"  Top 20% locs cover: {coverage_top20:.1f}% of visits")
    
    return stats, loc_counter, target_counter


def analyze_revisitation_patterns(data, name):
    """
    Analyze location revisitation patterns - KEY for pointer mechanism benefit.
    
    The pointer mechanism excels when target locations appear in the input sequence.
    If targets rarely appear in history, pointer provides less benefit.
    """
    target_in_history = 0
    target_in_recent = 0  # In last 5 positions
    total_samples = len(data)
    
    history_positions = []  # Where in history does target appear
    
    for sample in data:
        history = set(sample['X'].tolist())
        target = sample['Y']
        
        if target in history:
            target_in_history += 1
            # Find positions (from end)
            x_list = sample['X'].tolist()
            for i, loc in enumerate(reversed(x_list)):
                if loc == target:
                    history_positions.append(i + 1)  # 1-indexed from end
                    break
            
            # Check if in recent history
            recent = set(x_list[-5:]) if len(x_list) >= 5 else set(x_list)
            if target in recent:
                target_in_recent += 1
    
    revisit_rate = target_in_history / total_samples * 100
    recent_revisit_rate = target_in_recent / total_samples * 100
    
    # Position distribution
    if history_positions:
        pos_mean = np.mean(history_positions)
        pos_median = np.median(history_positions)
    else:
        pos_mean = pos_median = 0
    
    stats = {
        'target_in_history': target_in_history,
        'target_in_history_rate': revisit_rate,
        'target_in_recent_rate': recent_revisit_rate,
        'avg_position_from_end': pos_mean,
        'median_position_from_end': pos_median,
    }
    
    print(f"\n=== Revisitation Pattern Analysis: {name} ===")
    print(f"  Total Samples:          {total_samples}")
    print(f"  Target in History:      {target_in_history} ({revisit_rate:.1f}%)")
    print(f"  Target in Recent (5):   {target_in_recent} ({recent_revisit_rate:.1f}%)")
    print(f"  Avg Position from End:  {pos_mean:.1f}")
    print(f"  Median Position from End: {pos_median:.1f}")
    print(f"\n  ** Higher revisitation rate = more benefit from pointer mechanism **")
    
    return stats, history_positions


def analyze_user_patterns(data, name):
    """Analyze per-user patterns."""
    user_samples = defaultdict(list)
    user_locs = defaultdict(set)
    
    for sample in data:
        user = sample['user_X'][0]
        user_samples[user].append(sample)
        user_locs[user].update(sample['X'].tolist())
        user_locs[user].add(sample['Y'])
    
    num_users = len(user_samples)
    samples_per_user = [len(v) for v in user_samples.values()]
    locs_per_user = [len(v) for v in user_locs.values()]
    
    stats = {
        'num_users': num_users,
        'samples_per_user_mean': np.mean(samples_per_user),
        'samples_per_user_std': np.std(samples_per_user),
        'locs_per_user_mean': np.mean(locs_per_user),
        'locs_per_user_std': np.std(locs_per_user),
    }
    
    print(f"\n=== User Pattern Analysis: {name} ===")
    print(f"  Number of Users:        {num_users}")
    print(f"  Samples per User:       {stats['samples_per_user_mean']:.1f} ± {stats['samples_per_user_std']:.1f}")
    print(f"  Locations per User:     {stats['locs_per_user_mean']:.1f} ± {stats['locs_per_user_std']:.1f}")
    
    return stats


def analyze_temporal_patterns(data, name):
    """Analyze temporal feature distributions."""
    time_slots = []
    weekdays = []
    durations = []
    
    for sample in data:
        time_slots.extend(sample['start_min_X'].tolist())
        weekdays.extend(sample['weekday_X'].tolist())
        durations.extend(sample['dur_X'].tolist())
    
    stats = {
        'duration_mean': np.mean(durations),
        'duration_std': np.std(durations),
        'duration_median': np.median(durations),
        'time_entropy': calculate_entropy(time_slots),
        'weekday_entropy': calculate_entropy(weekdays),
    }
    
    print(f"\n=== Temporal Pattern Analysis: {name} ===")
    print(f"  Duration Mean:    {stats['duration_mean']:.1f} ± {stats['duration_std']:.1f} min")
    print(f"  Duration Median:  {stats['duration_median']:.1f} min")
    print(f"  Time Entropy:     {stats['time_entropy']:.3f}")
    print(f"  Weekday Entropy:  {stats['weekday_entropy']:.3f}")
    
    return stats


def calculate_entropy(values):
    """Calculate entropy of a distribution."""
    counter = Counter(values)
    total = len(values)
    probs = [count / total for count in counter.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def analyze_target_prediction_difficulty(data, name):
    """
    Analyze how difficult target prediction is.
    
    Key insight: If targets are highly concentrated (few locations dominate),
    a simple frequency-based baseline performs well, limiting improvement headroom.
    """
    targets = [sample['Y'] for sample in data]
    target_counter = Counter(targets)
    
    total = len(targets)
    num_unique_targets = len(target_counter)
    
    # Top-K coverage
    sorted_targets = sorted(target_counter.values(), reverse=True)
    top1_coverage = sorted_targets[0] / total * 100
    top5_coverage = sum(sorted_targets[:5]) / total * 100 if len(sorted_targets) >= 5 else 100
    top10_coverage = sum(sorted_targets[:10]) / total * 100 if len(sorted_targets) >= 10 else 100
    top20_coverage = sum(sorted_targets[:20]) / total * 100 if len(sorted_targets) >= 20 else 100
    
    # Entropy of target distribution
    target_entropy = calculate_entropy(targets)
    max_entropy = np.log2(num_unique_targets)
    normalized_entropy = target_entropy / max_entropy if max_entropy > 0 else 0
    
    stats = {
        'num_unique_targets': num_unique_targets,
        'top1_coverage': top1_coverage,
        'top5_coverage': top5_coverage,
        'top10_coverage': top10_coverage,
        'top20_coverage': top20_coverage,
        'target_entropy': target_entropy,
        'normalized_entropy': normalized_entropy,
    }
    
    print(f"\n=== Target Prediction Difficulty: {name} ===")
    print(f"  Unique Targets:     {num_unique_targets}")
    print(f"  Top-1 Coverage:     {top1_coverage:.1f}%")
    print(f"  Top-5 Coverage:     {top5_coverage:.1f}%")
    print(f"  Top-10 Coverage:    {top10_coverage:.1f}%")
    print(f"  Top-20 Coverage:    {top20_coverage:.1f}%")
    print(f"  Target Entropy:     {target_entropy:.3f} (normalized: {normalized_entropy:.3f})")
    print(f"\n  ** Higher top-K coverage = easier prediction, less room for improvement **")
    
    return stats


def main():
    print("=" * 80)
    print("DATASET STATISTICS ANALYSIS FOR PERFORMANCE GAP INVESTIGATION")
    print("=" * 80)
    print("\nGoal: Understand why PGT shows +20.78pp improvement on Geolife")
    print("      but only +3.71pp improvement on DIY dataset.")
    print("=" * 80)
    
    # Paths
    geolife_dir = "data/geolife_eps20/processed"
    diy_dir = "data/diy_eps50/processed"
    geolife_prefix = "geolife_eps20_prev7"
    diy_prefix = "diy_eps50_prev7"
    
    # Load datasets
    print("\nLoading datasets...")
    geolife_train = load_dataset(geolife_dir, geolife_prefix, "train")
    geolife_test = load_dataset(geolife_dir, geolife_prefix, "test")
    diy_train = load_dataset(diy_dir, diy_prefix, "train")
    diy_test = load_dataset(diy_dir, diy_prefix, "test")
    
    geolife_meta = load_metadata(geolife_dir, geolife_prefix)
    diy_meta = load_metadata(diy_dir, diy_prefix)
    
    print(f"\nGeolife: {len(geolife_train)} train, {len(geolife_test)} test samples")
    print(f"DIY:     {len(diy_train)} train, {len(diy_test)} test samples")
    
    all_results = {}
    
    # Combine train+test for comprehensive analysis
    geolife_all = geolife_train + geolife_test
    diy_all = diy_train + diy_test
    
    # 1. Sequence Length Analysis
    print("\n" + "=" * 80)
    print("1. SEQUENCE LENGTH ANALYSIS")
    print("=" * 80)
    geo_seq_stats, geo_lengths = analyze_sequence_lengths(geolife_all, "Geolife")
    diy_seq_stats, diy_lengths = analyze_sequence_lengths(diy_all, "DIY")
    all_results['sequence_length'] = {'geolife': geo_seq_stats, 'diy': diy_seq_stats}
    
    # 2. Location Distribution Analysis
    print("\n" + "=" * 80)
    print("2. LOCATION DISTRIBUTION ANALYSIS")
    print("=" * 80)
    geo_loc_stats, _, _ = analyze_location_distribution(geolife_all, "Geolife")
    diy_loc_stats, _, _ = analyze_location_distribution(diy_all, "DIY")
    all_results['location_distribution'] = {'geolife': geo_loc_stats, 'diy': diy_loc_stats}
    
    # 3. REVISITATION PATTERN ANALYSIS (KEY FOR POINTER!)
    print("\n" + "=" * 80)
    print("3. REVISITATION PATTERN ANALYSIS (KEY FOR POINTER MECHANISM)")
    print("=" * 80)
    geo_revisit_stats, _ = analyze_revisitation_patterns(geolife_all, "Geolife")
    diy_revisit_stats, _ = analyze_revisitation_patterns(diy_all, "DIY")
    all_results['revisitation'] = {'geolife': geo_revisit_stats, 'diy': diy_revisit_stats}
    
    # 4. User Pattern Analysis
    print("\n" + "=" * 80)
    print("4. USER PATTERN ANALYSIS")
    print("=" * 80)
    geo_user_stats = analyze_user_patterns(geolife_all, "Geolife")
    diy_user_stats = analyze_user_patterns(diy_all, "DIY")
    all_results['user_patterns'] = {'geolife': geo_user_stats, 'diy': diy_user_stats}
    
    # 5. Temporal Pattern Analysis
    print("\n" + "=" * 80)
    print("5. TEMPORAL PATTERN ANALYSIS")
    print("=" * 80)
    geo_temporal_stats = analyze_temporal_patterns(geolife_all, "Geolife")
    diy_temporal_stats = analyze_temporal_patterns(diy_all, "DIY")
    all_results['temporal_patterns'] = {'geolife': geo_temporal_stats, 'diy': diy_temporal_stats}
    
    # 6. Target Prediction Difficulty
    print("\n" + "=" * 80)
    print("6. TARGET PREDICTION DIFFICULTY ANALYSIS")
    print("=" * 80)
    geo_target_stats = analyze_target_prediction_difficulty(geolife_all, "Geolife")
    diy_target_stats = analyze_target_prediction_difficulty(diy_all, "DIY")
    all_results['target_difficulty'] = {'geolife': geo_target_stats, 'diy': diy_target_stats}
    
    # Summary and Key Findings
    print("\n" + "=" * 80)
    print("SUMMARY: KEY DIFFERENCES BETWEEN DATASETS")
    print("=" * 80)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                        COMPARISON TABLE                             │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Metric                        │  Geolife      │  DIY          │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Total Samples                 │  {len(geolife_all):>10}   │  {len(diy_all):>10}   │")
    print(f"│ Unique Locations              │  {geo_loc_stats['num_unique']:>10}   │  {diy_loc_stats['num_unique']:>10}   │")
    print(f"│ Avg Sequence Length           │  {geo_seq_stats['mean']:>10.1f}   │  {diy_seq_stats['mean']:>10.1f}   │")
    print(f"│ Target in History Rate (%)    │  {geo_revisit_stats['target_in_history_rate']:>10.1f}   │  {diy_revisit_stats['target_in_history_rate']:>10.1f}   │")
    print(f"│ Target in Recent-5 Rate (%)   │  {geo_revisit_stats['target_in_recent_rate']:>10.1f}   │  {diy_revisit_stats['target_in_recent_rate']:>10.1f}   │")
    print(f"│ Top-5 Target Coverage (%)     │  {geo_target_stats['top5_coverage']:>10.1f}   │  {diy_target_stats['top5_coverage']:>10.1f}   │")
    print(f"│ Top-10 Target Coverage (%)    │  {geo_target_stats['top10_coverage']:>10.1f}   │  {diy_target_stats['top10_coverage']:>10.1f}   │")
    print(f"│ Target Entropy                │  {geo_target_stats['target_entropy']:>10.2f}   │  {diy_target_stats['target_entropy']:>10.2f}   │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    # Save results
    output_dir = "scripts/analysis_performance_gap/results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "dataset_statistics.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()

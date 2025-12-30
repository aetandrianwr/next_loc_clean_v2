"""
03. Location Frequency Distribution Analysis
Analyze the distribution of location visits to understand prediction difficulty.

This script investigates:
- Location frequency distribution (Zipf's law)
- Concentration of visits among top locations
- Long-tail analysis
- Vocabulary utilization

KEY HYPOTHESIS: More concentrated distributions favor simpler models,
while long-tail distributions make prediction harder and require more
sophisticated mechanisms.
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
from scipy import stats

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_sequences(dataset_name, epsilon, split, prev_day=7):
    """Load sequence data from pickle file."""
    path = os.path.join(BASE_DIR, "data", f"{dataset_name}_eps{epsilon}",
                        "processed", f"{dataset_name}_eps{epsilon}_prev{prev_day}_{split}.pk")
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_location_frequencies(sequences, count_type='target'):
    """
    Count location frequencies.
    
    Args:
        sequences: List of sequence dictionaries
        count_type: 'target' for target locations, 'history' for history locations, 'all' for both
    """
    location_counts = Counter()
    
    for seq in sequences:
        if count_type in ['target', 'all']:
            location_counts[seq['Y']] += 1
        if count_type in ['history', 'all']:
            for loc in seq['X']:
                location_counts[loc] += 1
    
    return location_counts


def analyze_frequency_distribution(location_counts):
    """Analyze the frequency distribution of locations."""
    counts = sorted(location_counts.values(), reverse=True)
    total_visits = sum(counts)
    num_locations = len(counts)
    
    # Concentration metrics
    cumsum = np.cumsum(counts)
    cumsum_pct = cumsum / total_visits * 100
    
    # How many locations cover X% of visits
    top_10_pct_locs = np.sum(cumsum_pct <= 10)
    top_20_pct_locs = np.sum(cumsum_pct <= 20)
    top_50_pct_locs = np.sum(cumsum_pct <= 50)
    
    # What percentage of visits come from top N locations
    top_10_locs_pct = cumsum[min(10, num_locations-1)] / total_visits * 100 if num_locations >= 10 else 100
    top_50_locs_pct = cumsum[min(50, num_locations-1)] / total_visits * 100 if num_locations >= 50 else 100
    top_100_locs_pct = cumsum[min(100, num_locations-1)] / total_visits * 100 if num_locations >= 100 else 100
    
    # Entropy (measure of distribution spread)
    probs = np.array(counts) / total_visits
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(num_locations)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Gini coefficient (inequality measure)
    counts_arr = np.array(sorted(counts))
    n = len(counts_arr)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * counts_arr) - (n + 1) * np.sum(counts_arr)) / (n * np.sum(counts_arr))
    
    return {
        'num_locations': num_locations,
        'total_visits': total_visits,
        'avg_visits_per_location': total_visits / num_locations,
        'median_visits_per_location': np.median(counts),
        'max_visits': max(counts),
        'min_visits': min(counts),
        'top_10_locs_cover_pct': top_10_locs_pct,
        'top_50_locs_cover_pct': top_50_locs_pct,
        'top_100_locs_cover_pct': top_100_locs_pct,
        'locs_for_50pct': top_50_pct_locs,
        'locs_for_50pct_ratio': top_50_pct_locs / num_locations * 100,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'gini_coefficient': gini,
        'counts': counts,
        'cumsum_pct': cumsum_pct
    }


def analyze_long_tail(sequences, location_counts):
    """Analyze long-tail phenomenon in target prediction."""
    target_counts = get_location_frequencies(sequences, 'target')
    counts = sorted(target_counts.values(), reverse=True)
    
    # Define rare locations (bottom 50% by frequency rank)
    rare_threshold_idx = len(counts) // 2
    if rare_threshold_idx < len(counts):
        rare_threshold = counts[rare_threshold_idx]
    else:
        rare_threshold = 1
    
    # Count how many targets are rare locations
    rare_targets = sum(1 for seq in sequences if target_counts[seq['Y']] <= rare_threshold)
    rare_target_rate = rare_targets / len(sequences) * 100
    
    # Singleton targets (appear only once in training)
    singleton_locs = set(loc for loc, count in target_counts.items() if count == 1)
    singleton_targets = sum(1 for seq in sequences if seq['Y'] in singleton_locs)
    singleton_rate = singleton_targets / len(sequences) * 100
    
    return {
        'rare_threshold': rare_threshold,
        'rare_target_rate': rare_target_rate,
        'singleton_count': len(singleton_locs),
        'singleton_target_rate': singleton_rate,
        'singleton_locs': singleton_locs
    }


def analyze_user_location_overlap(sequences):
    """Analyze how much users share locations vs have unique locations."""
    user_locations = {}
    
    for seq in sequences:
        user = seq['user_X'][0]
        if user not in user_locations:
            user_locations[user] = set()
        user_locations[user].update(seq['X'])
        user_locations[user].add(seq['Y'])
    
    # All unique locations
    all_locations = set()
    for locs in user_locations.values():
        all_locations.update(locs)
    
    # Location to user count
    loc_user_count = Counter()
    for user, locs in user_locations.items():
        for loc in locs:
            loc_user_count[loc] += 1
    
    # User-exclusive locations (visited by only 1 user)
    exclusive_locs = sum(1 for count in loc_user_count.values() if count == 1)
    shared_locs = len(all_locations) - exclusive_locs
    
    # Average users per location
    avg_users_per_loc = np.mean(list(loc_user_count.values()))
    
    return {
        'total_locations': len(all_locations),
        'total_users': len(user_locations),
        'exclusive_locations': exclusive_locs,
        'exclusive_rate': exclusive_locs / len(all_locations) * 100,
        'shared_locations': shared_locs,
        'shared_rate': shared_locs / len(all_locations) * 100,
        'avg_users_per_location': avg_users_per_loc,
        'max_users_per_location': max(loc_user_count.values()),
        'avg_locs_per_user': np.mean([len(locs) for locs in user_locations.values()])
    }


def main():
    print("="*80)
    print("03. Location Frequency Distribution Analysis")
    print("="*80)
    
    # Load sequences
    print("[Loading sequences...]")
    geo_train = load_sequences("geolife", 20, "train")
    geo_test = load_sequences("geolife", 20, "test")
    diy_train = load_sequences("diy", 50, "train")
    diy_test = load_sequences("diy", 50, "test")
    
    # Get location frequencies (targets only - what we predict)
    print("\n[Analyzing target location frequencies...]")
    geo_train_counts = get_location_frequencies(geo_train, 'target')
    geo_test_counts = get_location_frequencies(geo_test, 'target')
    diy_train_counts = get_location_frequencies(diy_train, 'target')
    diy_test_counts = get_location_frequencies(diy_test, 'target')
    
    # Frequency distribution analysis
    geo_train_freq = analyze_frequency_distribution(geo_train_counts)
    geo_test_freq = analyze_frequency_distribution(geo_test_counts)
    diy_train_freq = analyze_frequency_distribution(diy_train_counts)
    diy_test_freq = analyze_frequency_distribution(diy_test_counts)
    
    print("\n" + "="*80)
    print("TARGET LOCATION FREQUENCY DISTRIBUTION")
    print("="*80)
    
    freq_comparison = {
        'Metric': [
            'Unique Target Locations',
            'Total Targets',
            'Avg Visits per Location',
            'Median Visits per Location',
            'Top 10 Locations Cover (%)',
            'Top 50 Locations Cover (%)',
            'Locations for 50% Coverage (%)',
            'Entropy (bits)',
            'Normalized Entropy',
            'Gini Coefficient'
        ],
        'Geolife Train': [
            geo_train_freq['num_locations'],
            geo_train_freq['total_visits'],
            f"{geo_train_freq['avg_visits_per_location']:.2f}",
            f"{geo_train_freq['median_visits_per_location']:.1f}",
            f"{geo_train_freq['top_10_locs_cover_pct']:.1f}%",
            f"{geo_train_freq['top_50_locs_cover_pct']:.1f}%",
            f"{geo_train_freq['locs_for_50pct_ratio']:.1f}%",
            f"{geo_train_freq['entropy']:.2f}",
            f"{geo_train_freq['normalized_entropy']:.3f}",
            f"{geo_train_freq['gini_coefficient']:.3f}"
        ],
        'Geolife Test': [
            geo_test_freq['num_locations'],
            geo_test_freq['total_visits'],
            f"{geo_test_freq['avg_visits_per_location']:.2f}",
            f"{geo_test_freq['median_visits_per_location']:.1f}",
            f"{geo_test_freq['top_10_locs_cover_pct']:.1f}%",
            f"{geo_test_freq['top_50_locs_cover_pct']:.1f}%",
            f"{geo_test_freq['locs_for_50pct_ratio']:.1f}%",
            f"{geo_test_freq['entropy']:.2f}",
            f"{geo_test_freq['normalized_entropy']:.3f}",
            f"{geo_test_freq['gini_coefficient']:.3f}"
        ],
        'DIY Train': [
            diy_train_freq['num_locations'],
            diy_train_freq['total_visits'],
            f"{diy_train_freq['avg_visits_per_location']:.2f}",
            f"{diy_train_freq['median_visits_per_location']:.1f}",
            f"{diy_train_freq['top_10_locs_cover_pct']:.1f}%",
            f"{diy_train_freq['top_50_locs_cover_pct']:.1f}%",
            f"{diy_train_freq['locs_for_50pct_ratio']:.1f}%",
            f"{diy_train_freq['entropy']:.2f}",
            f"{diy_train_freq['normalized_entropy']:.3f}",
            f"{diy_train_freq['gini_coefficient']:.3f}"
        ],
        'DIY Test': [
            diy_test_freq['num_locations'],
            diy_test_freq['total_visits'],
            f"{diy_test_freq['avg_visits_per_location']:.2f}",
            f"{diy_test_freq['median_visits_per_location']:.1f}",
            f"{diy_test_freq['top_10_locs_cover_pct']:.1f}%",
            f"{diy_test_freq['top_50_locs_cover_pct']:.1f}%",
            f"{diy_test_freq['locs_for_50pct_ratio']:.1f}%",
            f"{diy_test_freq['entropy']:.2f}",
            f"{diy_test_freq['normalized_entropy']:.3f}",
            f"{diy_test_freq['gini_coefficient']:.3f}"
        ],
    }
    
    freq_df = pd.DataFrame(freq_comparison)
    print(freq_df.to_string(index=False))
    freq_df.to_csv(os.path.join(RESULTS_DIR, "03_frequency_distribution.csv"), index=False)
    
    # Long-tail analysis
    print("\n[Analyzing long-tail phenomenon...]")
    geo_train_tail = analyze_long_tail(geo_train, geo_train_counts)
    geo_test_tail = analyze_long_tail(geo_test, geo_test_counts)
    diy_train_tail = analyze_long_tail(diy_train, diy_train_counts)
    diy_test_tail = analyze_long_tail(diy_test, diy_test_counts)
    
    print("\n" + "="*80)
    print("LONG-TAIL ANALYSIS (Rare/Singleton Targets)")
    print("="*80)
    
    tail_comparison = {
        'Metric': ['Singleton Locations (count=1)', 'Singleton Target Rate (%)', 'Rare Target Rate (%)'],
        'Geolife Train': [geo_train_tail['singleton_count'], 
                         f"{geo_train_tail['singleton_target_rate']:.2f}%",
                         f"{geo_train_tail['rare_target_rate']:.2f}%"],
        'Geolife Test': [geo_test_tail['singleton_count'],
                        f"{geo_test_tail['singleton_target_rate']:.2f}%",
                        f"{geo_test_tail['rare_target_rate']:.2f}%"],
        'DIY Train': [diy_train_tail['singleton_count'],
                     f"{diy_train_tail['singleton_target_rate']:.2f}%",
                     f"{diy_train_tail['rare_target_rate']:.2f}%"],
        'DIY Test': [diy_test_tail['singleton_count'],
                    f"{diy_test_tail['singleton_target_rate']:.2f}%",
                    f"{diy_test_tail['rare_target_rate']:.2f}%"],
    }
    
    tail_df = pd.DataFrame(tail_comparison)
    print(tail_df.to_string(index=False))
    tail_df.to_csv(os.path.join(RESULTS_DIR, "03_long_tail_analysis.csv"), index=False)
    
    # User-location overlap analysis
    print("\n[Analyzing user-location overlap...]")
    geo_overlap = analyze_user_location_overlap(geo_train + geo_test)
    diy_overlap = analyze_user_location_overlap(diy_train + diy_test)
    
    print("\n" + "="*80)
    print("USER-LOCATION OVERLAP ANALYSIS")
    print("="*80)
    
    overlap_comparison = {
        'Metric': ['Total Users', 'Total Locations', 'Exclusive Locations (%)', 
                   'Shared Locations (%)', 'Avg Users per Location', 'Avg Locations per User'],
        'Geolife': [geo_overlap['total_users'], geo_overlap['total_locations'],
                   f"{geo_overlap['exclusive_rate']:.1f}%", f"{geo_overlap['shared_rate']:.1f}%",
                   f"{geo_overlap['avg_users_per_location']:.2f}", f"{geo_overlap['avg_locs_per_user']:.1f}"],
        'DIY': [diy_overlap['total_users'], diy_overlap['total_locations'],
               f"{diy_overlap['exclusive_rate']:.1f}%", f"{diy_overlap['shared_rate']:.1f}%",
               f"{diy_overlap['avg_users_per_location']:.2f}", f"{diy_overlap['avg_locs_per_user']:.1f}"],
    }
    
    overlap_df = pd.DataFrame(overlap_comparison)
    print(overlap_df.to_string(index=False))
    overlap_df.to_csv(os.path.join(RESULTS_DIR, "03_user_location_overlap.csv"), index=False)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Frequency distribution (log-log plot)
    ax1 = axes[0, 0]
    geo_ranks = np.arange(1, len(geo_test_freq['counts']) + 1)
    diy_ranks = np.arange(1, len(diy_test_freq['counts']) + 1)
    ax1.loglog(geo_ranks, geo_test_freq['counts'], 'o-', alpha=0.7, markersize=3, label='Geolife Test')
    ax1.loglog(diy_ranks, diy_test_freq['counts'], 'o-', alpha=0.7, markersize=3, label='DIY Test')
    ax1.set_xlabel('Location Rank (log scale)')
    ax1.set_ylabel('Frequency (log scale)')
    ax1.set_title('Target Location Frequency Distribution (Zipf Plot)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative coverage
    ax2 = axes[0, 1]
    geo_locs_pct = np.arange(1, len(geo_test_freq['cumsum_pct']) + 1) / len(geo_test_freq['cumsum_pct']) * 100
    diy_locs_pct = np.arange(1, len(diy_test_freq['cumsum_pct']) + 1) / len(diy_test_freq['cumsum_pct']) * 100
    ax2.plot(geo_locs_pct, geo_test_freq['cumsum_pct'], label='Geolife Test', linewidth=2)
    ax2.plot(diy_locs_pct, diy_test_freq['cumsum_pct'], label='DIY Test', linewidth=2)
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% coverage')
    ax2.axhline(90, color='gray', linestyle=':', alpha=0.5, label='90% coverage')
    ax2.set_xlabel('% of Locations')
    ax2.set_ylabel('Cumulative % of Visits')
    ax2.set_title('Cumulative Visit Coverage')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution metrics comparison
    ax3 = axes[1, 0]
    metrics = ['Gini', 'Norm. Entropy', 'Top-10 Cover\n(÷10)']
    geolife_vals = [geo_test_freq['gini_coefficient'], 
                   geo_test_freq['normalized_entropy'],
                   geo_test_freq['top_10_locs_cover_pct']/10]
    diy_vals = [diy_test_freq['gini_coefficient'],
               diy_test_freq['normalized_entropy'],
               diy_test_freq['top_10_locs_cover_pct']/10]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, geolife_vals, width, label='Geolife', color='steelblue')
    ax3.bar(x + width/2, diy_vals, width, label='DIY', color='coral')
    ax3.set_ylabel('Value')
    ax3.set_title('Distribution Metrics Comparison (Test Set)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    
    # 4. User-location overlap comparison
    ax4 = axes[1, 1]
    metrics = ['Exclusive\nLocs (%)', 'Shared\nLocs (%)', 'Avg Users\nper Loc']
    geolife_vals = [geo_overlap['exclusive_rate'], geo_overlap['shared_rate'], 
                   geo_overlap['avg_users_per_location']]
    diy_vals = [diy_overlap['exclusive_rate'], diy_overlap['shared_rate'],
               diy_overlap['avg_users_per_location']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, geolife_vals, width, label='Geolife', color='steelblue')
    ax4.bar(x + width/2, diy_vals, width, label='DIY', color='coral')
    ax4.set_ylabel('Value / %')
    ax4.set_title('User-Location Overlap')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "03_frequency_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    findings = []
    
    # Vocabulary concentration
    findings.append(f"1. Vocabulary Concentration:")
    findings.append(f"   - Geolife: Top 10 locations cover {geo_test_freq['top_10_locs_cover_pct']:.1f}% of targets")
    findings.append(f"   - DIY: Top 10 locations cover {diy_test_freq['top_10_locs_cover_pct']:.1f}% of targets")
    
    # Gini coefficient (inequality)
    findings.append(f"2. Gini Coefficient (higher = more unequal):")
    findings.append(f"   - Geolife: {geo_test_freq['gini_coefficient']:.3f}")
    findings.append(f"   - DIY: {diy_test_freq['gini_coefficient']:.3f}")
    
    # Entropy
    findings.append(f"3. Normalized Entropy (higher = more spread):")
    findings.append(f"   - Geolife: {geo_test_freq['normalized_entropy']:.3f}")
    findings.append(f"   - DIY: {diy_test_freq['normalized_entropy']:.3f}")
    
    # Exclusive locations
    findings.append(f"4. User-Exclusive Locations:")
    findings.append(f"   - Geolife: {geo_overlap['exclusive_rate']:.1f}% of locations visited by only 1 user")
    findings.append(f"   - DIY: {diy_overlap['exclusive_rate']:.1f}% of locations visited by only 1 user")
    
    findings.append("")
    findings.append("INTERPRETATION:")
    if geo_test_freq['gini_coefficient'] > diy_test_freq['gini_coefficient']:
        findings.append("  - Geolife has more concentrated distribution (fewer locations dominate)")
        findings.append("  - This makes it easier for pointer to focus on frequent locations")
    if geo_overlap['exclusive_rate'] > diy_overlap['exclusive_rate']:
        findings.append("  - Geolife users have more personalized location sets")
        findings.append("  - Personal history is more predictive in Geolife")
    
    for finding in findings:
        print(finding)
    
    with open(os.path.join(RESULTS_DIR, "03_key_findings.txt"), 'w') as f:
        f.write("Location Frequency Distribution Key Findings\n")
        f.write("="*50 + "\n\n")
        for finding in findings:
            f.write(finding + "\n")
    
    print(f"\n✓ Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

"""
04. User Behavior Analysis
Analyze user-level behavior patterns to understand prediction difficulty.

This script investigates:
- User mobility patterns
- User-specific prediction difficulty
- Home/work location regularity
- User trajectory predictability

KEY HYPOTHESIS: Pointer mechanism benefits more when users have 
more regular, predictable patterns.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy.stats import entropy

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


def analyze_user_patterns(sequences):
    """Analyze mobility patterns for each user."""
    user_data = defaultdict(lambda: {
        'sequences': [],
        'all_locations': [],
        'targets': []
    })
    
    for seq in sequences:
        user = seq['user_X'][0]
        user_data[user]['sequences'].append(seq)
        user_data[user]['all_locations'].extend(seq['X'].tolist())
        user_data[user]['targets'].append(seq['Y'])
    
    user_metrics = {}
    
    for user, data in user_data.items():
        loc_counts = Counter(data['all_locations'])
        target_counts = Counter(data['targets'])
        
        # Number of unique locations
        unique_locs = len(loc_counts)
        
        # Location entropy (measure of randomness)
        total_visits = sum(loc_counts.values())
        probs = np.array(list(loc_counts.values())) / total_visits
        loc_entropy = entropy(probs, base=2)
        max_entropy = np.log2(unique_locs) if unique_locs > 1 else 1
        normalized_entropy = loc_entropy / max_entropy if max_entropy > 0 else 0
        
        # Top-k location concentration
        sorted_counts = sorted(loc_counts.values(), reverse=True)
        top_1_pct = sorted_counts[0] / total_visits * 100 if sorted_counts else 0
        top_3_pct = sum(sorted_counts[:3]) / total_visits * 100 if len(sorted_counts) >= 3 else 100
        top_5_pct = sum(sorted_counts[:5]) / total_visits * 100 if len(sorted_counts) >= 5 else 100
        
        # Target predictability (how often target is in history)
        target_in_history = 0
        for seq in data['sequences']:
            if seq['Y'] in seq['X']:
                target_in_history += 1
        target_in_history_rate = target_in_history / len(data['sequences']) * 100
        
        # Target concentration (how concentrated are targets)
        if target_counts:
            target_probs = np.array(list(target_counts.values())) / len(data['targets'])
            target_entropy = entropy(target_probs, base=2)
            max_target_entropy = np.log2(len(target_counts)) if len(target_counts) > 1 else 1
            target_normalized_entropy = target_entropy / max_target_entropy if max_target_entropy > 0 else 0
        else:
            target_normalized_entropy = 0
        
        # Most frequent location percentage
        most_frequent_loc_pct = sorted_counts[0] / total_visits * 100 if sorted_counts else 0
        
        user_metrics[user] = {
            'num_sequences': len(data['sequences']),
            'unique_locations': unique_locs,
            'total_visits': total_visits,
            'loc_entropy': loc_entropy,
            'normalized_entropy': normalized_entropy,
            'top_1_concentration': top_1_pct,
            'top_3_concentration': top_3_pct,
            'top_5_concentration': top_5_pct,
            'target_in_history_rate': target_in_history_rate,
            'target_normalized_entropy': target_normalized_entropy,
            'most_frequent_loc_pct': most_frequent_loc_pct
        }
    
    return user_metrics


def analyze_regularity(sequences):
    """Analyze temporal regularity patterns."""
    user_time_patterns = defaultdict(lambda: defaultdict(Counter))
    
    for seq in sequences:
        user = seq['user_X'][0]
        # Last location in history -> target transition
        if len(seq['X']) > 0:
            last_loc = seq['X'][-1]
            target = seq['Y']
            user_time_patterns[user][last_loc][target] += 1
    
    # Calculate transition predictability
    user_transition_predictability = {}
    
    for user, loc_transitions in user_time_patterns.items():
        predictabilities = []
        for loc, targets in loc_transitions.items():
            total = sum(targets.values())
            if total > 1:
                max_target = max(targets.values())
                predictability = max_target / total
                predictabilities.append(predictability)
        
        if predictabilities:
            user_transition_predictability[user] = np.mean(predictabilities)
        else:
            user_transition_predictability[user] = 0
    
    return user_transition_predictability


def aggregate_user_metrics(user_metrics):
    """Aggregate user metrics to dataset level."""
    metrics_df = pd.DataFrame(user_metrics).T
    
    return {
        'num_users': len(user_metrics),
        'avg_sequences_per_user': metrics_df['num_sequences'].mean(),
        'avg_unique_locations': metrics_df['unique_locations'].mean(),
        'median_unique_locations': metrics_df['unique_locations'].median(),
        'avg_normalized_entropy': metrics_df['normalized_entropy'].mean(),
        'avg_top_1_concentration': metrics_df['top_1_concentration'].mean(),
        'avg_top_3_concentration': metrics_df['top_3_concentration'].mean(),
        'avg_target_in_history_rate': metrics_df['target_in_history_rate'].mean(),
        'avg_target_entropy': metrics_df['target_normalized_entropy'].mean(),
        'metrics_df': metrics_df
    }


def main():
    print("="*80)
    print("04. User Behavior Analysis")
    print("="*80)
    
    # Load all sequences
    print("[Loading sequences...]")
    geo_train = load_sequences("geolife", 20, "train")
    geo_test = load_sequences("geolife", 20, "test")
    diy_train = load_sequences("diy", 50, "train")
    diy_test = load_sequences("diy", 50, "test")
    
    # Combine train and test for comprehensive user analysis
    geo_all = geo_train + geo_test
    diy_all = diy_train + diy_test
    
    # Analyze user patterns
    print("\n[Analyzing user patterns...]")
    geo_user_metrics = analyze_user_patterns(geo_all)
    diy_user_metrics = analyze_user_patterns(diy_all)
    
    # Test set analysis for evaluation comparison
    geo_test_metrics = analyze_user_patterns(geo_test)
    diy_test_metrics = analyze_user_patterns(diy_test)
    
    # Aggregate metrics
    geo_agg = aggregate_user_metrics(geo_user_metrics)
    diy_agg = aggregate_user_metrics(diy_user_metrics)
    geo_test_agg = aggregate_user_metrics(geo_test_metrics)
    diy_test_agg = aggregate_user_metrics(diy_test_metrics)
    
    print("\n" + "="*80)
    print("USER BEHAVIOR PATTERNS (All Data)")
    print("="*80)
    
    user_comparison = {
        'Metric': [
            'Number of Users',
            'Avg Sequences per User',
            'Avg Unique Locations per User',
            'Median Unique Locations per User',
            'Avg Location Entropy (normalized)',
            'Avg Top-1 Location Concentration (%)',
            'Avg Top-3 Location Concentration (%)',
            'Avg Target-in-History Rate (%)',
            'Avg Target Entropy (normalized)'
        ],
        'Geolife': [
            geo_agg['num_users'],
            f"{geo_agg['avg_sequences_per_user']:.1f}",
            f"{geo_agg['avg_unique_locations']:.1f}",
            f"{geo_agg['median_unique_locations']:.1f}",
            f"{geo_agg['avg_normalized_entropy']:.3f}",
            f"{geo_agg['avg_top_1_concentration']:.1f}%",
            f"{geo_agg['avg_top_3_concentration']:.1f}%",
            f"{geo_agg['avg_target_in_history_rate']:.1f}%",
            f"{geo_agg['avg_target_entropy']:.3f}"
        ],
        'DIY': [
            diy_agg['num_users'],
            f"{diy_agg['avg_sequences_per_user']:.1f}",
            f"{diy_agg['avg_unique_locations']:.1f}",
            f"{diy_agg['median_unique_locations']:.1f}",
            f"{diy_agg['avg_normalized_entropy']:.3f}",
            f"{diy_agg['avg_top_1_concentration']:.1f}%",
            f"{diy_agg['avg_top_3_concentration']:.1f}%",
            f"{diy_agg['avg_target_in_history_rate']:.1f}%",
            f"{diy_agg['avg_target_entropy']:.3f}"
        ],
    }
    
    user_df = pd.DataFrame(user_comparison)
    print(user_df.to_string(index=False))
    user_df.to_csv(os.path.join(RESULTS_DIR, "04_user_behavior.csv"), index=False)
    
    # Test set specific analysis
    print("\n" + "="*80)
    print("USER BEHAVIOR PATTERNS (Test Set Only)")
    print("="*80)
    
    test_comparison = {
        'Metric': [
            'Number of Users with Test Data',
            'Avg Target-in-History Rate (%)',
            'Avg Location Entropy (normalized)',
            'Avg Top-1 Location Concentration (%)'
        ],
        'Geolife': [
            geo_test_agg['num_users'],
            f"{geo_test_agg['avg_target_in_history_rate']:.1f}%",
            f"{geo_test_agg['avg_normalized_entropy']:.3f}",
            f"{geo_test_agg['avg_top_1_concentration']:.1f}%"
        ],
        'DIY': [
            diy_test_agg['num_users'],
            f"{diy_test_agg['avg_target_in_history_rate']:.1f}%",
            f"{diy_test_agg['avg_normalized_entropy']:.3f}",
            f"{diy_test_agg['avg_top_1_concentration']:.1f}%"
        ],
    }
    
    test_df = pd.DataFrame(test_comparison)
    print(test_df.to_string(index=False))
    test_df.to_csv(os.path.join(RESULTS_DIR, "04_user_behavior_test.csv"), index=False)
    
    # Analyze transition predictability
    print("\n[Analyzing transition patterns...]")
    geo_transition = analyze_regularity(geo_all)
    diy_transition = analyze_regularity(diy_all)
    
    geo_avg_predictability = np.mean(list(geo_transition.values()))
    diy_avg_predictability = np.mean(list(diy_transition.values()))
    
    print("\n" + "="*80)
    print("TRANSITION PREDICTABILITY")
    print("="*80)
    print(f"Geolife - Avg transition predictability: {geo_avg_predictability:.3f}")
    print(f"DIY     - Avg transition predictability: {diy_avg_predictability:.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution of target-in-history rate per user
    ax1 = axes[0, 0]
    geo_rates = [m['target_in_history_rate'] for m in geo_user_metrics.values()]
    diy_rates = [m['target_in_history_rate'] for m in diy_user_metrics.values()]
    ax1.hist(geo_rates, bins=20, alpha=0.7, label=f'Geolife (mean={np.mean(geo_rates):.1f}%)', density=True)
    ax1.hist(diy_rates, bins=20, alpha=0.7, label=f'DIY (mean={np.mean(diy_rates):.1f}%)', density=True)
    ax1.set_xlabel('Target-in-History Rate (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Per-User Target-in-History Rate Distribution')
    ax1.legend()
    ax1.axvline(np.mean(geo_rates), color='blue', linestyle='--', alpha=0.7)
    ax1.axvline(np.mean(diy_rates), color='orange', linestyle='--', alpha=0.7)
    
    # 2. Distribution of normalized entropy per user
    ax2 = axes[0, 1]
    geo_entropy = [m['normalized_entropy'] for m in geo_user_metrics.values()]
    diy_entropy = [m['normalized_entropy'] for m in diy_user_metrics.values()]
    ax2.hist(geo_entropy, bins=20, alpha=0.7, label=f'Geolife (mean={np.mean(geo_entropy):.3f})', density=True)
    ax2.hist(diy_entropy, bins=20, alpha=0.7, label=f'DIY (mean={np.mean(diy_entropy):.3f})', density=True)
    ax2.set_xlabel('Normalized Location Entropy')
    ax2.set_ylabel('Density')
    ax2.set_title('Per-User Location Entropy Distribution\n(Lower = More Concentrated)')
    ax2.legend()
    
    # 3. Top-1 concentration distribution
    ax3 = axes[1, 0]
    geo_conc = [m['top_1_concentration'] for m in geo_user_metrics.values()]
    diy_conc = [m['top_1_concentration'] for m in diy_user_metrics.values()]
    ax3.hist(geo_conc, bins=20, alpha=0.7, label=f'Geolife (mean={np.mean(geo_conc):.1f}%)', density=True)
    ax3.hist(diy_conc, bins=20, alpha=0.7, label=f'DIY (mean={np.mean(diy_conc):.1f}%)', density=True)
    ax3.set_xlabel('Top-1 Location Concentration (%)')
    ax3.set_ylabel('Density')
    ax3.set_title('Per-User Top Location Concentration\n(Higher = More Dominant Home/Work)')
    ax3.legend()
    
    # 4. Summary comparison
    ax4 = axes[1, 1]
    metrics = ['Target in\nHistory (%)', 'Norm. Entropy\n(inv, ×100)', 'Top-1 Conc.\n(%)', 'Transition\nPred. (×100)']
    geolife_vals = [
        geo_agg['avg_target_in_history_rate'],
        (1 - geo_agg['avg_normalized_entropy']) * 100,
        geo_agg['avg_top_1_concentration'],
        geo_avg_predictability * 100
    ]
    diy_vals = [
        diy_agg['avg_target_in_history_rate'],
        (1 - diy_agg['avg_normalized_entropy']) * 100,
        diy_agg['avg_top_1_concentration'],
        diy_avg_predictability * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, geolife_vals, width, label='Geolife', color='steelblue')
    ax4.bar(x + width/2, diy_vals, width, label='DIY', color='coral')
    ax4.set_ylabel('Value / %')
    ax4.set_title('User Behavior Summary\n(All Higher = Better for Pointer)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "04_user_behavior.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save detailed user metrics
    geo_metrics_detailed = pd.DataFrame(geo_user_metrics).T
    diy_metrics_detailed = pd.DataFrame(diy_user_metrics).T
    geo_metrics_detailed.to_csv(os.path.join(RESULTS_DIR, "04_geolife_user_details.csv"))
    diy_metrics_detailed.to_csv(os.path.join(RESULTS_DIR, "04_diy_user_details.csv"))
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    findings = []
    
    # Target in history
    findings.append(f"1. Per-User Target-in-History Rate:")
    findings.append(f"   - Geolife: {geo_agg['avg_target_in_history_rate']:.1f}% average")
    findings.append(f"   - DIY: {diy_agg['avg_target_in_history_rate']:.1f}% average")
    diff = geo_agg['avg_target_in_history_rate'] - diy_agg['avg_target_in_history_rate']
    findings.append(f"   - Difference: {diff:+.1f}% (Geolife {'higher' if diff > 0 else 'lower'})")
    
    # Entropy
    findings.append(f"2. Location Entropy (normalized, lower=more concentrated):")
    findings.append(f"   - Geolife: {geo_agg['avg_normalized_entropy']:.3f}")
    findings.append(f"   - DIY: {diy_agg['avg_normalized_entropy']:.3f}")
    
    # Top location concentration
    findings.append(f"3. Top-1 Location Concentration:")
    findings.append(f"   - Geolife: {geo_agg['avg_top_1_concentration']:.1f}%")
    findings.append(f"   - DIY: {diy_agg['avg_top_1_concentration']:.1f}%")
    
    # Transition predictability
    findings.append(f"4. Transition Predictability:")
    findings.append(f"   - Geolife: {geo_avg_predictability:.3f}")
    findings.append(f"   - DIY: {diy_avg_predictability:.3f}")
    
    findings.append("")
    findings.append("INTERPRETATION:")
    if geo_agg['avg_target_in_history_rate'] > diy_agg['avg_target_in_history_rate']:
        findings.append("  - Geolife users have more repetitive visit patterns")
        findings.append("  - Next location is more likely to be a recently visited place")
        findings.append("  - Pointer mechanism can effectively 'copy' from history")
    
    if geo_agg['avg_normalized_entropy'] < diy_agg['avg_normalized_entropy']:
        findings.append("  - Geolife users have more concentrated location distributions")
        findings.append("  - Fewer locations dominate their mobility")
    
    if geo_agg['avg_top_1_concentration'] > diy_agg['avg_top_1_concentration']:
        findings.append("  - Geolife users have stronger home/work anchor points")
    
    for finding in findings:
        print(finding)
    
    with open(os.path.join(RESULTS_DIR, "04_key_findings.txt"), 'w') as f:
        f.write("User Behavior Key Findings\n")
        f.write("="*50 + "\n\n")
        for finding in findings:
            f.write(finding + "\n")
    
    print(f"\n✓ Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

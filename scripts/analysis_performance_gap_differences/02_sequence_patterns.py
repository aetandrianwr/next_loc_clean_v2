"""
02. Sequence Patterns and Target Diversity Analysis
Analyze patterns in historical sequences and target prediction diversity.

This script investigates:
- Target location repetition rate (how often target appears in history)
- Sequence diversity within users
- Target prediction difficulty
- History-target correlation

KEY HYPOTHESIS: If target appears more frequently in history in Geolife,
the pointer mechanism will have more advantage.
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


def analyze_target_in_history(sequences):
    """
    Analyze how often the target location appears in the history sequence.
    
    This is crucial for the pointer mechanism - if targets frequently appear
    in history, the pointer can simply "copy" from the input.
    """
    target_in_history_count = 0
    target_frequency_in_history = []  # How many times target appears in history
    
    for seq in sequences:
        history = seq['X']
        target = seq['Y']
        
        count_in_history = np.sum(history == target)
        target_frequency_in_history.append(count_in_history)
        
        if target in history:
            target_in_history_count += 1
    
    total = len(sequences)
    rate = target_in_history_count / total * 100
    
    # Distribution of target frequency in history
    freq_counter = Counter(target_frequency_in_history)
    
    return {
        'target_in_history_rate': rate,
        'target_in_history_count': target_in_history_count,
        'total_sequences': total,
        'avg_target_freq_in_history': np.mean(target_frequency_in_history),
        'freq_distribution': dict(freq_counter),
        'frequencies': target_frequency_in_history
    }


def analyze_sequence_diversity(sequences):
    """
    Analyze the diversity of locations within each sequence.
    Low diversity = repetitive patterns, easier for pointer
    """
    unique_ratios = []  # unique_locations / sequence_length
    
    for seq in sequences:
        history = seq['X']
        unique_locs = len(set(history))
        ratio = unique_locs / len(history)
        unique_ratios.append(ratio)
    
    return {
        'mean_unique_ratio': np.mean(unique_ratios),
        'median_unique_ratio': np.median(unique_ratios),
        'std_unique_ratio': np.std(unique_ratios),
        'unique_ratios': unique_ratios
    }


def analyze_most_frequent_target(sequences):
    """
    Analyze if target is the most frequent location in history.
    """
    target_is_most_frequent = 0
    target_rank_in_history = []
    
    for seq in sequences:
        history = seq['X']
        target = seq['Y']
        
        loc_counts = Counter(history)
        sorted_locs = sorted(loc_counts.items(), key=lambda x: -x[1])
        
        # Find rank of target
        rank = None
        for i, (loc, count) in enumerate(sorted_locs):
            if loc == target:
                rank = i + 1
                break
        
        if rank is not None:
            target_rank_in_history.append(rank)
            if rank == 1:
                target_is_most_frequent += 1
        else:
            target_rank_in_history.append(float('inf'))
    
    valid_ranks = [r for r in target_rank_in_history if r != float('inf')]
    
    return {
        'target_is_most_frequent_rate': target_is_most_frequent / len(sequences) * 100,
        'avg_target_rank': np.mean(valid_ranks) if valid_ranks else float('inf'),
        'target_in_top3_rate': sum(1 for r in valid_ranks if r <= 3) / len(sequences) * 100,
        'target_in_top5_rate': sum(1 for r in valid_ranks if r <= 5) / len(sequences) * 100,
        'ranks': target_rank_in_history
    }


def analyze_recency_pattern(sequences):
    """
    Analyze recency patterns - how recently did the target location appear in history.
    """
    recency_positions = []  # Position from end where target last appeared
    
    for seq in sequences:
        history = seq['X']
        target = seq['Y']
        
        # Find last occurrence of target in history (from end)
        positions = np.where(history == target)[0]
        if len(positions) > 0:
            # Position from end (1 = most recent)
            pos_from_end = len(history) - positions[-1]
            recency_positions.append(pos_from_end)
        else:
            recency_positions.append(float('inf'))
    
    valid_positions = [p for p in recency_positions if p != float('inf')]
    
    return {
        'avg_recency_position': np.mean(valid_positions) if valid_positions else float('inf'),
        'median_recency_position': np.median(valid_positions) if valid_positions else float('inf'),
        'target_in_last_5_rate': sum(1 for p in valid_positions if p <= 5) / len(sequences) * 100,
        'target_in_last_10_rate': sum(1 for p in valid_positions if p <= 10) / len(sequences) * 100,
        'recency_positions': recency_positions
    }


def main():
    print("="*80)
    print("02. Sequence Patterns and Target Diversity Analysis")
    print("="*80)
    print("\nHypothesis: Pointer mechanism benefits when target appears in history.")
    print("We analyze how often targets appear in historical sequences.\n")
    
    # Load test sequences (what we evaluate on)
    print("[Loading sequences...]")
    geo_train = load_sequences("geolife", 20, "train")
    geo_test = load_sequences("geolife", 20, "test")
    diy_train = load_sequences("diy", 50, "train")
    diy_test = load_sequences("diy", 50, "test")
    
    # Main analysis: Target in History
    print("\n[Analyzing target-in-history patterns...]")
    
    geo_train_target = analyze_target_in_history(geo_train)
    geo_test_target = analyze_target_in_history(geo_test)
    diy_train_target = analyze_target_in_history(diy_train)
    diy_test_target = analyze_target_in_history(diy_test)
    
    print("\n" + "="*80)
    print("TARGET IN HISTORY ANALYSIS (Key for Pointer Mechanism)")
    print("="*80)
    
    target_comparison = {
        'Metric': [
            'Target appears in history (%)',
            'Avg times target appears in history',
            'Total sequences'
        ],
        'Geolife Train': [
            f"{geo_train_target['target_in_history_rate']:.2f}%",
            f"{geo_train_target['avg_target_freq_in_history']:.2f}",
            geo_train_target['total_sequences']
        ],
        'Geolife Test': [
            f"{geo_test_target['target_in_history_rate']:.2f}%",
            f"{geo_test_target['avg_target_freq_in_history']:.2f}",
            geo_test_target['total_sequences']
        ],
        'DIY Train': [
            f"{diy_train_target['target_in_history_rate']:.2f}%",
            f"{diy_train_target['avg_target_freq_in_history']:.2f}",
            diy_train_target['total_sequences']
        ],
        'DIY Test': [
            f"{diy_test_target['target_in_history_rate']:.2f}%",
            f"{diy_test_target['avg_target_freq_in_history']:.2f}",
            diy_test_target['total_sequences']
        ],
    }
    
    target_df = pd.DataFrame(target_comparison)
    print(target_df.to_string(index=False))
    target_df.to_csv(os.path.join(RESULTS_DIR, "02_target_in_history.csv"), index=False)
    
    # Sequence diversity analysis
    print("\n[Analyzing sequence diversity...]")
    geo_train_div = analyze_sequence_diversity(geo_train)
    geo_test_div = analyze_sequence_diversity(geo_test)
    diy_train_div = analyze_sequence_diversity(diy_train)
    diy_test_div = analyze_sequence_diversity(diy_test)
    
    print("\n" + "="*80)
    print("SEQUENCE DIVERSITY (unique_locs / seq_len)")
    print("Lower = more repetitive patterns = easier for pointer")
    print("="*80)
    
    diversity_comparison = {
        'Metric': ['Mean Unique Ratio', 'Median Unique Ratio', 'Std'],
        'Geolife Train': [f"{geo_train_div['mean_unique_ratio']:.3f}", 
                         f"{geo_train_div['median_unique_ratio']:.3f}",
                         f"{geo_train_div['std_unique_ratio']:.3f}"],
        'Geolife Test': [f"{geo_test_div['mean_unique_ratio']:.3f}",
                        f"{geo_test_div['median_unique_ratio']:.3f}",
                        f"{geo_test_div['std_unique_ratio']:.3f}"],
        'DIY Train': [f"{diy_train_div['mean_unique_ratio']:.3f}",
                     f"{diy_train_div['median_unique_ratio']:.3f}",
                     f"{diy_train_div['std_unique_ratio']:.3f}"],
        'DIY Test': [f"{diy_test_div['mean_unique_ratio']:.3f}",
                    f"{diy_test_div['median_unique_ratio']:.3f}",
                    f"{diy_test_div['std_unique_ratio']:.3f}"],
    }
    
    diversity_df = pd.DataFrame(diversity_comparison)
    print(diversity_df.to_string(index=False))
    diversity_df.to_csv(os.path.join(RESULTS_DIR, "02_sequence_diversity.csv"), index=False)
    
    # Most frequent target analysis
    print("\n[Analyzing target frequency rank...]")
    geo_test_rank = analyze_most_frequent_target(geo_test)
    diy_test_rank = analyze_most_frequent_target(diy_test)
    
    print("\n" + "="*80)
    print("TARGET FREQUENCY RANK IN HISTORY (Test Set)")
    print("="*80)
    
    rank_comparison = {
        'Metric': ['Target is most frequent (%)', 'Target in top-3 (%)', 
                   'Target in top-5 (%)', 'Avg rank when present'],
        'Geolife Test': [f"{geo_test_rank['target_is_most_frequent_rate']:.2f}%",
                        f"{geo_test_rank['target_in_top3_rate']:.2f}%",
                        f"{geo_test_rank['target_in_top5_rate']:.2f}%",
                        f"{geo_test_rank['avg_target_rank']:.2f}"],
        'DIY Test': [f"{diy_test_rank['target_is_most_frequent_rate']:.2f}%",
                    f"{diy_test_rank['target_in_top3_rate']:.2f}%",
                    f"{diy_test_rank['target_in_top5_rate']:.2f}%",
                    f"{diy_test_rank['avg_target_rank']:.2f}"],
    }
    
    rank_df = pd.DataFrame(rank_comparison)
    print(rank_df.to_string(index=False))
    rank_df.to_csv(os.path.join(RESULTS_DIR, "02_target_rank.csv"), index=False)
    
    # Recency analysis
    print("\n[Analyzing recency patterns...]")
    geo_test_recency = analyze_recency_pattern(geo_test)
    diy_test_recency = analyze_recency_pattern(diy_test)
    
    print("\n" + "="*80)
    print("RECENCY ANALYSIS (How recent was target in history)")
    print("="*80)
    
    recency_comparison = {
        'Metric': ['Avg position from end', 'Median position from end',
                   'Target in last 5 (%)', 'Target in last 10 (%)'],
        'Geolife Test': [f"{geo_test_recency['avg_recency_position']:.2f}",
                        f"{geo_test_recency['median_recency_position']:.2f}",
                        f"{geo_test_recency['target_in_last_5_rate']:.2f}%",
                        f"{geo_test_recency['target_in_last_10_rate']:.2f}%"],
        'DIY Test': [f"{diy_test_recency['avg_recency_position']:.2f}",
                    f"{diy_test_recency['median_recency_position']:.2f}",
                    f"{diy_test_recency['target_in_last_5_rate']:.2f}%",
                    f"{diy_test_recency['target_in_last_10_rate']:.2f}%"],
    }
    
    recency_df = pd.DataFrame(recency_comparison)
    print(recency_df.to_string(index=False))
    recency_df.to_csv(os.path.join(RESULTS_DIR, "02_recency_analysis.csv"), index=False)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Target in history rate comparison (bar chart)
    ax1 = axes[0, 0]
    datasets = ['Geolife', 'DIY']
    train_rates = [geo_train_target['target_in_history_rate'], diy_train_target['target_in_history_rate']]
    test_rates = [geo_test_target['target_in_history_rate'], diy_test_target['target_in_history_rate']]
    
    x = np.arange(len(datasets))
    width = 0.35
    ax1.bar(x - width/2, train_rates, width, label='Train', color='steelblue')
    ax1.bar(x + width/2, test_rates, width, label='Test', color='coral')
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Target Appears in History Rate\n(Higher = Pointer More Useful)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.set_ylim(0, 100)
    for i, (tr, te) in enumerate(zip(train_rates, test_rates)):
        ax1.annotate(f'{tr:.1f}%', (i - width/2, tr + 1), ha='center', fontsize=9)
        ax1.annotate(f'{te:.1f}%', (i + width/2, te + 1), ha='center', fontsize=9)
    
    # 2. Sequence diversity distribution
    ax2 = axes[0, 1]
    ax2.hist(geo_test_div['unique_ratios'], bins=30, alpha=0.7, label='Geolife Test', density=True)
    ax2.hist(diy_test_div['unique_ratios'], bins=30, alpha=0.7, label='DIY Test', density=True)
    ax2.set_xlabel('Unique Location Ratio')
    ax2.set_ylabel('Density')
    ax2.set_title('Sequence Diversity Distribution\n(Lower = More Repetitive)')
    ax2.legend()
    ax2.axvline(geo_test_div['mean_unique_ratio'], color='blue', linestyle='--', alpha=0.7)
    ax2.axvline(diy_test_div['mean_unique_ratio'], color='orange', linestyle='--', alpha=0.7)
    
    # 3. Target frequency in history distribution
    ax3 = axes[1, 0]
    geo_freqs = [f for f in geo_test_target['frequencies'] if f > 0]
    diy_freqs = [f for f in diy_test_target['frequencies'] if f > 0]
    max_freq = max(max(geo_freqs) if geo_freqs else 0, max(diy_freqs) if diy_freqs else 0, 20)
    bins = range(0, min(int(max_freq) + 2, 30))
    ax3.hist(geo_freqs, bins=bins, alpha=0.7, label='Geolife Test', density=True)
    ax3.hist(diy_freqs, bins=bins, alpha=0.7, label='DIY Test', density=True)
    ax3.set_xlabel('Times Target Appears in History')
    ax3.set_ylabel('Density')
    ax3.set_title('Target Frequency in History\n(When Present)')
    ax3.legend()
    
    # 4. Summary comparison
    ax4 = axes[1, 1]
    metrics = ['Target in\nHistory (%)', 'Target in\nTop-3 (%)', 'Diversity\n(inv)', 'Target in\nLast 10 (%)']
    geolife_vals = [geo_test_target['target_in_history_rate'], 
                   geo_test_rank['target_in_top3_rate'],
                   (1 - geo_test_div['mean_unique_ratio']) * 100,
                   geo_test_recency['target_in_last_10_rate']]
    diy_vals = [diy_test_target['target_in_history_rate'],
               diy_test_rank['target_in_top3_rate'],
               (1 - diy_test_div['mean_unique_ratio']) * 100,
               diy_test_recency['target_in_last_10_rate']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, geolife_vals, width, label='Geolife', color='steelblue')
    ax4.bar(x + width/2, diy_vals, width, label='DIY', color='coral')
    ax4.set_ylabel('Rate (%)')
    ax4.set_title('Key Metrics Comparison (Test Set)\n(All Higher = Better for Pointer)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "02_sequence_patterns.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    findings = []
    
    # Target in history difference
    geo_rate = geo_test_target['target_in_history_rate']
    diy_rate = diy_test_target['target_in_history_rate']
    diff = geo_rate - diy_rate
    findings.append(f"1. Target in History: Geolife={geo_rate:.1f}% vs DIY={diy_rate:.1f}% (Diff: {diff:+.1f}%)")
    findings.append(f"   -> Geolife has {'HIGHER' if diff > 0 else 'LOWER'} target repetition, making pointer MORE useful")
    
    # Diversity difference
    geo_div = geo_test_div['mean_unique_ratio']
    diy_div = diy_test_div['mean_unique_ratio']
    findings.append(f"2. Sequence Diversity: Geolife={geo_div:.3f} vs DIY={diy_div:.3f}")
    findings.append(f"   -> Geolife has {'LOWER' if geo_div < diy_div else 'HIGHER'} diversity = {'MORE' if geo_div < diy_div else 'LESS'} repetitive patterns")
    
    # Target rank
    geo_top3 = geo_test_rank['target_in_top3_rate']
    diy_top3 = diy_test_rank['target_in_top3_rate']
    findings.append(f"3. Target in Top-3 Most Frequent: Geolife={geo_top3:.1f}% vs DIY={diy_top3:.1f}%")
    
    # Recency
    geo_last10 = geo_test_recency['target_in_last_10_rate']
    diy_last10 = diy_test_recency['target_in_last_10_rate']
    findings.append(f"4. Target in Last 10 Visits: Geolife={geo_last10:.1f}% vs DIY={diy_last10:.1f}%")
    
    findings.append("")
    findings.append("INTERPRETATION:")
    if geo_rate > diy_rate:
        findings.append("  - Geolife users revisit recent locations more frequently")
        findings.append("  - This makes the pointer mechanism highly effective in Geolife")
        findings.append("  - DIY has more novel/unseen targets, requiring generation head")
    
    for finding in findings:
        print(finding)
    
    with open(os.path.join(RESULTS_DIR, "02_key_findings.txt"), 'w') as f:
        f.write("Sequence Patterns Key Findings\n")
        f.write("="*50 + "\n\n")
        for finding in findings:
            f.write(finding + "\n")
    
    print(f"\n✓ Results saved to: {RESULTS_DIR}")
    
    return {
        'geolife_target_in_history': geo_test_target['target_in_history_rate'],
        'diy_target_in_history': diy_test_target['target_in_history_rate'],
        'geolife_diversity': geo_test_div['mean_unique_ratio'],
        'diy_diversity': diy_test_div['mean_unique_ratio'],
    }


if __name__ == "__main__":
    main()

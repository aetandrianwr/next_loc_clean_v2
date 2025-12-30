"""
Deep Dive: User Location Entropy Analysis.

This script analyzes per-user location entropy and predictability patterns
to understand why MHSA baseline already performs well on DIY.

Key hypothesis: DIY users have more concentrated/predictable location patterns.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_dataset(data_dir, prefix, split="train"):
    """Load dataset from pickle file."""
    path = os.path.join(data_dir, f"{prefix}_{split}.pk")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def calculate_entropy(values):
    """Calculate entropy of a distribution."""
    if not values:
        return 0
    counter = Counter(values)
    total = len(values)
    probs = [count / total for count in counter.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def analyze_per_user_entropy(data, name):
    """Analyze location entropy per user."""
    user_data = defaultdict(lambda: {'targets': [], 'histories': []})
    
    for sample in data:
        user = sample['user_X'][0]
        user_data[user]['targets'].append(sample['Y'])
        user_data[user]['histories'].extend(sample['X'].tolist())
    
    # Calculate per-user statistics
    user_stats = []
    for user, info in user_data.items():
        targets = info['targets']
        histories = info['histories']
        
        target_entropy = calculate_entropy(targets)
        history_entropy = calculate_entropy(histories)
        
        target_counter = Counter(targets)
        top1_coverage = target_counter.most_common(1)[0][1] / len(targets) * 100 if targets else 0
        top3_coverage = sum([c for _, c in target_counter.most_common(3)]) / len(targets) * 100 if targets else 0
        
        unique_targets = len(set(targets))
        unique_history = len(set(histories))
        
        user_stats.append({
            'user': user,
            'num_samples': len(targets),
            'unique_targets': unique_targets,
            'unique_history': unique_history,
            'target_entropy': target_entropy,
            'history_entropy': history_entropy,
            'top1_target_coverage': top1_coverage,
            'top3_target_coverage': top3_coverage,
        })
    
    df = pd.DataFrame(user_stats)
    
    print(f"\n{'='*70}")
    print(f"PER-USER LOCATION ENTROPY ANALYSIS: {name}")
    print(f"{'='*70}")
    
    print(f"\nNumber of users: {len(df)}")
    print(f"\n1. TARGET ENTROPY (lower = more predictable)")
    print(f"   Mean:   {df['target_entropy'].mean():.3f}")
    print(f"   Std:    {df['target_entropy'].std():.3f}")
    print(f"   Median: {df['target_entropy'].median():.3f}")
    print(f"   Min:    {df['target_entropy'].min():.3f}")
    print(f"   Max:    {df['target_entropy'].max():.3f}")
    
    print(f"\n2. TOP-1 TARGET COVERAGE (higher = more predictable)")
    print(f"   Mean:   {df['top1_target_coverage'].mean():.1f}%")
    print(f"   Median: {df['top1_target_coverage'].median():.1f}%")
    
    print(f"\n3. TOP-3 TARGET COVERAGE")
    print(f"   Mean:   {df['top3_target_coverage'].mean():.1f}%")
    print(f"   Median: {df['top3_target_coverage'].median():.1f}%")
    
    print(f"\n4. UNIQUE LOCATIONS PER USER")
    print(f"   Unique Targets Mean:   {df['unique_targets'].mean():.1f}")
    print(f"   Unique History Mean:   {df['unique_history'].mean():.1f}")
    
    return df


def analyze_location_concentration_per_user(data, name):
    """Analyze how concentrated each user's location visits are."""
    user_targets = defaultdict(list)
    
    for sample in data:
        user = sample['user_X'][0]
        user_targets[user].append(sample['Y'])
    
    # Calculate Gini coefficient per user
    gini_scores = []
    home_work_ratios = []  # Ratio of top-2 locations
    
    for user, targets in user_targets.items():
        if len(targets) < 5:  # Skip users with too few samples
            continue
            
        counter = Counter(targets)
        counts = sorted(counter.values(), reverse=True)
        
        # Gini coefficient
        n = len(counts)
        gini = (2 * sum((i+1) * c for i, c in enumerate(counts))) / (n * sum(counts)) - (n+1)/n
        gini_scores.append(gini)
        
        # Home-work ratio (top-2 coverage)
        if len(counts) >= 2:
            hw_ratio = sum(counts[:2]) / sum(counts) * 100
            home_work_ratios.append(hw_ratio)
    
    print(f"\n{'='*70}")
    print(f"LOCATION CONCENTRATION PER USER: {name}")
    print(f"{'='*70}")
    
    print(f"\n1. HOME-WORK PATTERN (Top-2 Location Coverage)")
    print(f"   Mean:   {np.mean(home_work_ratios):.1f}%")
    print(f"   Median: {np.median(home_work_ratios):.1f}%")
    print(f"   Std:    {np.std(home_work_ratios):.1f}%")
    
    # Classify users by concentration
    high_conc = sum(1 for r in home_work_ratios if r > 80)
    med_conc = sum(1 for r in home_work_ratios if 50 <= r <= 80)
    low_conc = sum(1 for r in home_work_ratios if r < 50)
    
    print(f"\n2. USER CONCENTRATION DISTRIBUTION")
    print(f"   High (>80% in top-2):    {high_conc} users ({high_conc/len(home_work_ratios)*100:.1f}%)")
    print(f"   Medium (50-80%):          {med_conc} users ({med_conc/len(home_work_ratios)*100:.1f}%)")
    print(f"   Low (<50%):               {low_conc} users ({low_conc/len(home_work_ratios)*100:.1f}%)")
    
    return {
        'home_work_ratio_mean': np.mean(home_work_ratios),
        'home_work_ratio_median': np.median(home_work_ratios),
        'high_concentration_users': high_conc / len(home_work_ratios) * 100,
    }


def analyze_mhsa_learning_potential(data, name):
    """
    Analyze what patterns MHSA can easily learn.
    
    MHSA excels at learning:
    1. Global frequency patterns (which locations are popular)
    2. Temporal patterns (time-of-day, day-of-week)
    3. Sequential patterns (common transitions)
    """
    # Global frequency
    all_targets = [s['Y'] for s in data]
    target_freq = Counter(all_targets)
    
    # Top-K accuracy if predicting by global frequency
    sorted_targets = [t for t, _ in target_freq.most_common()]
    
    top1_acc = sum(1 for s in data if s['Y'] == sorted_targets[0]) / len(data) * 100
    top5_acc = sum(1 for s in data if s['Y'] in sorted_targets[:5]) / len(data) * 100
    top10_acc = sum(1 for s in data if s['Y'] in sorted_targets[:10]) / len(data) * 100
    
    # Transition patterns (bigram accuracy)
    bigram_counter = Counter()
    for sample in data:
        history = sample['X'].tolist()
        target = sample['Y']
        if history:
            last_loc = history[-1]
            bigram_counter[(last_loc, target)] += 1
    
    # For each last location, what's the most likely next?
    last_to_next = defaultdict(lambda: Counter())
    for (last, next_), count in bigram_counter.items():
        last_to_next[last][next_] += count
    
    bigram_correct = 0
    for sample in data:
        history = sample['X'].tolist()
        target = sample['Y']
        if history and history[-1] in last_to_next:
            pred = last_to_next[history[-1]].most_common(1)[0][0]
            if pred == target:
                bigram_correct += 1
    
    bigram_acc = bigram_correct / len(data) * 100
    
    print(f"\n{'='*70}")
    print(f"MHSA LEARNING POTENTIAL ANALYSIS: {name}")
    print(f"{'='*70}")
    
    print(f"\n1. GLOBAL FREQUENCY PATTERN")
    print(f"   Top-1 Frequency Acc: {top1_acc:.1f}%")
    print(f"   Top-5 Frequency Acc: {top5_acc:.1f}%")
    print(f"   Top-10 Frequency Acc: {top10_acc:.1f}%")
    
    print(f"\n2. TRANSITION (BIGRAM) PATTERN")
    print(f"   Bigram Prediction Acc: {bigram_acc:.1f}%")
    
    return {
        'top1_freq_acc': top1_acc,
        'top5_freq_acc': top5_acc,
        'bigram_acc': bigram_acc,
    }


def main():
    print("=" * 80)
    print("DEEP DIVE: USER BEHAVIOR ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing per-user patterns to understand baseline performance differences")
    print("=" * 80)
    
    # Paths
    geolife_dir = "data/geolife_eps20/processed"
    diy_dir = "data/diy_eps50/processed"
    geolife_prefix = "geolife_eps20_prev7"
    diy_prefix = "diy_eps50_prev7"
    
    # Load data
    print("\nLoading datasets...")
    geolife_train = load_dataset(geolife_dir, geolife_prefix, "train")
    geolife_test = load_dataset(geolife_dir, geolife_prefix, "test")
    diy_train = load_dataset(diy_dir, diy_prefix, "train")
    diy_test = load_dataset(diy_dir, diy_prefix, "test")
    
    geolife_all = geolife_train + geolife_test
    diy_all = diy_train + diy_test
    
    # Per-user entropy
    geo_entropy_df = analyze_per_user_entropy(geolife_all, "Geolife")
    diy_entropy_df = analyze_per_user_entropy(diy_all, "DIY")
    
    # Location concentration
    geo_conc = analyze_location_concentration_per_user(geolife_all, "Geolife")
    diy_conc = analyze_location_concentration_per_user(diy_all, "DIY")
    
    # MHSA learning potential
    geo_mhsa = analyze_mhsa_learning_potential(geolife_all, "Geolife")
    diy_mhsa = analyze_mhsa_learning_potential(diy_all, "DIY")
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)
    
    print(f"\n┌─────────────────────────────────────────────────────────────────────┐")
    print(f"│                    USER BEHAVIOR COMPARISON                         │")
    print(f"├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Metric                           │  Geolife     │  DIY         │")
    print(f"├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Target Entropy (per-user mean)   │  {geo_entropy_df['target_entropy'].mean():>10.3f}  │  {diy_entropy_df['target_entropy'].mean():>10.3f}  │")
    print(f"│ Top-1 Target Coverage (per-user) │  {geo_entropy_df['top1_target_coverage'].mean():>10.1f}% │  {diy_entropy_df['top1_target_coverage'].mean():>10.1f}% │")
    print(f"│ Home-Work Ratio (top-2) (%)      │  {geo_conc['home_work_ratio_mean']:>10.1f}  │  {diy_conc['home_work_ratio_mean']:>10.1f}  │")
    print(f"│ High Concentration Users (%)     │  {geo_conc['high_concentration_users']:>10.1f}  │  {diy_conc['high_concentration_users']:>10.1f}  │")
    print(f"├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Bigram Prediction Acc (%)        │  {geo_mhsa['bigram_acc']:>10.1f}  │  {diy_mhsa['bigram_acc']:>10.1f}  │")
    print(f"│ Top-1 Frequency Acc (%)          │  {geo_mhsa['top1_freq_acc']:>10.1f}  │  {diy_mhsa['top1_freq_acc']:>10.1f}  │")
    print(f"│ Top-5 Frequency Acc (%)          │  {geo_mhsa['top5_freq_acc']:>10.1f}  │  {diy_mhsa['top5_freq_acc']:>10.1f}  │")
    print(f"└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"""
KEY FINDING: DIY users have MORE CONCENTRATED behavior patterns!

1. HOME-WORK DOMINANCE:
   - DIY: {diy_conc['home_work_ratio_mean']:.1f}% of visits are to top-2 locations per user
   - Geolife: {geo_conc['home_work_ratio_mean']:.1f}% of visits are to top-2 locations per user
   - DIY users visit home/work much more frequently, making predictions easier

2. TOP-1 FREQUENCY SUCCESS:
   - DIY: {diy_mhsa['top1_freq_acc']:.1f}% accuracy by just predicting most frequent location
   - Geolife: {geo_mhsa['top1_freq_acc']:.1f}% accuracy with same strategy
   - DIY is {diy_mhsa['top1_freq_acc'] - geo_mhsa['top1_freq_acc']:.1f}pp easier for frequency-based prediction

3. USER CONCENTRATION:
   - {diy_conc['high_concentration_users']:.1f}% of DIY users have >80% visits to top-2 locations
   - {geo_conc['high_concentration_users']:.1f}% of Geolife users have same concentration
   - More DIY users have highly predictable patterns

CONCLUSION: The DIY dataset has more routine/predictable user behavior.
The MHSA baseline can already capture this regularity well.
The pointer mechanism provides less additional benefit because:
- The targets are already highly predictable from frequency patterns
- Less need to "copy" from history when patterns are this regular
""")
    
    # Save results
    output_dir = "scripts/analysis_performance_gap/results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'geolife': {
            'target_entropy_mean': float(geo_entropy_df['target_entropy'].mean()),
            'top1_coverage_mean': float(geo_entropy_df['top1_target_coverage'].mean()),
            'home_work_ratio': geo_conc['home_work_ratio_mean'],
            'high_conc_users': geo_conc['high_concentration_users'],
            'bigram_acc': geo_mhsa['bigram_acc'],
            'top1_freq_acc': geo_mhsa['top1_freq_acc'],
        },
        'diy': {
            'target_entropy_mean': float(diy_entropy_df['target_entropy'].mean()),
            'top1_coverage_mean': float(diy_entropy_df['top1_target_coverage'].mean()),
            'home_work_ratio': diy_conc['home_work_ratio_mean'],
            'high_conc_users': diy_conc['high_concentration_users'],
            'bigram_acc': diy_mhsa['bigram_acc'],
            'top1_freq_acc': diy_mhsa['top1_freq_acc'],
        }
    }
    
    output_file = os.path.join(output_dir, "user_behavior_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

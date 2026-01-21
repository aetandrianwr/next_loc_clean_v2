"""
Baseline Saturation Analysis.

This script analyzes whether the DIY dataset has "baseline saturation" -
a scenario where the baseline model (MHSA) is already performing close to
practical limits, leaving little room for improvement.

Key analyses:
1. Frequency-based baseline accuracy
2. Target entropy and predictability
3. MHSA performance ceiling analysis
4. Headroom analysis for improvement
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
    counter = Counter(values)
    total = len(values)
    probs = [count / total for count in counter.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def simulate_frequency_baseline(train_data, test_data, name):
    """
    Simulate a simple frequency baseline that predicts the most common location.
    
    This helps understand the "floor" of prediction difficulty.
    """
    # Build frequency model from training data
    user_loc_freq = defaultdict(lambda: Counter())
    global_freq = Counter()
    
    for sample in train_data:
        user = sample['user_X'][0]
        target = sample['Y']
        user_loc_freq[user][target] += 1
        global_freq[target] += 1
    
    # Evaluate on test data
    user_correct = 0
    global_correct = 0
    total = len(test_data)
    
    for sample in test_data:
        user = sample['user_X'][0]
        target = sample['Y']
        
        # User-level most frequent
        if user_loc_freq[user]:
            user_pred = user_loc_freq[user].most_common(1)[0][0]
            if user_pred == target:
                user_correct += 1
        
        # Global most frequent
        global_pred = global_freq.most_common(1)[0][0]
        if global_pred == target:
            global_correct += 1
    
    print(f"\n{'='*70}")
    print(f"FREQUENCY BASELINE ANALYSIS: {name}")
    print(f"{'='*70}")
    print(f"  Global Frequency Baseline Acc@1: {global_correct/total*100:.2f}%")
    print(f"  Per-User Frequency Baseline Acc@1: {user_correct/total*100:.2f}%")
    
    return {
        'global_baseline': global_correct / total * 100,
        'user_baseline': user_correct / total * 100,
    }


def simulate_history_frequency_baseline(test_data, name):
    """
    Predict the most frequent location from the user's current history sequence.
    This is closer to what pointer mechanism could achieve with simple frequency.
    """
    correct = 0
    total = len(test_data)
    
    for sample in test_data:
        history = sample['X'].tolist()
        target = sample['Y']
        
        # Most frequent in history
        hist_counter = Counter(history)
        pred = hist_counter.most_common(1)[0][0]
        
        if pred == target:
            correct += 1
    
    acc = correct / total * 100
    print(f"\n  History Frequency Baseline Acc@1: {acc:.2f}%")
    
    return acc


def simulate_recency_baseline(test_data, name):
    """
    Predict the most recent location as next location.
    Tests the strong recency bias in mobility patterns.
    """
    correct = 0
    total = len(test_data)
    
    for sample in test_data:
        history = sample['X'].tolist()
        target = sample['Y']
        
        # Most recent
        pred = history[-1]
        
        if pred == target:
            correct += 1
    
    acc = correct / total * 100
    print(f"  Recency Baseline (last location) Acc@1: {acc:.2f}%")
    
    return acc


def analyze_prediction_headroom(test_data, name, actual_mhsa_acc, actual_pointer_acc):
    """
    Analyze the theoretical headroom for improvement.
    """
    # Calculate oracle pointer accuracy (upper bound for pointer)
    pointer_oracle = 0
    for sample in test_data:
        if sample['Y'] in sample['X'].tolist():
            pointer_oracle += 1
    pointer_oracle_acc = pointer_oracle / len(test_data) * 100
    
    # Perfect prediction is 100%
    perfect = 100.0
    
    print(f"\n{'='*70}")
    print(f"PREDICTION HEADROOM ANALYSIS: {name}")
    print(f"{'='*70}")
    print(f"\n  MHSA Baseline Accuracy:      {actual_mhsa_acc:.2f}%")
    print(f"  Pointer Generator Transformer Accuracy:        {actual_pointer_acc:.2f}%")
    print(f"  Improvement Achieved:        +{actual_pointer_acc - actual_mhsa_acc:.2f}pp")
    print(f"\n  Pointer Oracle (upper bound): {pointer_oracle_acc:.2f}%")
    print(f"  Perfect Accuracy:             {perfect:.2f}%")
    
    # Calculate headroom
    mhsa_headroom = perfect - actual_mhsa_acc
    pointer_headroom = pointer_oracle_acc - actual_mhsa_acc
    
    print(f"\n  Theoretical Headroom (to 100%): {mhsa_headroom:.2f}pp")
    print(f"  Pointer-Achievable Headroom:    {pointer_headroom:.2f}pp")
    print(f"  Improvement Efficiency:         {(actual_pointer_acc - actual_mhsa_acc) / pointer_headroom * 100:.1f}%")
    
    return {
        'mhsa_acc': actual_mhsa_acc,
        'pointer_acc': actual_pointer_acc,
        'improvement': actual_pointer_acc - actual_mhsa_acc,
        'pointer_oracle': pointer_oracle_acc,
        'headroom': mhsa_headroom,
        'pointer_headroom': pointer_headroom,
    }


def analyze_target_concentration(train_data, test_data, name):
    """
    Analyze how concentrated target predictions are.
    High concentration means easier prediction but less room for improvement.
    """
    # Combine for analysis
    all_data = train_data + test_data
    targets = [sample['Y'] for sample in all_data]
    target_counter = Counter(targets)
    
    total = len(targets)
    num_unique = len(target_counter)
    
    # Top-K accuracy (if we always predict top-K most common)
    sorted_counts = sorted(target_counter.values(), reverse=True)
    
    top_k_coverage = {}
    for k in [1, 5, 10, 20, 50, 100]:
        if k <= len(sorted_counts):
            coverage = sum(sorted_counts[:k]) / total * 100
            top_k_coverage[k] = coverage
    
    # Gini coefficient (measure of inequality)
    sorted_probs = np.array(sorted_counts) / total
    n = len(sorted_probs)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_probs)) / (n * np.sum(sorted_probs)) - (n+1)/n
    
    print(f"\n{'='*70}")
    print(f"TARGET CONCENTRATION ANALYSIS: {name}")
    print(f"{'='*70}")
    print(f"\n  Unique Targets: {num_unique}")
    print(f"  Gini Coefficient: {gini:.3f} (0=equal, 1=concentrated)")
    print(f"\n  Top-K Target Coverage:")
    for k, cov in top_k_coverage.items():
        print(f"    Top-{k:3d}: {cov:.1f}%")
    
    # Calculate entropy
    entropy = calculate_entropy(targets)
    max_entropy = np.log2(num_unique)
    normalized_entropy = entropy / max_entropy
    
    print(f"\n  Target Entropy: {entropy:.2f} bits (normalized: {normalized_entropy:.3f})")
    
    return {
        'num_unique': num_unique,
        'gini': gini,
        'top_k_coverage': top_k_coverage,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
    }


def main():
    print("=" * 80)
    print("BASELINE SATURATION ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing whether DIY dataset shows baseline saturation,")
    print("which would explain limited improvement from PGT.")
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
    
    # Actual model results (from experiments)
    # Geolife: MHSA 33.18%, PGT 53.96% (improvement: +20.78pp)
    # DIY: MHSA 53.17%, PGT 56.88% (improvement: +3.71pp)
    geolife_mhsa = 33.18
    geolife_pointer = 53.96  # Using best available result approximation
    diy_mhsa = 53.17
    diy_pointer = 56.88  # Using best available result approximation
    
    # Frequency baselines
    geo_freq = simulate_frequency_baseline(geolife_train, geolife_test, "Geolife")
    geo_hist_freq = simulate_history_frequency_baseline(geolife_test, "Geolife")
    geo_recency = simulate_recency_baseline(geolife_test, "Geolife")
    
    diy_freq = simulate_frequency_baseline(diy_train, diy_test, "DIY")
    diy_hist_freq = simulate_history_frequency_baseline(diy_test, "DIY")
    diy_recency = simulate_recency_baseline(diy_test, "DIY")
    
    # Target concentration
    geo_conc = analyze_target_concentration(geolife_train, geolife_test, "Geolife")
    diy_conc = analyze_target_concentration(diy_train, diy_test, "DIY")
    
    # Headroom analysis
    geo_headroom = analyze_prediction_headroom(geolife_test, "Geolife", geolife_mhsa, geolife_pointer)
    diy_headroom = analyze_prediction_headroom(diy_test, "DIY", diy_mhsa, diy_pointer)
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)
    
    print(f"\n┌─────────────────────────────────────────────────────────────────────┐")
    print(f"│                   BASELINE SATURATION COMPARISON                   │")
    print(f"├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Metric                           │  Geolife     │  DIY         │")
    print(f"├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ MHSA Baseline Acc@1 (%)          │  {geolife_mhsa:>10.2f}  │  {diy_mhsa:>10.2f}  │")
    print(f"│ PGT Acc@1 (%)             │  {geolife_pointer:>10.2f}  │  {diy_pointer:>10.2f}  │")
    print(f"│ Improvement (pp)                 │  {geolife_pointer-geolife_mhsa:>10.2f}  │  {diy_pointer-diy_mhsa:>10.2f}  │")
    print(f"├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Per-User Freq Baseline (%)       │  {geo_freq['user_baseline']:>10.2f}  │  {diy_freq['user_baseline']:>10.2f}  │")
    print(f"│ History Freq Baseline (%)        │  {geo_hist_freq:>10.2f}  │  {diy_hist_freq:>10.2f}  │")
    print(f"│ Recency Baseline (%)             │  {geo_recency:>10.2f}  │  {diy_recency:>10.2f}  │")
    print(f"├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Pointer Oracle (upper bound) (%) │  {geo_headroom['pointer_oracle']:>10.2f}  │  {diy_headroom['pointer_oracle']:>10.2f}  │")
    print(f"│ Pointer-Achievable Headroom (pp) │  {geo_headroom['pointer_headroom']:>10.2f}  │  {diy_headroom['pointer_headroom']:>10.2f}  │")
    print(f"├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Target Concentration (Gini)      │  {geo_conc['gini']:>10.3f}  │  {diy_conc['gini']:>10.3f}  │")
    print(f"│ Target Entropy (normalized)      │  {geo_conc['normalized_entropy']:>10.3f}  │  {diy_conc['normalized_entropy']:>10.3f}  │")
    print(f"└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"""
The analysis reveals several key insights:

1. BASELINE PERFORMANCE GAP:
   - DIY MHSA baseline ({diy_mhsa:.1f}%) is already {diy_mhsa - geolife_mhsa:.1f}pp higher than Geolife ({geolife_mhsa:.1f}%)
   - This suggests DIY is an "easier" prediction task for the baseline

2. IMPROVEMENT HEADROOM:
   - Geolife pointer-achievable headroom: {geo_headroom['pointer_headroom']:.1f}pp
   - DIY pointer-achievable headroom: {diy_headroom['pointer_headroom']:.1f}pp
   - Geolife has {geo_headroom['pointer_headroom'] - diy_headroom['pointer_headroom']:.1f}pp MORE room for pointer improvement

3. BASELINE SATURATION EVIDENCE:
   - DIY's higher baselines (frequency, recency) indicate more predictable patterns
   - DIY has higher target concentration (top locations dominate predictions)
   - The MHSA already captures much of this predictability on DIY

4. POINTER ORACLE ANALYSIS:
   - On Geolife, pointer could theoretically reach {geo_headroom['pointer_oracle']:.1f}% (copy-from-history cases)
   - On DIY, pointer could theoretically reach {diy_headroom['pointer_oracle']:.1f}%
   - Geolife has more "copyable" targets, benefiting pointer more
""")
    
    # Save results
    output_dir = "scripts/analysis_performance_gap/results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'geolife': {
            'mhsa_acc': geolife_mhsa,
            'pointer_acc': geolife_pointer,
            'freq_baseline': geo_freq,
            'hist_freq': geo_hist_freq,
            'recency': geo_recency,
            'headroom': geo_headroom,
            'concentration': geo_conc,
        },
        'diy': {
            'mhsa_acc': diy_mhsa,
            'pointer_acc': diy_pointer,
            'freq_baseline': diy_freq,
            'hist_freq': diy_hist_freq,
            'recency': diy_recency,
            'headroom': diy_headroom,
            'concentration': diy_conc,
        }
    }
    
    output_file = os.path.join(output_dir, "baseline_saturation.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

"""
Pointer Mechanism Effectiveness Analysis.

This script analyzes how well the pointer mechanism works on both datasets
by simulating pointer-based predictions and measuring their accuracy.

Key analyses:
1. Pointer hit rate - how often target is in history
2. Pointer rank - when target is in history, at what attention rank
3. History size vs pointer effectiveness
4. User-level pointer effectiveness
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


def analyze_pointer_effectiveness(data, name):
    """
    Analyze how effective a perfect pointer mechanism would be.
    
    Returns detailed statistics about:
    - Target-in-history rate (pointer hit rate)
    - Position of target in history (recency analysis)
    - Frequency of target in history (repetition patterns)
    """
    results = {
        'target_in_history': 0,
        'target_not_in_history': 0,
        'target_positions': [],  # positions from end where target found
        'target_frequencies': [],  # how many times target appears in history
        'target_in_last_n': {1: 0, 3: 0, 5: 0, 10: 0, 20: 0},
        'history_sizes': [],
        'unique_locs_in_history': [],
    }
    
    for sample in data:
        history = sample['X'].tolist()
        target = sample['Y']
        
        results['history_sizes'].append(len(history))
        results['unique_locs_in_history'].append(len(set(history)))
        
        if target in history:
            results['target_in_history'] += 1
            
            # Find all positions (from end)
            positions = [i + 1 for i, loc in enumerate(reversed(history)) if loc == target]
            results['target_positions'].append(positions[0])  # Closest position
            results['target_frequencies'].append(len(positions))
            
            # Check last-N positions
            for n in results['target_in_last_n'].keys():
                last_n = history[-n:] if len(history) >= n else history
                if target in last_n:
                    results['target_in_last_n'][n] += 1
        else:
            results['target_not_in_history'] += 1
    
    total = results['target_in_history'] + results['target_not_in_history']
    
    print(f"\n{'='*70}")
    print(f"POINTER MECHANISM EFFECTIVENESS ANALYSIS: {name}")
    print(f"{'='*70}")
    
    hit_rate = results['target_in_history'] / total * 100
    print(f"\n1. POINTER HIT RATE (Target in History)")
    print(f"   Target IN history:     {results['target_in_history']:,} ({hit_rate:.1f}%)")
    print(f"   Target NOT in history: {results['target_not_in_history']:,} ({100-hit_rate:.1f}%)")
    
    print(f"\n2. TARGET POSITION FROM END (when in history)")
    if results['target_positions']:
        positions = results['target_positions']
        print(f"   Mean position:   {np.mean(positions):.1f}")
        print(f"   Median position: {np.median(positions):.1f}")
        print(f"   Min/Max:         {np.min(positions)} / {np.max(positions)}")
    
    print(f"\n3. TARGET IN LAST N POSITIONS")
    for n, count in results['target_in_last_n'].items():
        rate = count / total * 100
        print(f"   Last-{n:2d}: {count:,} ({rate:.1f}%)")
    
    print(f"\n4. TARGET REPETITION IN HISTORY")
    if results['target_frequencies']:
        freqs = results['target_frequencies']
        print(f"   Mean frequency:   {np.mean(freqs):.2f}")
        print(f"   Median frequency: {np.median(freqs):.0f}")
        print(f"   Max frequency:    {np.max(freqs)}")
    
    print(f"\n5. HISTORY CHARACTERISTICS")
    print(f"   Mean history length:    {np.mean(results['history_sizes']):.1f}")
    print(f"   Mean unique locations:  {np.mean(results['unique_locs_in_history']):.1f}")
    
    return results


def analyze_pointer_vs_generation_potential(data, name):
    """
    Analyze when pointer vs generation should be preferred.
    
    The pointer mechanism is best for copy scenarios.
    The generation head is best for novel locations.
    """
    copy_cases = 0  # Target in history - pointer can help
    novel_cases = 0  # Target not in history - need generation
    
    # For copy cases, analyze how easy it is to copy
    copy_position_1 = 0  # Target is the most recent location
    copy_position_3 = 0  # Target in top-3 recent
    
    for sample in data:
        history = sample['X'].tolist()
        target = sample['Y']
        
        if target in history:
            copy_cases += 1
            
            # Check position
            rev_history = list(reversed(history))
            pos = rev_history.index(target) + 1
            
            if pos == 1:
                copy_position_1 += 1
            if pos <= 3:
                copy_position_3 += 1
        else:
            novel_cases += 1
    
    total = copy_cases + novel_cases
    
    print(f"\n{'='*70}")
    print(f"POINTER VS GENERATION ANALYSIS: {name}")
    print(f"{'='*70}")
    
    print(f"\nCOPY vs NOVEL SPLIT:")
    print(f"  Copy cases (target in history):     {copy_cases:,} ({copy_cases/total*100:.1f}%)")
    print(f"  Novel cases (target not in history): {novel_cases:,} ({novel_cases/total*100:.1f}%)")
    
    print(f"\nCOPY CASE DIFFICULTY:")
    if copy_cases > 0:
        print(f"  Target at position 1 (easiest):     {copy_position_1:,} ({copy_position_1/copy_cases*100:.1f}% of copy cases)")
        print(f"  Target in top-3 positions:          {copy_position_3:,} ({copy_position_3/copy_cases*100:.1f}% of copy cases)")
    
    return {
        'copy_cases': copy_cases,
        'novel_cases': novel_cases,
        'copy_rate': copy_cases / total * 100,
        'copy_position_1': copy_position_1,
        'copy_position_3': copy_position_3,
    }


def simulate_oracle_pointer(data, name):
    """
    Simulate an oracle pointer that always picks the correct position if possible.
    This gives upper bound of pointer mechanism accuracy.
    """
    oracle_correct = 0
    total = len(data)
    
    for sample in data:
        history = sample['X'].tolist()
        target = sample['Y']
        
        if target in history:
            oracle_correct += 1
    
    print(f"\n{'='*70}")
    print(f"ORACLE POINTER SIMULATION: {name}")
    print(f"{'='*70}")
    print(f"  Oracle Pointer Accuracy: {oracle_correct/total*100:.1f}%")
    print(f"  (Upper bound if pointer mechanism is perfect)")
    
    return oracle_correct / total * 100


def analyze_per_user_pointer_effectiveness(data, name):
    """Analyze pointer effectiveness per user."""
    user_stats = defaultdict(lambda: {'total': 0, 'in_history': 0})
    
    for sample in data:
        user = sample['user_X'][0]
        target = sample['Y']
        history = set(sample['X'].tolist())
        
        user_stats[user]['total'] += 1
        if target in history:
            user_stats[user]['in_history'] += 1
    
    # Calculate per-user hit rates
    hit_rates = []
    for user, stats in user_stats.items():
        if stats['total'] > 0:
            rate = stats['in_history'] / stats['total'] * 100
            hit_rates.append(rate)
    
    print(f"\n{'='*70}")
    print(f"PER-USER POINTER HIT RATE: {name}")
    print(f"{'='*70}")
    print(f"  Number of users: {len(hit_rates)}")
    print(f"  Mean hit rate:   {np.mean(hit_rates):.1f}%")
    print(f"  Std hit rate:    {np.std(hit_rates):.1f}%")
    print(f"  Min/Max:         {np.min(hit_rates):.1f}% / {np.max(hit_rates):.1f}%")
    
    return hit_rates


def main():
    print("=" * 80)
    print("POINTER MECHANISM EFFECTIVENESS ANALYSIS")
    print("=" * 80)
    print("\nThis analysis measures how beneficial the pointer mechanism is for each dataset")
    print("=" * 80)
    
    # Paths
    geolife_dir = "data/geolife_eps20/processed"
    diy_dir = "data/diy_eps50/processed"
    geolife_prefix = "geolife_eps20_prev7"
    diy_prefix = "diy_eps50_prev7"
    
    # Load test sets (what we actually evaluate on)
    print("\nLoading test datasets...")
    geolife_test = load_dataset(geolife_dir, geolife_prefix, "test")
    diy_test = load_dataset(diy_dir, diy_prefix, "test")
    
    print(f"Geolife test: {len(geolife_test)} samples")
    print(f"DIY test:     {len(diy_test)} samples")
    
    # Analyses
    geo_ptr_stats = analyze_pointer_effectiveness(geolife_test, "Geolife")
    diy_ptr_stats = analyze_pointer_effectiveness(diy_test, "DIY")
    
    geo_ptr_gen = analyze_pointer_vs_generation_potential(geolife_test, "Geolife")
    diy_ptr_gen = analyze_pointer_vs_generation_potential(diy_test, "DIY")
    
    geo_oracle = simulate_oracle_pointer(geolife_test, "Geolife")
    diy_oracle = simulate_oracle_pointer(diy_test, "DIY")
    
    geo_user_rates = analyze_per_user_pointer_effectiveness(geolife_test, "Geolife")
    diy_user_rates = analyze_per_user_pointer_effectiveness(diy_test, "DIY")
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)
    
    geo_hit = geo_ptr_stats['target_in_history'] / (geo_ptr_stats['target_in_history'] + geo_ptr_stats['target_not_in_history']) * 100
    diy_hit = diy_ptr_stats['target_in_history'] / (diy_ptr_stats['target_in_history'] + diy_ptr_stats['target_not_in_history']) * 100
    
    print(f"\n┌────────────────────────────────────────────────────────────────┐")
    print(f"│                    POINTER BENEFIT COMPARISON                  │")
    print(f"├────────────────────────────────────────────────────────────────┤")
    print(f"│ Metric                          │ Geolife    │ DIY        │")
    print(f"├────────────────────────────────────────────────────────────────┤")
    print(f"│ Pointer Hit Rate (%)            │ {geo_hit:>10.1f} │ {diy_hit:>10.1f} │")
    print(f"│ Oracle Pointer Accuracy (%)     │ {geo_oracle:>10.1f} │ {diy_oracle:>10.1f} │")
    print(f"│ Target in Last-3 Rate (%)       │ {geo_ptr_stats['target_in_last_n'][3]/len(geolife_test)*100:>10.1f} │ {diy_ptr_stats['target_in_last_n'][3]/len(diy_test)*100:>10.1f} │")
    print(f"│ User Mean Hit Rate (%)          │ {np.mean(geo_user_rates):>10.1f} │ {np.mean(diy_user_rates):>10.1f} │")
    print(f"└────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    diff = geo_hit - diy_hit
    print(f"\nGeolife has {diff:.1f}pp HIGHER pointer hit rate than DIY.")
    print(f"This means the pointer mechanism has more opportunity to help on Geolife.")
    print(f"\nOn DIY, with {diy_hit:.1f}% hit rate, the pointer mechanism can only")
    print(f"help with about {diy_hit:.0f}% of predictions, limiting potential improvement.")
    
    # Save results
    output_dir = "scripts/analysis_performance_gap/results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'geolife': {
            'hit_rate': geo_hit,
            'oracle_accuracy': geo_oracle,
            'copy_rate': geo_ptr_gen['copy_rate'],
        },
        'diy': {
            'hit_rate': diy_hit,
            'oracle_accuracy': diy_oracle,
            'copy_rate': diy_ptr_gen['copy_rate'],
        }
    }
    
    output_file = os.path.join(output_dir, "pointer_effectiveness.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

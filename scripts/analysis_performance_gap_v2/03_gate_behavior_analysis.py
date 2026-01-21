"""
Pointer Generator Transformer Component Analysis - Script 3: Gate Behavior Analysis

This script analyzes the Pointer-Generation Gate behavior:
1. When should the gate prefer pointer? (target in history, rare)
2. When should the gate prefer generation? (target common, not in history)
3. What dataset characteristics affect optimal gate behavior?

The Gate learns to blend pointer and generation distributions:
    final_probs = gate * ptr_dist + (1 - gate) * gen_probs

Higher gate value → more reliance on pointer
Lower gate value → more reliance on generation
"""

import os
import sys
import json
import pickle
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_pickle_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def analyze_gate_scenarios(name, data_dir, prefix, output_dir):
    """
    Analyze what gate behavior would be optimal for each dataset.
    
    The gate must learn when to:
    - Use pointer (copy from history)
    - Use generation (predict from vocabulary)
    
    This analysis examines the "optimal" scenarios for each.
    """
    print(f"\n{'='*70}")
    print(f"GATE BEHAVIOR ANALYSIS: {name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    train_data = load_pickle_data(os.path.join(data_dir, f"{prefix}_train.pk"))
    test_data = load_pickle_data(os.path.join(data_dir, f"{prefix}_test.pk"))
    
    # Build training frequency
    train_targets = [s['Y'] for s in train_data]
    train_freq = Counter(train_targets)
    total_train = len(train_targets)
    
    stats = {"dataset": name, "test_size": len(test_data)}
    
    def get_freq_pct(loc):
        if loc not in train_freq:
            return 0
        return train_freq[loc] / total_train * 100
    
    # Classify each test sample into gate scenarios
    print("\n[1] GATE SCENARIO CLASSIFICATION")
    
    scenarios = {
        "pointer_only": [],      # In history, rare → gate should be ~1.0
        "pointer_preferred": [], # In history, moderate → gate should be high
        "balanced": [],          # In history, common → both could work
        "generation_preferred": [], # Not in history, common → gate should be low
        "generation_only": [],   # Not in history, rare/unseen → difficult
    }
    
    for sample in test_data:
        history = sample['X'].tolist()
        target = sample['Y']
        freq_pct = get_freq_pct(target)
        in_history = target in history
        
        if in_history:
            if freq_pct < 0.1:
                scenarios["pointer_only"].append(sample)
            elif freq_pct < 1:
                scenarios["pointer_preferred"].append(sample)
            else:
                scenarios["balanced"].append(sample)
        else:
            if freq_pct >= 0.5:
                scenarios["generation_preferred"].append(sample)
            else:
                scenarios["generation_only"].append(sample)
    
    total = len(test_data)
    
    stats["scenarios"] = {}
    print(f"\n  Scenario breakdown:")
    for scenario, samples in scenarios.items():
        pct = len(samples) / total * 100
        stats["scenarios"][scenario] = {
            "count": len(samples),
            "pct": pct,
        }
        print(f"    {scenario:<25}: {len(samples):>5} ({pct:>5.1f}%)")
    
    # 2. Analyze pointer-favorable scenarios
    print("\n[2] POINTER-FAVORABLE SCENARIOS")
    
    pointer_favorable = scenarios["pointer_only"] + scenarios["pointer_preferred"]
    pointer_favorable_pct = len(pointer_favorable) / total * 100
    
    stats["pointer_favorable"] = {
        "count": len(pointer_favorable),
        "pct": pointer_favorable_pct,
    }
    
    print(f"  Pointer should dominate: {len(pointer_favorable)} ({pointer_favorable_pct:.2f}%)")
    print(f"  (Target in history AND not very common)")
    
    # 3. Analyze generation-favorable scenarios
    print("\n[3] GENERATION-FAVORABLE SCENARIOS")
    
    generation_favorable = scenarios["generation_preferred"]
    generation_favorable_pct = len(generation_favorable) / total * 100
    
    stats["generation_favorable"] = {
        "count": len(generation_favorable),
        "pct": generation_favorable_pct,
    }
    
    print(f"  Generation should dominate: {len(generation_favorable)} ({generation_favorable_pct:.2f}%)")
    print(f"  (Target NOT in history BUT very common)")
    
    # 4. Analyze difficult scenarios (neither pointer nor generation is ideal)
    print("\n[4] DIFFICULT SCENARIOS")
    
    difficult = scenarios["generation_only"]
    difficult_pct = len(difficult) / total * 100
    
    stats["difficult"] = {
        "count": len(difficult),
        "pct": difficult_pct,
    }
    
    print(f"  Difficult for both: {len(difficult)} ({difficult_pct:.2f}%)")
    print(f"  (Target NOT in history AND rare/unseen)")
    
    # 5. Gate "flexibility" requirement
    print("\n[5] GATE FLEXIBILITY REQUIREMENT")
    
    # How much does the gate need to vary?
    # If most samples are pointer-favorable, gate can be fixed high
    # If samples are mixed, gate needs to learn to adapt
    
    pointer_scenarios = len(scenarios["pointer_only"]) + len(scenarios["pointer_preferred"]) + len(scenarios["balanced"])
    generation_scenarios = len(scenarios["generation_preferred"]) + len(scenarios["generation_only"])
    
    pointer_dominant_pct = pointer_scenarios / total * 100
    generation_dominant_pct = generation_scenarios / total * 100
    
    # "Balance" metric: how even is the split?
    balance = min(pointer_dominant_pct, generation_dominant_pct) / max(pointer_dominant_pct, generation_dominant_pct)
    
    stats["gate_flexibility"] = {
        "pointer_dominant_pct": pointer_dominant_pct,
        "generation_dominant_pct": generation_dominant_pct,
        "balance_ratio": float(balance),
    }
    
    print(f"  Pointer-favorable scenarios: {pointer_dominant_pct:.1f}%")
    print(f"  Generation-favorable scenarios: {generation_dominant_pct:.1f}%")
    print(f"  Balance ratio: {balance:.3f}")
    print(f"  (Higher balance = gate must be more adaptive)")
    
    # 6. Per-user gate consistency
    print("\n[6] PER-USER GATE CONSISTENCY")
    
    user_pointer_rates = defaultdict(list)
    
    for sample in test_data:
        user = sample['user_X'][0]
        history = sample['X'].tolist()
        target = sample['Y']
        in_history = target in history
        user_pointer_rates[user].append(1 if in_history else 0)
    
    user_rates = [np.mean(rates) * 100 for rates in user_pointer_rates.values()]
    
    stats["user_consistency"] = {
        "mean_pointer_rate": float(np.mean(user_rates)),
        "std_pointer_rate": float(np.std(user_rates)),
        "min_pointer_rate": float(np.min(user_rates)),
        "max_pointer_rate": float(np.max(user_rates)),
    }
    
    print(f"  Mean user pointer-favorable rate: {np.mean(user_rates):.1f}%")
    print(f"  Std user pointer-favorable rate: {np.std(user_rates):.1f}%")
    print(f"  Range: [{np.min(user_rates):.1f}%, {np.max(user_rates):.1f}%]")
    
    # Save
    output_path = os.path.join(output_dir, f"{name}_gate_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")
    
    return stats


def compare_gate_behavior(geolife_stats, diy_stats, output_dir):
    """Compare gate behavior analysis between datasets."""
    print(f"\n{'='*70}")
    print("GATE BEHAVIOR COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<45} {'Geolife':>12} {'DIY':>12}")
    print("-" * 70)
    
    g = geolife_stats
    d = diy_stats
    
    metrics = [
        ("Pointer-only scenarios (%)", g["scenarios"]["pointer_only"]["pct"], d["scenarios"]["pointer_only"]["pct"]),
        ("Pointer-preferred scenarios (%)", g["scenarios"]["pointer_preferred"]["pct"], d["scenarios"]["pointer_preferred"]["pct"]),
        ("Balanced scenarios (%)", g["scenarios"]["balanced"]["pct"], d["scenarios"]["balanced"]["pct"]),
        ("Generation-preferred (%)", g["scenarios"]["generation_preferred"]["pct"], d["scenarios"]["generation_preferred"]["pct"]),
        ("Difficult scenarios (%)", g["scenarios"]["generation_only"]["pct"], d["scenarios"]["generation_only"]["pct"]),
        ("Pointer favorable total (%)", g["pointer_favorable"]["pct"], d["pointer_favorable"]["pct"]),
        ("Gate balance ratio", g["gate_flexibility"]["balance_ratio"], d["gate_flexibility"]["balance_ratio"]),
        ("User std in pointer rate", g["user_consistency"]["std_pointer_rate"], d["user_consistency"]["std_pointer_rate"]),
    ]
    
    for name, g_val, d_val in metrics:
        print(f"{name:<45} {g_val:>12.2f} {d_val:>12.2f}")
    
    print(f"\n[KEY INSIGHT]")
    g_ptr = g["pointer_favorable"]["pct"]
    d_ptr = d["pointer_favorable"]["pct"]
    g_diff = g["difficult"]["pct"]
    d_diff = d["difficult"]["pct"]
    
    print(f"  Pointer-favorable: Geolife {g_ptr:.1f}% vs DIY {d_ptr:.1f}%")
    print(f"  Difficult (neither): Geolife {g_diff:.1f}% vs DIY {d_diff:.1f}%")
    
    if g_ptr < d_ptr:
        print(f"\n  DIY has MORE pointer-favorable scenarios ({d_ptr:.1f}% vs {g_ptr:.1f}%)")
        print(f"  BUT the improvement is smaller - why?")
        print(f"  → The baseline MHSA is already capturing these patterns in DIY!")
    
    # Save comparison
    comparison = {
        "geolife": geolife_stats,
        "diy": diy_stats,
    }
    
    output_path = os.path.join(output_dir, "gate_behavior_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")


def main():
    geolife_dir = "data/geolife_eps20/processed"
    geolife_prefix = "geolife_eps20_prev7"
    
    diy_dir = "data/diy_eps50/processed"
    diy_prefix = "diy_eps50_prev7"
    
    output_dir = "scripts/analysis_performance_gap_v2/results"
    
    geolife_stats = analyze_gate_scenarios("geolife", geolife_dir, geolife_prefix, output_dir)
    diy_stats = analyze_gate_scenarios("diy", diy_dir, diy_prefix, output_dir)
    
    compare_gate_behavior(geolife_stats, diy_stats, output_dir)


if __name__ == "__main__":
    main()

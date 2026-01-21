"""
Pointer Generator Transformer Component Analysis - Script 2: Generation Head Effectiveness

This script analyzes the Generation Head component of Pointer Generator Transformer:
1. When does the generation head need to work? (target NOT in history)
2. How hard are these generation tasks? (rare vs common targets)
3. What characteristics make generation easier/harder?

The Generation head predicts over the full vocabulary - it excels at
predicting common/frequent locations but struggles with rare ones.
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


def analyze_generation_component(name, data_dir, prefix, output_dir):
    """
    Analyze the potential effectiveness of the Generation Head component.
    
    The Generation head predicts over the full location vocabulary.
    It learns from training data, so it works best for:
    - Frequently occurring locations
    - Locations that appear in many users' data
    """
    print(f"\n{'='*70}")
    print(f"GENERATION HEAD ANALYSIS: {name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    train_data = load_pickle_data(os.path.join(data_dir, f"{prefix}_train.pk"))
    test_data = load_pickle_data(os.path.join(data_dir, f"{prefix}_test.pk"))
    
    # Build training target distribution
    train_targets = [s['Y'] for s in train_data]
    train_freq = Counter(train_targets)
    total_train = len(train_targets)
    
    stats = {"dataset": name, "test_size": len(test_data)}
    
    # 1. Generation Coverage: Targets that require generation
    print("\n[1] GENERATION REQUIREMENT (Target NOT in History)")
    
    generation_required = []  # samples where pointer cannot help
    pointer_possible = []  # samples where pointer could help
    
    for sample in test_data:
        history = sample['X'].tolist()
        target = sample['Y']
        
        if target not in history:
            generation_required.append(sample)
        else:
            pointer_possible.append(sample)
    
    gen_required_pct = len(generation_required) / len(test_data) * 100
    
    stats["generation_coverage"] = {
        "generation_required_count": len(generation_required),
        "generation_required_pct": gen_required_pct,
        "pointer_possible_count": len(pointer_possible),
        "pointer_possible_pct": 100 - gen_required_pct,
    }
    
    print(f"  Generation REQUIRED (not in history): {len(generation_required)} ({gen_required_pct:.2f}%)")
    print(f"  Pointer possible (in history): {len(pointer_possible)} ({100-gen_required_pct:.2f}%)")
    
    # 2. Generation Difficulty: Analyze targets requiring generation
    print("\n[2] GENERATION DIFFICULTY (Frequency Analysis)")
    
    def get_freq_tier(loc):
        """Categorize location by training frequency."""
        if loc not in train_freq:
            return "unseen"
        freq_pct = train_freq[loc] / total_train * 100
        if freq_pct >= 1:
            return "very_common"  # >= 1% of training
        elif freq_pct >= 0.1:
            return "common"  # 0.1% - 1%
        elif freq_pct >= 0.01:
            return "moderate"  # 0.01% - 0.1%
        else:
            return "rare"  # < 0.01%
    
    gen_required_tiers = Counter()
    all_test_tiers = Counter()
    
    for sample in generation_required:
        tier = get_freq_tier(sample['Y'])
        gen_required_tiers[tier] += 1
    
    for sample in test_data:
        tier = get_freq_tier(sample['Y'])
        all_test_tiers[tier] += 1
    
    stats["generation_difficulty"] = {
        "gen_required_by_tier": dict(gen_required_tiers),
        "all_test_by_tier": dict(all_test_tiers),
    }
    
    print(f"\n  Targets requiring generation by frequency tier:")
    total_gen = len(generation_required)
    for tier in ["very_common", "common", "moderate", "rare", "unseen"]:
        count = gen_required_tiers.get(tier, 0)
        pct = count / total_gen * 100 if total_gen > 0 else 0
        print(f"    {tier:<15}: {count:>5} ({pct:>5.1f}%)")
    
    print(f"\n  All test targets by frequency tier:")
    for tier in ["very_common", "common", "moderate", "rare", "unseen"]:
        count = all_test_tiers.get(tier, 0)
        pct = count / len(test_data) * 100
        print(f"    {tier:<15}: {count:>5} ({pct:>5.1f}%)")
    
    # 3. Generation Advantage: Cases where generation might beat pointer
    print("\n[3] GENERATION ADVANTAGE ANALYSIS")
    
    # Cases where target IS in history but is VERY COMMON
    # In these cases, generation head might also predict correctly
    gen_could_help = 0
    gen_exclusive = 0  # Only generation can help (not in history)
    
    for sample in test_data:
        history = sample['X'].tolist()
        target = sample['Y']
        tier = get_freq_tier(target)
        
        if target not in history:
            gen_exclusive += 1
        elif tier in ["very_common", "common"]:
            gen_could_help += 1
    
    stats["generation_advantage"] = {
        "generation_exclusive": gen_exclusive,
        "generation_exclusive_pct": gen_exclusive / len(test_data) * 100,
        "generation_could_help": gen_could_help,
        "generation_could_help_pct": gen_could_help / len(test_data) * 100,
    }
    
    print(f"  Generation exclusive (must use gen): {gen_exclusive} ({gen_exclusive/len(test_data)*100:.2f}%)")
    print(f"  Generation could help (common + in history): {gen_could_help} ({gen_could_help/len(test_data)*100:.2f}%)")
    
    # 4. Training data richness for generation
    print("\n[4] TRAINING DATA RICHNESS")
    
    unique_train_targets = len(train_freq)
    train_coverage = [1 for s in test_data if s['Y'] in train_freq]
    
    # Entropy of training target distribution
    probs = [count/total_train for count in train_freq.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(unique_train_targets)
    
    stats["training_richness"] = {
        "unique_train_targets": unique_train_targets,
        "test_covered_by_train_pct": sum(train_coverage) / len(test_data) * 100,
        "entropy": float(entropy),
        "max_entropy": float(max_entropy),
        "normalized_entropy": float(entropy / max_entropy),
    }
    
    print(f"  Unique targets in training: {unique_train_targets}")
    print(f"  Test targets seen in training: {sum(train_coverage)/len(test_data)*100:.2f}%")
    print(f"  Training entropy: {entropy:.2f} (max: {max_entropy:.2f})")
    print(f"  Normalized entropy: {entropy/max_entropy:.4f}")
    
    # 5. Top-K analysis for generation
    print("\n[5] TOP-K GENERATION POTENTIAL")
    
    sorted_locs = [loc for loc, _ in train_freq.most_common()]
    
    def check_topk_coverage(k):
        topk = set(sorted_locs[:k])
        covered = sum(1 for s in test_data if s['Y'] in topk)
        return covered / len(test_data) * 100
    
    top1 = check_topk_coverage(1)
    top5 = check_topk_coverage(5)
    top10 = check_topk_coverage(10)
    top50 = check_topk_coverage(50)
    top100 = check_topk_coverage(100)
    
    stats["topk_coverage"] = {
        "top1": top1,
        "top5": top5,
        "top10": top10,
        "top50": top50,
        "top100": top100,
    }
    
    print(f"  If generation predicts most frequent:")
    print(f"    Top-1:   {top1:.2f}%")
    print(f"    Top-5:   {top5:.2f}%")
    print(f"    Top-10:  {top10:.2f}%")
    print(f"    Top-50:  {top50:.2f}%")
    print(f"    Top-100: {top100:.2f}%")
    
    # Save
    output_path = os.path.join(output_dir, f"{name}_generation_component.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")
    
    return stats


def compare_generation_components(geolife_stats, diy_stats, output_dir):
    """Compare generation component analysis between datasets."""
    print(f"\n{'='*70}")
    print("GENERATION HEAD COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<45} {'Geolife':>12} {'DIY':>12}")
    print("-" * 70)
    
    g = geolife_stats
    d = diy_stats
    
    metrics = [
        ("Generation required (%)", g["generation_coverage"]["generation_required_pct"], d["generation_coverage"]["generation_required_pct"]),
        ("Generation exclusive (%)", g["generation_advantage"]["generation_exclusive_pct"], d["generation_advantage"]["generation_exclusive_pct"]),
        ("Test covered by training (%)", g["training_richness"]["test_covered_by_train_pct"], d["training_richness"]["test_covered_by_train_pct"]),
        ("Normalized entropy", g["training_richness"]["normalized_entropy"], d["training_richness"]["normalized_entropy"]),
        ("Top-10 coverage (%)", g["topk_coverage"]["top10"], d["topk_coverage"]["top10"]),
        ("Top-50 coverage (%)", g["topk_coverage"]["top50"], d["topk_coverage"]["top50"]),
    ]
    
    for name, g_val, d_val in metrics:
        print(f"{name:<45} {g_val:>12.2f} {d_val:>12.2f}")
    
    print(f"\n[KEY INSIGHT]")
    g_gen_req = g["generation_coverage"]["generation_required_pct"]
    d_gen_req = d["generation_coverage"]["generation_required_pct"]
    
    print(f"  Generation is required for {g_gen_req:.1f}% of Geolife vs {d_gen_req:.1f}% of DIY")
    print(f"  This means Pointer mechanism has more opportunity in Geolife")
    
    # Save comparison
    comparison = {
        "geolife": geolife_stats,
        "diy": diy_stats,
    }
    
    output_path = os.path.join(output_dir, "generation_component_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")


def main():
    geolife_dir = "data/geolife_eps20/processed"
    geolife_prefix = "geolife_eps20_prev7"
    
    diy_dir = "data/diy_eps50/processed"
    diy_prefix = "diy_eps50_prev7"
    
    output_dir = "scripts/analysis_performance_gap_v2/results"
    
    geolife_stats = analyze_generation_component("geolife", geolife_dir, geolife_prefix, output_dir)
    diy_stats = analyze_generation_component("diy", diy_dir, diy_prefix, output_dir)
    
    compare_generation_components(geolife_stats, diy_stats, output_dir)


if __name__ == "__main__":
    main()

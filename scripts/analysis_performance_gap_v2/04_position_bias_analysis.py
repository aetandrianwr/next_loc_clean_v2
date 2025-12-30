"""
Pointer V45 Component Analysis - Script 4: Position Bias Effectiveness

This script analyzes the Position Bias component of Pointer V45:
    ptr_scores = ptr_scores + self.position_bias[pos_from_end]

The position bias helps the model learn that:
1. Recent locations are often more relevant
2. Position patterns vary by dataset

Key Question: Does the position bias help more in one dataset?
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


def analyze_position_patterns(name, data_dir, prefix, output_dir):
    """
    Analyze position patterns that the position bias can exploit.
    """
    print(f"\n{'='*70}")
    print(f"POSITION BIAS ANALYSIS: {name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    train_data = load_pickle_data(os.path.join(data_dir, f"{prefix}_train.pk"))
    test_data = load_pickle_data(os.path.join(data_dir, f"{prefix}_test.pk"))
    
    stats = {"dataset": name, "test_size": len(test_data)}
    
    # 1. Position distribution of targets (when in history)
    print("\n[1] TARGET POSITION DISTRIBUTION (when in history)")
    
    position_counts = defaultdict(int)
    total_in_history = 0
    
    for sample in test_data:
        history = sample['X'].tolist()
        target = sample['Y']
        
        if target in history:
            total_in_history += 1
            # Find closest position from end
            for i, loc in enumerate(history):
                if loc == target:
                    pos_from_end = len(history) - i
                    position_counts[pos_from_end] += 1
                    break  # Only count closest
    
    if total_in_history > 0:
        # Calculate position distribution
        position_probs = {}
        for pos in range(1, 21):
            count = position_counts.get(pos, 0)
            position_probs[pos] = count / total_in_history * 100
        
        stats["position_distribution"] = {
            "total_in_history": total_in_history,
            "position_probs": position_probs,
        }
        
        print(f"  Position distribution (top 10):")
        for pos in range(1, 11):
            prob = position_probs.get(pos, 0)
            bar = "█" * int(prob / 2)
            print(f"    Position {pos:>2}: {prob:>5.1f}% {bar}")
    
    # 2. Recency bias strength
    print("\n[2] RECENCY BIAS STRENGTH")
    
    pos_1 = position_counts.get(1, 0)
    pos_1_5 = sum(position_counts.get(i, 0) for i in range(1, 6))
    pos_1_10 = sum(position_counts.get(i, 0) for i in range(1, 11))
    
    if total_in_history > 0:
        recency_1 = pos_1 / total_in_history * 100
        recency_5 = pos_1_5 / total_in_history * 100
        recency_10 = pos_1_10 / total_in_history * 100
        
        stats["recency_bias"] = {
            "position_1_pct": recency_1,
            "position_1_5_pct": recency_5,
            "position_1_10_pct": recency_10,
        }
        
        print(f"  Target at position 1: {recency_1:.1f}%")
        print(f"  Target in positions 1-5: {recency_5:.1f}%")
        print(f"  Target in positions 1-10: {recency_10:.1f}%")
    
    # 3. Position entropy (how predictable is the position?)
    print("\n[3] POSITION PREDICTABILITY")
    
    if total_in_history > 0:
        probs = [count / total_in_history for count in position_counts.values() if count > 0]
        position_entropy = -sum(p * np.log2(p) for p in probs)
        max_entropy = np.log2(len(position_counts))  # if uniform
        
        stats["position_entropy"] = {
            "entropy": float(position_entropy),
            "max_entropy": float(max_entropy),
            "normalized_entropy": float(position_entropy / max_entropy) if max_entropy > 0 else 0,
        }
        
        print(f"  Position entropy: {position_entropy:.2f} (max: {max_entropy:.2f})")
        print(f"  Normalized entropy: {position_entropy/max_entropy:.4f}")
        print(f"  (Lower = more predictable position → position bias helps more)")
    
    # 4. Position consistency by user
    print("\n[4] PER-USER POSITION CONSISTENCY")
    
    user_positions = defaultdict(list)
    
    for sample in test_data:
        user = sample['user_X'][0]
        history = sample['X'].tolist()
        target = sample['Y']
        
        if target in history:
            for i, loc in enumerate(history):
                if loc == target:
                    pos_from_end = len(history) - i
                    user_positions[user].append(pos_from_end)
                    break
    
    user_mean_positions = [np.mean(positions) for positions in user_positions.values() if positions]
    user_std_positions = [np.std(positions) for positions in user_positions.values() if len(positions) > 1]
    
    if user_mean_positions:
        stats["user_position_consistency"] = {
            "mean_of_user_means": float(np.mean(user_mean_positions)),
            "std_of_user_means": float(np.std(user_mean_positions)),
            "mean_of_user_stds": float(np.mean(user_std_positions)) if user_std_positions else 0,
        }
        
        print(f"  Mean of user mean positions: {np.mean(user_mean_positions):.2f}")
        print(f"  Std of user mean positions: {np.std(user_mean_positions):.2f}")
        print(f"  Mean of user position stds: {np.mean(user_std_positions):.2f}")
        print(f"  (High consistency across users → global position bias works well)")
    
    # 5. Position bias potential
    print("\n[5] POSITION BIAS POTENTIAL")
    
    if total_in_history > 0:
        # If we always predict position 1, what's the accuracy?
        pos_1_accuracy = recency_1
        
        # If we predict the mode position, what's the accuracy?
        mode_pos = max(position_counts, key=position_counts.get)
        mode_accuracy = position_counts[mode_pos] / total_in_history * 100
        
        stats["position_bias_potential"] = {
            "always_pos_1_accuracy": pos_1_accuracy,
            "mode_position": mode_pos,
            "mode_position_accuracy": mode_accuracy,
        }
        
        print(f"  If always predict position 1: {pos_1_accuracy:.1f}%")
        print(f"  Mode position: {mode_pos} (accuracy: {mode_accuracy:.1f}%)")
    
    # Save
    output_path = os.path.join(output_dir, f"{name}_position_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")
    
    return stats


def compare_position_patterns(geolife_stats, diy_stats, output_dir):
    """Compare position patterns between datasets."""
    print(f"\n{'='*70}")
    print("POSITION PATTERN COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<45} {'Geolife':>12} {'DIY':>12}")
    print("-" * 70)
    
    g = geolife_stats
    d = diy_stats
    
    metrics = [
        ("Target at position 1 (%)", g["recency_bias"]["position_1_pct"], d["recency_bias"]["position_1_pct"]),
        ("Target in positions 1-5 (%)", g["recency_bias"]["position_1_5_pct"], d["recency_bias"]["position_1_5_pct"]),
        ("Target in positions 1-10 (%)", g["recency_bias"]["position_1_10_pct"], d["recency_bias"]["position_1_10_pct"]),
        ("Position normalized entropy", g["position_entropy"]["normalized_entropy"], d["position_entropy"]["normalized_entropy"]),
        ("Mode position", g["position_bias_potential"]["mode_position"], d["position_bias_potential"]["mode_position"]),
        ("Always pos-1 accuracy (%)", g["position_bias_potential"]["always_pos_1_accuracy"], d["position_bias_potential"]["always_pos_1_accuracy"]),
    ]
    
    for name, g_val, d_val in metrics:
        print(f"{name:<45} {g_val:>12.2f} {d_val:>12.2f}")
    
    print(f"\n[KEY INSIGHT]")
    g_p1 = g["recency_bias"]["position_1_pct"]
    d_p1 = d["recency_bias"]["position_1_pct"]
    g_ent = g["position_entropy"]["normalized_entropy"]
    d_ent = d["position_entropy"]["normalized_entropy"]
    
    print(f"  Position 1 concentration: Geolife {g_p1:.1f}% vs DIY {d_p1:.1f}%")
    print(f"  Position entropy: Geolife {g_ent:.4f} vs DIY {d_ent:.4f}")
    
    if g_p1 > d_p1:
        print(f"\n  Geolife has MORE concentration at position 1")
        print(f"  → Position bias can learn stronger recency preference")
    else:
        print(f"\n  DIY has MORE concentration at position 1")
        print(f"  → Position bias has clear signal to exploit")
    
    # Save comparison
    comparison = {
        "geolife": geolife_stats,
        "diy": diy_stats,
    }
    
    output_path = os.path.join(output_dir, "position_pattern_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")


def main():
    geolife_dir = "data/geolife_eps20/processed"
    geolife_prefix = "geolife_eps20_prev7"
    
    diy_dir = "data/diy_eps50/processed"
    diy_prefix = "diy_eps50_prev7"
    
    output_dir = "scripts/analysis_performance_gap_v2/results"
    
    geolife_stats = analyze_position_patterns("geolife", geolife_dir, geolife_prefix, output_dir)
    diy_stats = analyze_position_patterns("diy", diy_dir, diy_prefix, output_dir)
    
    compare_position_patterns(geolife_stats, diy_stats, output_dir)


if __name__ == "__main__":
    main()

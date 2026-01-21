"""
Pointer Generator Transformer Component Analysis - Script 1: Pointer vs Generation Effectiveness

This script analyzes WHY the Pointer Generator Transformer model's improvement differs between datasets
by focusing on the model's internal mechanisms:
1. Pointer mechanism effectiveness
2. Generation head effectiveness  
3. Gate behavior (how the model blends pointer and generation)

Key Question: Why does Pointer Generator Transformer achieve +20.79% improvement in Geolife but only +3.68% in DIY?

Focus: The PROPOSED MODEL (Pointer Generator Transformer) perspective
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


def load_pickle_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def analyze_pointer_component_potential(name, data_dir, prefix, output_dir):
    """
    Analyze the potential effectiveness of the Pointer mechanism component.
    
    The Pointer mechanism can ONLY predict locations that appear in the history.
    This analysis examines:
    1. How often can Pointer potentially succeed? (target in history)
    2. How "easy" are these copy targets? (position, frequency)
    3. What makes copying harder in one dataset vs another?
    """
    print(f"\n{'='*70}")
    print(f"POINTER COMPONENT ANALYSIS: {name.upper()}")
    print(f"{'='*70}")
    
    # Load data
    train_data = load_pickle_data(os.path.join(data_dir, f"{prefix}_train.pk"))
    test_data = load_pickle_data(os.path.join(data_dir, f"{prefix}_test.pk"))
    
    all_data = train_data + test_data
    
    stats = {"dataset": name, "test_size": len(test_data)}
    
    # 1. Pointer Coverage: How often can pointer mechanism succeed?
    print("\n[1] POINTER COVERAGE (Target in History)")
    
    target_in_history = 0
    target_positions = []  # position from end (1 = most recent)
    target_occurrences = []  # how many times target appears in history
    
    for sample in test_data:
        history = sample['X'].tolist()
        target = sample['Y']
        
        if target in history:
            target_in_history += 1
            # Find positions where target appears
            positions = [len(history) - i for i, loc in enumerate(history) if loc == target]
            target_positions.append(min(positions))  # closest to end
            target_occurrences.append(len(positions))
    
    coverage = target_in_history / len(test_data) * 100
    
    stats["pointer_coverage"] = {
        "target_in_history_count": target_in_history,
        "target_in_history_pct": coverage,
        "not_in_history_count": len(test_data) - target_in_history,
        "not_in_history_pct": 100 - coverage,
    }
    
    print(f"  Target in history: {target_in_history}/{len(test_data)} ({coverage:.2f}%)")
    print(f"  NOT in history: {len(test_data) - target_in_history} ({100-coverage:.2f}%)")
    
    # 2. Pointer Difficulty: How hard is the copy task?
    print("\n[2] POINTER DIFFICULTY (Position & Ambiguity)")
    
    if target_positions:
        mean_pos = np.mean(target_positions)
        median_pos = np.median(target_positions)
        mean_occ = np.mean(target_occurrences)
        
        # Position distribution
        pos_1 = sum(1 for p in target_positions if p == 1)
        pos_2_5 = sum(1 for p in target_positions if 2 <= p <= 5)
        pos_6_10 = sum(1 for p in target_positions if 6 <= p <= 10)
        pos_gt_10 = sum(1 for p in target_positions if p > 10)
        
        total_copyable = len(target_positions)
        
        stats["pointer_difficulty"] = {
            "mean_target_position": float(mean_pos),
            "median_target_position": float(median_pos),
            "mean_occurrences_in_history": float(mean_occ),
            "position_1_pct": pos_1 / total_copyable * 100,
            "position_2_5_pct": pos_2_5 / total_copyable * 100,
            "position_6_10_pct": pos_6_10 / total_copyable * 100,
            "position_gt_10_pct": pos_gt_10 / total_copyable * 100,
        }
        
        print(f"  Mean closest position: {mean_pos:.2f}")
        print(f"  Median closest position: {median_pos:.2f}")
        print(f"  Mean occurrences in history: {mean_occ:.2f}")
        print(f"\n  Position distribution (when target in history):")
        print(f"    Position 1 (most recent): {pos_1:>5} ({pos_1/total_copyable*100:>5.1f}%)")
        print(f"    Position 2-5:             {pos_2_5:>5} ({pos_2_5/total_copyable*100:>5.1f}%)")
        print(f"    Position 6-10:            {pos_6_10:>5} ({pos_6_10/total_copyable*100:>5.1f}%)")
        print(f"    Position >10:             {pos_gt_10:>5} ({pos_gt_10/total_copyable*100:>5.1f}%)")
        
        # Ambiguity analysis: when target appears multiple times, which position is "correct"?
        multi_occurrence = [occ for occ in target_occurrences if occ > 1]
        single_occurrence = [occ for occ in target_occurrences if occ == 1]
        
        stats["pointer_difficulty"]["single_occurrence_pct"] = len(single_occurrence) / total_copyable * 100
        stats["pointer_difficulty"]["multi_occurrence_pct"] = len(multi_occurrence) / total_copyable * 100
        
        print(f"\n  Ambiguity analysis:")
        print(f"    Single occurrence: {len(single_occurrence)} ({len(single_occurrence)/total_copyable*100:.1f}%) - Easier")
        print(f"    Multiple occurrences: {len(multi_occurrence)} ({len(multi_occurrence)/total_copyable*100:.1f}%) - Harder")
    
    # 3. Sequence Length Impact on Pointer
    print("\n[3] SEQUENCE LENGTH IMPACT ON POINTER")
    
    seq_lengths = [len(sample['X']) for sample in test_data]
    copyable_seq_lengths = [len(sample['X']) for sample in test_data if sample['Y'] in sample['X'].tolist()]
    
    stats["sequence_length"] = {
        "mean_all": float(np.mean(seq_lengths)),
        "mean_copyable": float(np.mean(copyable_seq_lengths)) if copyable_seq_lengths else 0,
        "max_all": int(np.max(seq_lengths)),
    }
    
    print(f"  Mean sequence length (all): {np.mean(seq_lengths):.2f}")
    print(f"  Mean sequence length (copyable): {np.mean(copyable_seq_lengths):.2f}")
    
    # Longer sequences = more positions to attend to = harder for pointer
    long_seq_threshold = np.percentile(seq_lengths, 75)
    long_seqs = [(sample, len(sample['X'])) for sample in test_data if len(sample['X']) >= long_seq_threshold]
    long_seq_copyable = sum(1 for sample, _ in long_seqs if sample['Y'] in sample['X'].tolist())
    
    print(f"  Long sequences (>= {long_seq_threshold:.0f}): {len(long_seqs)}")
    print(f"  Long sequences with copyable target: {long_seq_copyable} ({long_seq_copyable/len(long_seqs)*100:.1f}%)")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_pointer_component.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")
    
    return stats


def compare_pointer_components(geolife_stats, diy_stats, output_dir):
    """Compare pointer component analysis between datasets."""
    print(f"\n{'='*70}")
    print("POINTER COMPONENT COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<45} {'Geolife':>12} {'DIY':>12}")
    print("-" * 70)
    
    g = geolife_stats
    d = diy_stats
    
    metrics = [
        ("Target in history (%)", g["pointer_coverage"]["target_in_history_pct"], d["pointer_coverage"]["target_in_history_pct"]),
        ("Mean target position from end", g["pointer_difficulty"]["mean_target_position"], d["pointer_difficulty"]["mean_target_position"]),
        ("Position 1 (most recent) (%)", g["pointer_difficulty"]["position_1_pct"], d["pointer_difficulty"]["position_1_pct"]),
        ("Position 2-5 (%)", g["pointer_difficulty"]["position_2_5_pct"], d["pointer_difficulty"]["position_2_5_pct"]),
        ("Position >10 (%)", g["pointer_difficulty"]["position_gt_10_pct"], d["pointer_difficulty"]["position_gt_10_pct"]),
        ("Single occurrence (%)", g["pointer_difficulty"]["single_occurrence_pct"], d["pointer_difficulty"]["single_occurrence_pct"]),
        ("Multiple occurrences (%)", g["pointer_difficulty"]["multi_occurrence_pct"], d["pointer_difficulty"]["multi_occurrence_pct"]),
        ("Mean sequence length", g["sequence_length"]["mean_all"], d["sequence_length"]["mean_all"]),
    ]
    
    for name, g_val, d_val in metrics:
        print(f"{name:<45} {g_val:>12.2f} {d_val:>12.2f}")
    
    # Save comparison
    comparison = {
        "geolife": geolife_stats,
        "diy": diy_stats,
    }
    
    output_path = os.path.join(output_dir, "pointer_component_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")


def main():
    geolife_dir = "data/geolife_eps20/processed"
    geolife_prefix = "geolife_eps20_prev7"
    
    diy_dir = "data/diy_eps50/processed"
    diy_prefix = "diy_eps50_prev7"
    
    output_dir = "scripts/analysis_performance_gap_v2/results"
    
    geolife_stats = analyze_pointer_component_potential("geolife", geolife_dir, geolife_prefix, output_dir)
    diy_stats = analyze_pointer_component_potential("diy", diy_dir, diy_prefix, output_dir)
    
    compare_pointer_components(geolife_stats, diy_stats, output_dir)


if __name__ == "__main__":
    main()

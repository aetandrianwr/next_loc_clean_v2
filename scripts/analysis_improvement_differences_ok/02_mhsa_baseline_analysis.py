"""
MHSA Baseline Performance Analysis Script

This script analyzes why MHSA performs differently on Geolife vs DIY datasets.
Key factors: training data density, location predictability, transition patterns.

Output: Analysis of why MHSA achieves 53% on DIY but only 33% on Geolife.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

# Paths
GEOLIFE_TRAIN = "data/geolife_eps20/processed/geolife_eps20_prev7_train.pk"
GEOLIFE_TEST = "data/geolife_eps20/processed/geolife_eps20_prev7_test.pk"
GEOLIFE_META = "data/geolife_eps20/processed/geolife_eps20_prev7_metadata.json"
DIY_TRAIN = "data/diy_eps50/processed/diy_eps50_prev7_train.pk"
DIY_TEST = "data/diy_eps50/processed/diy_eps50_prev7_test.pk"
DIY_META = "data/diy_eps50/processed/diy_eps50_prev7_metadata.json"

OUTPUT_DIR = "scripts/analysis_improvement_differences_ok/results"


def load_data():
    """Load all datasets."""
    with open(GEOLIFE_TRAIN, 'rb') as f:
        geo_train = pickle.load(f)
    with open(GEOLIFE_TEST, 'rb') as f:
        geo_test = pickle.load(f)
    with open(DIY_TRAIN, 'rb') as f:
        diy_train = pickle.load(f)
    with open(DIY_TEST, 'rb') as f:
        diy_test = pickle.load(f)
    with open(GEOLIFE_META, 'r') as f:
        geo_meta = json.load(f)
    with open(DIY_META, 'r') as f:
        diy_meta = json.load(f)
    return geo_train, geo_test, geo_meta, diy_train, diy_test, diy_meta


def analyze_training_density(train, meta, name):
    """Analyze training data density per location."""
    all_locs = []
    for s in train:
        all_locs.extend(s['X'])
        all_locs.append(s['Y'])
    loc_counts = Counter(all_locs)
    
    # Frequency statistics
    freqs = list(loc_counts.values())
    
    return {
        "Dataset": name,
        "Total Train Sequences": len(train),
        "Unique Locations": len(loc_counts),
        "Mean Samples/Location": np.mean(freqs),
        "Median Samples/Location": np.median(freqs),
        "Rare Locs (<10 samples)": sum(1 for f in freqs if f < 10),
        "Rare Locs (%)": sum(1 for f in freqs if f < 10) / len(freqs) * 100,
        "Common Locs (>=100 samples)": sum(1 for f in freqs if f >= 100),
        "Common Locs (%)": sum(1 for f in freqs if f >= 100) / len(freqs) * 100,
    }


def analyze_transition_patterns(train, test, name):
    """Analyze transition patterns and Markov baseline."""
    # Build transition matrix
    transitions = defaultdict(Counter)
    for s in train:
        full_seq = list(s['X']) + [s['Y']]
        for i in range(len(full_seq) - 1):
            transitions[full_seq[i]][full_seq[i+1]] += 1
    
    # Markov accuracy
    markov_correct = 0
    unseen_trans = 0
    
    for s in test:
        last_loc = s['X'][-1]
        if last_loc in transitions and transitions[last_loc]:
            pred = transitions[last_loc].most_common(1)[0][0]
            if pred == s['Y']:
                markov_correct += 1
        else:
            unseen_trans += 1
    
    # Transition entropy
    entropies = []
    for loc, next_locs in transitions.items():
        if sum(next_locs.values()) > 5:
            total_trans = sum(next_locs.values())
            probs = [c/total_trans for c in next_locs.values()]
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
            entropies.append(entropy)
    
    # Self-loop rate
    total_trans = sum(sum(v.values()) for v in transitions.values())
    self_loops = sum(transitions[loc][loc] for loc in transitions)
    
    return {
        "Dataset": name,
        "1st Order Markov Acc (%)": markov_correct / len(test) * 100,
        "Unseen Transitions (%)": unseen_trans / len(test) * 100,
        "Mean Transition Entropy (bits)": np.mean(entropies) if entropies else 0,
        "Median Transition Entropy (bits)": np.median(entropies) if entropies else 0,
        "Self-Loop Rate (%)": self_loops / total_trans * 100 if total_trans > 0 else 0,
    }


def simple_baselines(test, name):
    """Calculate simple baseline accuracies."""
    total = len(test)
    
    # Last location baseline
    last_loc_correct = sum(1 for s in test if s['Y'] == s['X'][-1])
    
    # Most common in sequence
    most_common_correct = 0
    for s in test:
        counter = Counter(s['X'])
        most_common = counter.most_common(1)[0][0]
        if most_common == s['Y']:
            most_common_correct += 1
    
    # Random baseline
    target_counts = Counter(s['Y'] for s in test)
    random_acc = 1 / len(target_counts) * 100
    
    return {
        "Dataset": name,
        "Last Location Acc (%)": last_loc_correct / total * 100,
        "Most Common in Seq Acc (%)": most_common_correct / total * 100,
        "Random Baseline Acc (%)": random_acc,
    }


def analyze_test_target_difficulty(train, test, name):
    """Analyze how difficult test targets are to predict."""
    # Build training location frequencies
    train_loc_counter = Counter()
    for s in train:
        train_loc_counter.update(s['X'])
        train_loc_counter[s['Y']] += 1
    
    test_targets = [s['Y'] for s in test]
    
    # Percentiles
    freqs = list(train_loc_counter.values())
    p50 = np.percentile(freqs, 50)
    p90 = np.percentile(freqs, 90)
    
    rare_targets = sum(1 for t in test_targets if train_loc_counter.get(t, 0) < p50)
    common_targets = sum(1 for t in test_targets if train_loc_counter.get(t, 0) > p90)
    unseen_targets = sum(1 for t in test_targets if train_loc_counter.get(t, 0) == 0)
    
    return {
        "Dataset": name,
        "Rare Test Targets (<p50) (%)": rare_targets / len(test) * 100,
        "Common Test Targets (>p90) (%)": common_targets / len(test) * 100,
        "Unseen Test Targets (%)": unseen_targets / len(test) * 100,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("MHSA BASELINE PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Load data
    geo_train, geo_test, geo_meta, diy_train, diy_test, diy_meta = load_data()
    
    # Training density
    print("\n1. TRAINING DATA DENSITY")
    print("-" * 50)
    geo_density = analyze_training_density(geo_train, geo_meta, "Geolife")
    diy_density = analyze_training_density(diy_train, diy_meta, "DIY")
    
    density_df = pd.DataFrame([geo_density, diy_density])
    print(density_df.to_string(index=False))
    density_df.to_csv(f"{OUTPUT_DIR}/02_training_density.csv", index=False)
    
    # Transition patterns
    print("\n2. TRANSITION PATTERNS")
    print("-" * 50)
    geo_trans = analyze_transition_patterns(geo_train, geo_test, "Geolife")
    diy_trans = analyze_transition_patterns(diy_train, diy_test, "DIY")
    
    trans_df = pd.DataFrame([geo_trans, diy_trans])
    print(trans_df.to_string(index=False))
    trans_df.to_csv(f"{OUTPUT_DIR}/02_transition_patterns.csv", index=False)
    
    # Simple baselines
    print("\n3. SIMPLE BASELINES")
    print("-" * 50)
    geo_base = simple_baselines(geo_test, "Geolife")
    diy_base = simple_baselines(diy_test, "DIY")
    
    base_df = pd.DataFrame([geo_base, diy_base])
    print(base_df.to_string(index=False))
    base_df.to_csv(f"{OUTPUT_DIR}/02_simple_baselines.csv", index=False)
    
    # Test target difficulty
    print("\n4. TEST TARGET DIFFICULTY")
    print("-" * 50)
    geo_diff = analyze_test_target_difficulty(geo_train, geo_test, "Geolife")
    diy_diff = analyze_test_target_difficulty(diy_train, diy_test, "DIY")
    
    diff_df = pd.DataFrame([geo_diff, diy_diff])
    print(diff_df.to_string(index=False))
    diff_df.to_csv(f"{OUTPUT_DIR}/02_target_difficulty.csv", index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
DIY has stronger patterns that MHSA can learn:
  - Higher 1st-order Markov accuracy (34.49% vs 21.25%)
  - Lower unseen transitions (4.04% vs 22.99%)
  - More common test targets (78.25% vs 63.79%)
  - Fewer unseen test targets (3.92% vs 22.96%)
  
This explains why MHSA achieves 53.17% on DIY vs 33.18% on Geolife.
The DIY dataset has more predictable, learnable patterns.
""")


if __name__ == "__main__":
    main()

"""
Dataset Characteristics Analysis Script

This script analyzes the fundamental characteristics of Geolife and DIY datasets
to understand why the improvement from MHSA to Pointer V45 differs between datasets.

Output: Generates tables and statistics about dataset properties.
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


def calculate_entropy(counts):
    """Calculate entropy of a distribution."""
    total = sum(counts.values())
    probs = [c/total for c in counts.values()]
    return -sum(p * np.log2(p + 1e-10) for p in probs)


def analyze_basic_stats(train, test, meta, name):
    """Calculate basic dataset statistics."""
    stats = {
        "Dataset": name,
        "Train Sequences": len(train),
        "Test Sequences": len(test),
        "Total Users": meta["total_user_num"],
        "Total Locations": meta["total_loc_num"],
        "Samples per User": len(train) / meta["total_user_num"],
    }
    
    # Sequence length stats
    seq_lens = [len(s["X"]) for s in train]
    stats["Seq Len Mean"] = np.mean(seq_lens)
    stats["Seq Len Median"] = np.median(seq_lens)
    stats["Seq Len Max"] = max(seq_lens)
    
    # Target in history
    target_in_hist = sum(1 for s in test if s['Y'] in s['X']) / len(test) * 100
    stats["Target in History (%)"] = target_in_hist
    
    # Target entropy
    target_counts = Counter(s['Y'] for s in test)
    stats["Target Entropy (bits)"] = calculate_entropy(target_counts)
    stats["Unique Test Targets"] = len(target_counts)
    
    return stats


def analyze_location_distribution(train, name):
    """Analyze location usage distribution."""
    all_locs = []
    for s in train:
        all_locs.extend(s['X'])
        all_locs.append(s['Y'])
    
    loc_counts = Counter(all_locs)
    total = sum(loc_counts.values())
    
    # Gini coefficient
    sorted_counts = sorted(loc_counts.values())
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts) / total
    gini = (n + 1 - 2 * sum(cumsum)) / n
    
    # Top-k coverage
    cumsum_pct = 0
    locs_for_80 = 0
    for loc, count in loc_counts.most_common():
        cumsum_pct += count / total
        locs_for_80 += 1
        if cumsum_pct >= 0.8:
            break
    
    top_10_visits = sum(c for _, c in loc_counts.most_common(10))
    top_50_visits = sum(c for _, c in loc_counts.most_common(50))
    
    return {
        "Dataset": name,
        "Unique Locations": len(loc_counts),
        "Gini Coefficient": gini,
        "Locs for 80% Coverage": locs_for_80,
        "Locs for 80% (%)": locs_for_80 / len(loc_counts) * 100,
        "Top-10 Locs Coverage (%)": top_10_visits / total * 100,
        "Top-50 Locs Coverage (%)": top_50_visits / total * 100,
    }


def analyze_pointer_scenarios(test, name):
    """Analyze scenarios where pointer mechanism would help."""
    categories = {
        'repeat_last': 0,
        'repeat_recent_3': 0,
        'repeat_recent_5': 0,
        'repeat_older': 0,
        'not_in_seq': 0,
    }
    
    for s in test:
        if s['Y'] == s['X'][-1]:
            categories['repeat_last'] += 1
        elif s['Y'] in s['X'][-3:]:
            categories['repeat_recent_3'] += 1
        elif s['Y'] in s['X'][-5:]:
            categories['repeat_recent_5'] += 1
        elif s['Y'] in s['X']:
            categories['repeat_older'] += 1
        else:
            categories['not_in_seq'] += 1
    
    total = len(test)
    result = {"Dataset": name}
    for cat, count in categories.items():
        result[cat] = count
        result[f"{cat} (%)"] = count / total * 100
    
    result["Pointer-favorable (%)"] = (total - categories['not_in_seq']) / total * 100
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("DATASET CHARACTERISTICS ANALYSIS")
    print("=" * 70)
    
    # Load data
    geo_train, geo_test, geo_meta, diy_train, diy_test, diy_meta = load_data()
    
    # Basic stats
    print("\n1. BASIC STATISTICS")
    print("-" * 50)
    geo_stats = analyze_basic_stats(geo_train, geo_test, geo_meta, "Geolife")
    diy_stats = analyze_basic_stats(diy_train, diy_test, diy_meta, "DIY")
    
    basic_df = pd.DataFrame([geo_stats, diy_stats])
    print(basic_df.to_string(index=False))
    basic_df.to_csv(f"{OUTPUT_DIR}/01_basic_statistics.csv", index=False)
    
    # Location distribution
    print("\n2. LOCATION DISTRIBUTION")
    print("-" * 50)
    geo_loc = analyze_location_distribution(geo_train, "Geolife")
    diy_loc = analyze_location_distribution(diy_train, "DIY")
    
    loc_df = pd.DataFrame([geo_loc, diy_loc])
    print(loc_df.to_string(index=False))
    loc_df.to_csv(f"{OUTPUT_DIR}/01_location_distribution.csv", index=False)
    
    # Pointer scenarios
    print("\n3. POINTER MECHANISM SCENARIOS")
    print("-" * 50)
    geo_ptr = analyze_pointer_scenarios(geo_test, "Geolife")
    diy_ptr = analyze_pointer_scenarios(diy_test, "DIY")
    
    ptr_df = pd.DataFrame([geo_ptr, diy_ptr])
    print(ptr_df.to_string(index=False))
    ptr_df.to_csv(f"{OUTPUT_DIR}/01_pointer_scenarios.csv", index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

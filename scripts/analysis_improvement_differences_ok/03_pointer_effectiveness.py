"""
Pointer Mechanism Effectiveness Analysis Script

This script analyzes the effectiveness of the pointer mechanism and why
it provides larger improvements on Geolife vs DIY.

Key Analysis:
1. Target position in sequence (where pointer would point)
2. Recency patterns (how recent is the target in history)
3. Theoretical ceiling analysis
4. Room for improvement analysis
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

# Model performance (from experiments)
GEOLIFE_MHSA_ACC = 33.18
GEOLIFE_POINTER_ACC = 53.97
DIY_MHSA_ACC = 53.17
DIY_POINTER_ACC = 56.85


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
    return geo_train, geo_test, diy_train, diy_test


def analyze_target_position(test, name):
    """Analyze where target appears in sequence history."""
    positions_from_end = []
    not_in_hist = 0
    
    for s in test:
        if s['Y'] in s['X']:
            x_list = list(s['X'])
            # Find most recent position
            indices = [i for i, loc in enumerate(x_list) if loc == s['Y']]
            pos_from_end = len(x_list) - indices[-1] - 1
            positions_from_end.append(pos_from_end)
        else:
            not_in_hist += 1
    
    total = len(test)
    in_hist = total - not_in_hist
    
    result = {
        "Dataset": name,
        "Target in History (%)": in_hist / total * 100,
        "Target NOT in History (%)": not_in_hist / total * 100,
    }
    
    if positions_from_end:
        result["Mean Position from End"] = np.mean(positions_from_end)
        result["Median Position from End"] = np.median(positions_from_end)
        result["Target at Last (pos=0) (%)"] = sum(1 for p in positions_from_end if p == 0) / in_hist * 100
        result["Target in Last 3 (%)"] = sum(1 for p in positions_from_end if p < 3) / in_hist * 100
        result["Target in Last 5 (%)"] = sum(1 for p in positions_from_end if p < 5) / in_hist * 100
    
    return result


def analyze_repetition_patterns(test, name):
    """Analyze location repetition patterns."""
    total = len(test)
    
    return {
        "Dataset": name,
        "Next = Last Location (%)": sum(1 for s in test if s['Y'] == s['X'][-1]) / total * 100,
        "Next in Last 3 Locs (%)": sum(1 for s in test if s['Y'] in s['X'][-3:]) / total * 100,
        "Next in Last 5 Locs (%)": sum(1 for s in test if s['Y'] in s['X'][-5:]) / total * 100,
        "Next in Any History (%)": sum(1 for s in test if s['Y'] in s['X']) / total * 100,
    }


def theoretical_ceiling_analysis(test, mhsa_acc, pointer_acc, name):
    """Calculate theoretical maximum and actual utilization."""
    total = len(test)
    in_hist_count = sum(1 for s in test if s['Y'] in s['X'])
    in_hist_pct = in_hist_count / total * 100
    not_in_hist_pct = 100 - in_hist_pct
    
    # Theoretical max: 100% for in-history + baseline for not-in-history
    theoretical_max = in_hist_pct + (not_in_hist_pct * mhsa_acc / 100)
    
    return {
        "Dataset": name,
        "In History (%)": in_hist_pct,
        "Not in History (%)": not_in_hist_pct,
        "Theoretical Max (%)": theoretical_max,
        "MHSA Actual (%)": mhsa_acc,
        "Pointer Generator Transformer Actual (%)": pointer_acc,
        "MHSA % of Theoretical": mhsa_acc / theoretical_max * 100,
        "Pointer Generator Transformer % of Theoretical": pointer_acc / theoretical_max * 100,
        "Improvement Gap (pp)": pointer_acc - mhsa_acc,
        "Room Left (pp)": theoretical_max - pointer_acc,
    }


def analyze_improvement_breakdown(test, mhsa_acc, pointer_acc, name):
    """Break down where improvements come from."""
    total = len(test)
    
    # Categorize test samples
    repeat_last = sum(1 for s in test if s['Y'] == s['X'][-1])
    repeat_recent = sum(1 for s in test if s['Y'] != s['X'][-1] and s['Y'] in s['X'][-5:])
    repeat_older = sum(1 for s in test if s['Y'] not in s['X'][-5:] and s['Y'] in s['X'])
    not_in_seq = sum(1 for s in test if s['Y'] not in s['X'])
    
    # Calculate potential improvements
    # Pointer can help most with repeat_last and repeat_recent
    high_pointer_potential = (repeat_last + repeat_recent) / total * 100
    medium_pointer_potential = repeat_older / total * 100
    no_pointer_potential = not_in_seq / total * 100
    
    return {
        "Dataset": name,
        "Repeat Last (%)": repeat_last / total * 100,
        "Repeat Recent (not last, in 5) (%)": repeat_recent / total * 100,
        "Repeat Older (in seq, not in 5) (%)": repeat_older / total * 100,
        "Not in Sequence (%)": not_in_seq / total * 100,
        "High Pointer Potential (%)": high_pointer_potential,
        "Medium Pointer Potential (%)": medium_pointer_potential,
        "No Pointer Benefit (%)": no_pointer_potential,
        "Improvement (pp)": pointer_acc - mhsa_acc,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("POINTER MECHANISM EFFECTIVENESS ANALYSIS")
    print("=" * 70)
    
    # Load data
    geo_train, geo_test, diy_train, diy_test = load_data()
    
    # Target position analysis
    print("\n1. TARGET POSITION IN SEQUENCE")
    print("-" * 50)
    geo_pos = analyze_target_position(geo_test, "Geolife")
    diy_pos = analyze_target_position(diy_test, "DIY")
    
    pos_df = pd.DataFrame([geo_pos, diy_pos])
    print(pos_df.to_string(index=False))
    pos_df.to_csv(f"{OUTPUT_DIR}/03_target_position.csv", index=False)
    
    # Repetition patterns
    print("\n2. REPETITION PATTERNS")
    print("-" * 50)
    geo_rep = analyze_repetition_patterns(geo_test, "Geolife")
    diy_rep = analyze_repetition_patterns(diy_test, "DIY")
    
    rep_df = pd.DataFrame([geo_rep, diy_rep])
    print(rep_df.to_string(index=False))
    rep_df.to_csv(f"{OUTPUT_DIR}/03_repetition_patterns.csv", index=False)
    
    # Theoretical ceiling
    print("\n3. THEORETICAL CEILING ANALYSIS")
    print("-" * 50)
    geo_ceil = theoretical_ceiling_analysis(geo_test, GEOLIFE_MHSA_ACC, GEOLIFE_POINTER_ACC, "Geolife")
    diy_ceil = theoretical_ceiling_analysis(diy_test, DIY_MHSA_ACC, DIY_POINTER_ACC, "DIY")
    
    ceil_df = pd.DataFrame([geo_ceil, diy_ceil])
    print(ceil_df.to_string(index=False))
    ceil_df.to_csv(f"{OUTPUT_DIR}/03_theoretical_ceiling.csv", index=False)
    
    # Improvement breakdown
    print("\n4. IMPROVEMENT BREAKDOWN")
    print("-" * 50)
    geo_break = analyze_improvement_breakdown(geo_test, GEOLIFE_MHSA_ACC, GEOLIFE_POINTER_ACC, "Geolife")
    diy_break = analyze_improvement_breakdown(diy_test, DIY_MHSA_ACC, DIY_POINTER_ACC, "DIY")
    
    break_df = pd.DataFrame([geo_break, diy_break])
    print(break_df.to_string(index=False))
    break_df.to_csv(f"{OUTPUT_DIR}/03_improvement_breakdown.csv", index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"""
THEORETICAL CEILING ANALYSIS:

Geolife:
  - MHSA captures only {geo_ceil['MHSA % of Theoretical']:.1f}% of theoretical potential
  - Pointer Generator Transformer captures {geo_ceil['Pointer Generator Transformer % of Theoretical']:.1f}% of theoretical potential
  - Improvement: +{geo_ceil['Improvement Gap (pp)']:.2f} percentage points
  - Room left: {geo_ceil['Room Left (pp)']:.2f} percentage points

DIY:
  - MHSA already captures {diy_ceil['MHSA % of Theoretical']:.1f}% of theoretical potential
  - Pointer Generator Transformer captures {diy_ceil['Pointer Generator Transformer % of Theoretical']:.1f}% of theoretical potential  
  - Improvement: +{diy_ceil['Improvement Gap (pp)']:.2f} percentage points
  - Room left: {diy_ceil['Room Left (pp)']:.2f} percentage points

CONCLUSION:
The smaller improvement on DIY is because MHSA already performs well on DIY,
capturing ~57% of theoretical potential vs only ~37% on Geolife.
Both models reach similar utilization of theoretical potential (~60-61%),
but they start from very different baselines.
""")


if __name__ == "__main__":
    main()

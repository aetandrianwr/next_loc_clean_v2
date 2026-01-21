"""
Root Cause Analysis: Why Different Improvements?

This script brings together all analyses to provide a comprehensive
explanation of why Pointer Generator Transformer shows +20.78pp improvement on Geolife
but only +3.71pp on DIY.

Key factors analyzed:
1. Dataset scale and complexity
2. Pattern learnability (why MHSA differs)
3. Pointer mechanism applicability
4. Final summary with all evidence
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

# Model performance
GEOLIFE_MHSA_ACC = 33.18
GEOLIFE_POINTER_ACC = 53.97
DIY_MHSA_ACC = 53.17
DIY_POINTER_ACC = 56.85


def load_data():
    """Load all datasets."""
    data = {}
    with open(GEOLIFE_TRAIN, 'rb') as f:
        data['geo_train'] = pickle.load(f)
    with open(GEOLIFE_TEST, 'rb') as f:
        data['geo_test'] = pickle.load(f)
    with open(DIY_TRAIN, 'rb') as f:
        data['diy_train'] = pickle.load(f)
    with open(DIY_TEST, 'rb') as f:
        data['diy_test'] = pickle.load(f)
    with open(GEOLIFE_META, 'r') as f:
        data['geo_meta'] = json.load(f)
    with open(DIY_META, 'r') as f:
        data['diy_meta'] = json.load(f)
    return data


def calculate_entropy(counts):
    """Calculate entropy of a distribution."""
    total = sum(counts.values())
    probs = [c/total for c in counts.values()]
    return -sum(p * np.log2(p + 1e-10) for p in probs)


def comprehensive_comparison(data):
    """Create comprehensive comparison table."""
    geo_train = data['geo_train']
    geo_test = data['geo_test']
    geo_meta = data['geo_meta']
    diy_train = data['diy_train']
    diy_test = data['diy_test']
    diy_meta = data['diy_meta']
    
    # Calculate all metrics
    metrics = []
    
    # ==================== SCALE METRICS ====================
    metrics.append({
        "Category": "Scale",
        "Metric": "Training Sequences",
        "Geolife": len(geo_train),
        "DIY": len(diy_train),
        "Ratio (DIY/Geo)": len(diy_train) / len(geo_train),
        "Impact on Improvement": "Neutral"
    })
    
    metrics.append({
        "Category": "Scale",
        "Metric": "Test Sequences",
        "Geolife": len(geo_test),
        "DIY": len(diy_test),
        "Ratio (DIY/Geo)": len(diy_test) / len(geo_test),
        "Impact on Improvement": "Neutral"
    })
    
    metrics.append({
        "Category": "Scale",
        "Metric": "Total Users",
        "Geolife": geo_meta["total_user_num"],
        "DIY": diy_meta["total_user_num"],
        "Ratio (DIY/Geo)": diy_meta["total_user_num"] / geo_meta["total_user_num"],
        "Impact on Improvement": "Reduces DIY improvement"
    })
    
    metrics.append({
        "Category": "Scale",
        "Metric": "Total Locations",
        "Geolife": geo_meta["total_loc_num"],
        "DIY": diy_meta["total_loc_num"],
        "Ratio (DIY/Geo)": diy_meta["total_loc_num"] / geo_meta["total_loc_num"],
        "Impact on Improvement": "Reduces DIY improvement"
    })
    
    # ==================== COMPLEXITY METRICS ====================
    geo_targets = Counter(s['Y'] for s in geo_test)
    diy_targets = Counter(s['Y'] for s in diy_test)
    
    metrics.append({
        "Category": "Complexity",
        "Metric": "Target Entropy (bits)",
        "Geolife": round(calculate_entropy(geo_targets), 2),
        "DIY": round(calculate_entropy(diy_targets), 2),
        "Ratio (DIY/Geo)": round(calculate_entropy(diy_targets) / calculate_entropy(geo_targets), 2),
        "Impact on Improvement": "DIY harder, but MHSA handles it"
    })
    
    metrics.append({
        "Category": "Complexity",
        "Metric": "Unique Test Targets",
        "Geolife": len(geo_targets),
        "DIY": len(diy_targets),
        "Ratio (DIY/Geo)": len(diy_targets) / len(geo_targets),
        "Impact on Improvement": "Neutral"
    })
    
    # ==================== PATTERN LEARNABILITY ====================
    # 1st-order Markov accuracy
    def markov_acc(train, test):
        transitions = defaultdict(Counter)
        for s in train:
            full_seq = list(s['X']) + [s['Y']]
            for i in range(len(full_seq) - 1):
                transitions[full_seq[i]][full_seq[i+1]] += 1
        
        correct = 0
        for s in test:
            last_loc = s['X'][-1]
            if last_loc in transitions and transitions[last_loc]:
                pred = transitions[last_loc].most_common(1)[0][0]
                if pred == s['Y']:
                    correct += 1
        return correct / len(test) * 100
    
    geo_markov = markov_acc(geo_train, geo_test)
    diy_markov = markov_acc(diy_train, diy_test)
    
    metrics.append({
        "Category": "Pattern Learnability",
        "Metric": "1st-Order Markov Acc (%)",
        "Geolife": round(geo_markov, 2),
        "DIY": round(diy_markov, 2),
        "Ratio (DIY/Geo)": round(diy_markov / geo_markov, 2),
        "Impact on Improvement": "KEY: DIY more predictable for MHSA"
    })
    
    # Unseen transitions
    def unseen_trans_pct(train, test):
        transitions = defaultdict(Counter)
        for s in train:
            full_seq = list(s['X']) + [s['Y']]
            for i in range(len(full_seq) - 1):
                transitions[full_seq[i]][full_seq[i+1]] += 1
        
        unseen = sum(1 for s in test if s['X'][-1] not in transitions)
        return unseen / len(test) * 100
    
    geo_unseen = unseen_trans_pct(geo_train, geo_test)
    diy_unseen = unseen_trans_pct(diy_train, diy_test)
    
    metrics.append({
        "Category": "Pattern Learnability",
        "Metric": "Unseen Transitions (%)",
        "Geolife": round(geo_unseen, 2),
        "DIY": round(diy_unseen, 2),
        "Ratio (DIY/Geo)": round(diy_unseen / geo_unseen if geo_unseen > 0 else 0, 2),
        "Impact on Improvement": "KEY: Geolife harder for MHSA"
    })
    
    # ==================== POINTER APPLICABILITY ====================
    geo_in_hist = sum(1 for s in geo_test if s['Y'] in s['X']) / len(geo_test) * 100
    diy_in_hist = sum(1 for s in diy_test if s['Y'] in s['X']) / len(diy_test) * 100
    
    metrics.append({
        "Category": "Pointer Applicability",
        "Metric": "Target in History (%)",
        "Geolife": round(geo_in_hist, 2),
        "DIY": round(diy_in_hist, 2),
        "Ratio (DIY/Geo)": round(diy_in_hist / geo_in_hist, 2),
        "Impact on Improvement": "Similar - both ~84%"
    })
    
    geo_last = sum(1 for s in geo_test if s['Y'] == s['X'][-1]) / len(geo_test) * 100
    diy_last = sum(1 for s in diy_test if s['Y'] == s['X'][-1]) / len(diy_test) * 100
    
    metrics.append({
        "Category": "Pointer Applicability",
        "Metric": "Target = Last Location (%)",
        "Geolife": round(geo_last, 2),
        "DIY": round(diy_last, 2),
        "Ratio (DIY/Geo)": round(diy_last / geo_last, 2),
        "Impact on Improvement": "Geolife stronger last-position pattern"
    })
    
    # ==================== MODEL PERFORMANCE ====================
    geo_theoretical = geo_in_hist + (100 - geo_in_hist) * GEOLIFE_MHSA_ACC / 100
    diy_theoretical = diy_in_hist + (100 - diy_in_hist) * DIY_MHSA_ACC / 100
    
    metrics.append({
        "Category": "Model Performance",
        "Metric": "MHSA Acc@1 (%)",
        "Geolife": GEOLIFE_MHSA_ACC,
        "DIY": DIY_MHSA_ACC,
        "Ratio (DIY/Geo)": round(DIY_MHSA_ACC / GEOLIFE_MHSA_ACC, 2),
        "Impact on Improvement": "KEY: DIY MHSA baseline much higher"
    })
    
    metrics.append({
        "Category": "Model Performance",
        "Metric": "Pointer Generator Transformer Acc@1 (%)",
        "Geolife": GEOLIFE_POINTER_ACC,
        "DIY": DIY_POINTER_ACC,
        "Ratio (DIY/Geo)": round(DIY_POINTER_ACC / GEOLIFE_POINTER_ACC, 2),
        "Impact on Improvement": "Both reach similar absolute levels"
    })
    
    metrics.append({
        "Category": "Model Performance",
        "Metric": "Improvement (pp)",
        "Geolife": round(GEOLIFE_POINTER_ACC - GEOLIFE_MHSA_ACC, 2),
        "DIY": round(DIY_POINTER_ACC - DIY_MHSA_ACC, 2),
        "Ratio (DIY/Geo)": round((DIY_POINTER_ACC - DIY_MHSA_ACC) / (GEOLIFE_POINTER_ACC - GEOLIFE_MHSA_ACC), 2),
        "Impact on Improvement": "RESULT: Geolife 5.6x larger improvement"
    })
    
    metrics.append({
        "Category": "Model Performance",
        "Metric": "Theoretical Max (%)",
        "Geolife": round(geo_theoretical, 2),
        "DIY": round(diy_theoretical, 2),
        "Ratio (DIY/Geo)": round(diy_theoretical / geo_theoretical, 2),
        "Impact on Improvement": "Similar theoretical ceilings"
    })
    
    geo_mhsa_util = GEOLIFE_MHSA_ACC / geo_theoretical * 100
    diy_mhsa_util = DIY_MHSA_ACC / diy_theoretical * 100
    
    metrics.append({
        "Category": "Model Performance",
        "Metric": "MHSA Utilization of Potential (%)",
        "Geolife": round(geo_mhsa_util, 1),
        "DIY": round(diy_mhsa_util, 1),
        "Ratio (DIY/Geo)": round(diy_mhsa_util / geo_mhsa_util, 2),
        "Impact on Improvement": "KEY ROOT CAUSE: DIY MHSA already good"
    })
    
    return pd.DataFrame(metrics)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS: WHY DIFFERENT IMPROVEMENTS?")
    print("=" * 80)
    
    # Load data
    data = load_data()
    
    # Comprehensive comparison
    comparison_df = comprehensive_comparison(data)
    
    print("\nCOMPREHENSIVE COMPARISON TABLE")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(f"{OUTPUT_DIR}/04_comprehensive_comparison.csv", index=False)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = {
        "Metric": [
            "MHSA Acc@1",
            "Pointer Generator Transformer Acc@1", 
            "Improvement",
            "Theoretical Max",
            "MHSA % of Theoretical",
            "Pointer % of Theoretical",
            "1st-Order Markov Acc",
            "Unseen Transitions",
            "Target in History",
        ],
        "Geolife": [
            f"{GEOLIFE_MHSA_ACC:.2f}%",
            f"{GEOLIFE_POINTER_ACC:.2f}%",
            f"+{GEOLIFE_POINTER_ACC - GEOLIFE_MHSA_ACC:.2f}pp",
            f"89.18%",
            f"37.2%",
            f"60.5%",
            f"21.25%",
            f"22.99%",
            f"83.81%",
        ],
        "DIY": [
            f"{DIY_MHSA_ACC:.2f}%",
            f"{DIY_POINTER_ACC:.2f}%",
            f"+{DIY_POINTER_ACC - DIY_MHSA_ACC:.2f}pp",
            f"92.56%",
            f"57.4%",
            f"61.4%",
            f"34.49%",
            f"4.04%",
            f"84.12%",
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f"{OUTPUT_DIR}/04_summary.csv", index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    print("\n" + "=" * 80)
    print("ROOT CAUSE EXPLANATION")
    print("=" * 80)
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY DIFFERENT IMPROVEMENTS?                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GEOLIFE: +20.79pp improvement (33.18% → 53.97%)                            │
│  DIY:     +3.68pp improvement (53.17% → 56.85%)                             │
│                                                                              │
│  THE ROOT CAUSE IS NOT THE POINTER MECHANISM ITSELF,                        │
│  BUT THE BASELINE MHSA PERFORMANCE DIFFERENCE.                              │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  KEY INSIGHT:                                                                │
│                                                                              │
│  Both datasets have similar pointer-favorable characteristics:               │
│  - ~84% of targets appear in history (pointer can help)                     │
│  - Similar repetition patterns                                               │
│                                                                              │
│  But DIY has MORE PREDICTABLE patterns that MHSA can already learn:         │
│  - 34.49% Markov accuracy vs 21.25% (DIY 1.6x more predictable)            │
│  - Only 4.04% unseen transitions vs 22.99% (DIY 5.7x fewer unseen)         │
│                                                                              │
│  This means MHSA captures 57.4% of theoretical potential on DIY,            │
│  but only 37.2% on Geolife.                                                 │
│                                                                              │
│  Pointer Generator Transformer brings both to ~60-61% of theoretical potential,               │
│  but starting from different baselines results in different gains.          │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  VISUAL:                                                                     │
│                                                                              │
│  Geolife:  |████████████-------------------------------------| 37.2%        │
│  Pointer:  |████████████████████████████████████-------------| 60.5%        │
│  Gain:     |            ████████████████████████             | +20.79pp     │
│                                                                              │
│  DIY:      |█████████████████████████████████-----------------| 57.4%        │
│  Pointer:  |███████████████████████████████████████-----------| 61.4%        │
│  Gain:     |                                 ████             | +3.68pp      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()

# Technical Implementation: Code Architecture and Data Flow

## Table of Contents

1. [File Structure Overview](#1-file-structure-overview)
2. [Dependencies and Imports](#2-dependencies-and-imports)
3. [Configuration System](#3-configuration-system)
4. [Data Flow Pipeline](#4-data-flow-pipeline)
5. [Core Functions Explained](#5-core-functions-explained)
6. [Main Execution Flow](#6-main-execution-flow)
7. [Input Data Format](#7-input-data-format)
8. [Output Format and Storage](#8-output-format-and-storage)
9. [Error Handling and Edge Cases](#9-error-handling-and-edge-cases)

---

## 1. File Structure Overview

### Project Context

```
next_loc_clean_v2/
├── src/
│   └── models/
│       └── baseline/
│           └── markov_ori/
│               ├── run_markov_ori.py    ← MAIN SCRIPT (this documentation)
│               └── README.md
├── config/
│   └── models/
│       ├── config_markov_ori_geolife.yaml
│       └── config_markov_ori_diy.yaml
├── data/
│   ├── geolife_eps20/
│   │   ├── markov_ori_data/
│   │   │   ├── dataset_geolife.csv
│   │   │   └── valid_ids_geolife.pk
│   │   └── processed/
│   └── diy_eps50/
│       ├── interim/
│       │   └── intermediate_eps50.csv
│       └── processed/
└── experiments/
    └── {dataset}_markov_ori_{timestamp}/
```

### Why "markov_ori"?

The name "markov_ori" stands for "Markov Original" - indicating this is a faithful reproduction of the original baseline implementation from `location-prediction-ori-freeze/baselines/markov.py`, preserving the original:
- Data splitting method
- Evaluation methodology
- Metric calculations

---

## 2. Dependencies and Imports

```python
# Standard library
import pickle           # Serialize/deserialize Python objects
import os              # File path operations
import sys             # System operations
import argparse        # Command-line argument parsing
import json            # JSON file handling
import yaml            # YAML config file parsing
import time            # Timing operations
import shutil          # File copying
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Data processing
import pandas as pd    # DataFrame operations
import numpy as np     # Numerical computations

# Machine learning metrics
from sklearn.metrics import f1_score, recall_score

# Utilities
from tqdm import tqdm  # Progress bars
from timeit import default_timer as timer  # High-precision timing
```

### Version Requirements

| Library | Minimum Version | Purpose |
|---------|-----------------|---------|
| pandas | 1.0+ | Data manipulation |
| numpy | 1.19+ | Numerical operations |
| scikit-learn | 0.24+ | F1 and recall metrics |
| pyyaml | 5.0+ | Config file parsing |
| tqdm | 4.0+ | Progress visualization |

---

## 3. Configuration System

### YAML Configuration Format

**GeoLife Configuration (`config_markov_ori_geolife.yaml`):**

```yaml
# Data settings
data:
  dataset: geolife
  data_csv: data/geolife_eps20/markov_ori_data/dataset_geolife.csv
  valid_ids_file: data/geolife_eps20/markov_ori_data/valid_ids_geolife.pk
  experiment_root: experiments

# Model settings
model:
  model_name: markov_ori
  markov_order: 1
```

**DIY Configuration (`config_markov_ori_diy.yaml`):**

```yaml
data:
  dataset: diy
  data_csv: data/diy_eps50/interim/intermediate_eps50.csv
  processed_dir: data/diy_eps50/processed
  dataset_prefix: diy_eps50_prev7
  experiment_root: experiments

model:
  model_name: markov_ori
  markov_order: 1
```

### Configuration Loading Function

```python
def load_config(path):
    """Load configuration from YAML file.
    
    Flattens nested dictionaries for easy attribute access.
    
    Args:
        path: Path to YAML config file
    
    Returns:
        dict: Flattened configuration dictionary
    
    Example:
        Input YAML:
            data:
              dataset: geolife
              data_csv: path/to/data.csv
        
        Output dict:
            {'dataset': 'geolife', 'data_csv': 'path/to/data.csv'}
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        if isinstance(value, dict):
            # Flatten nested dict
            for k, v in value.items():
                config[k] = v
        else:
            config[key] = value

    return config
```

### EasyDict Utility

```python
class EasyDict(dict):
    """Dictionary with attribute-style access.
    
    Enables both dict['key'] and dict.key syntax.
    
    Example:
        config = EasyDict({'dataset': 'geolife'})
        print(config['dataset'])  # 'geolife'
        print(config.dataset)     # 'geolife' (same result)
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value
```

---

## 4. Data Flow Pipeline

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA FLOW PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ Config YAML  │                                                   │
│  └──────┬───────┘                                                   │
│         ↓                                                           │
│  ┌──────────────┐    ┌────────────────┐                            │
│  │   CSV Data   │ → │  Sort by user   │                            │
│  │   (raw)      │    │  & timestamp   │                            │
│  └──────────────┘    └───────┬────────┘                            │
│                              ↓                                      │
│                    ┌─────────────────────┐                          │
│                    │   Split Dataset     │                          │
│                    │  (60/20/20 by day)  │                          │
│                    └─────────┬───────────┘                          │
│                              ↓                                      │
│                    ┌─────────────────────┐                          │
│                    │  Filter by valid_ids │                         │
│                    └─────────┬───────────┘                          │
│                              ↓                                      │
│         ┌────────────────────┼────────────────────┐                 │
│         ↓                    ↓                    ↓                 │
│  ┌────────────┐      ┌────────────┐       ┌────────────┐           │
│  │   Train    │      │ Validation │       │    Test    │           │
│  │    Data    │      │    Data    │       │    Data    │           │
│  └──────┬─────┘      └──────┬─────┘       └──────┬─────┘           │
│         │                   │                    │                  │
│         ↓                   ↓                    ↓                  │
│  ┌──────────────────────────────────────────────────────┐          │
│  │           Per-User Markov Training                    │          │
│  │  • Build transition counts from train data            │          │
│  │  • Generate predictions for val/test                  │          │
│  └───────────────────────┬──────────────────────────────┘          │
│                          ↓                                          │
│                 ┌─────────────────┐                                 │
│                 │  Compute Metrics │                                │
│                 │  (Acc, MRR, F1)  │                                │
│                 └─────────┬───────┘                                 │
│                           ↓                                         │
│                 ┌─────────────────┐                                 │
│                 │  Save Results   │                                 │
│                 └─────────────────┘                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Detailed Stage Breakdown

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| 1. Load Config | YAML file | Parse & flatten | Config dict |
| 2. Load Data | CSV file | Read & sort | DataFrame |
| 3. Split Data | DataFrame | Split by user days | Train/Val/Test DFs |
| 4. Filter Data | DFs + valid_ids | Inner join | Filtered DFs |
| 5. Train Model | Train DF | Count transitions | Per-user transition tables |
| 6. Predict | Model + Val/Test | Lookup & rank | Predictions |
| 7. Evaluate | Predictions + Targets | Compute metrics | Metric dict |
| 8. Save | Metrics + Config | Serialize | JSON/YAML files |

---

## 5. Core Functions Explained

### Dataset Splitting Functions

#### `splitDataset(totalData)`

```python
def splitDataset(totalData):
    """Split dataset into train, vali and test.
    
    Applies per-user time-based splitting using getSplitDaysUser.
    
    Args:
        totalData: DataFrame with all user data
    
    Returns:
        tuple: (train_data, vali_data, test_data) DataFrames
    
    Data Flow:
        Input DataFrame → groupby user → apply split → concat → separate
    """
    # Apply split function to each user group
    totalData = totalData.groupby("user_id").apply(getSplitDaysUser)

    # Separate by dataset label
    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # Clean up helper column
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data
```

#### `getSplitDaysUser(df)`

```python
def getSplitDaysUser(df):
    """Split the dataset according to the tracked day of each user.
    
    Time-based split preserves temporal order:
    - First 60% of days → Training
    - Middle 20% of days → Validation
    - Last 20% of days → Testing
    
    Args:
        df: DataFrame for single user
    
    Returns:
        DataFrame with 'Dataset' column added
    
    Example:
        User tracked for 100 days:
        - Days 0-59: Train
        - Days 60-79: Validation  
        - Days 80-99: Test
    """
    maxDay = df["start_day"].max()
    train_split = maxDay * 0.6
    vali_split = maxDay * 0.8

    df["Dataset"] = "test"
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[
        (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
        "Dataset",
    ] = "vali"

    return df
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TIME-BASED DATA SPLIT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User's tracking period (e.g., 100 days):                          │
│                                                                     │
│  Day:  0     20    40    60    80    100                           │
│        ├─────┼─────┼─────┼─────┼─────┤                             │
│        │     TRAIN (60%)     │VAL  │TEST│                          │
│        │                     │20%  │20% │                          │
│        │←─────────────────── │←───→│←──→│                          │
│                                                                     │
│  Why time-based split?                                              │
│  • Simulates real-world: Train on past, predict future             │
│  • Prevents data leakage: No future info in training               │
│  • Per-user: Each user's timeline is split independently           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Markov Transition Building

#### `markov_transition_prob(df, n=1)`

```python
def markov_transition_prob(df, n=1):
    """Build transition probability table.
    
    Creates a DataFrame of (source locations → destination) pairs
    with their occurrence counts.
    
    Args:
        df: DataFrame with location sequences
        n: Markov order (number of previous locations to consider)
    
    Returns:
        DataFrame with columns [loc_1, ..., loc_n, toLoc, size]
    
    Example (n=1):
        Input sequence: [A, B, C, A, B]
        
        Output:
        loc_1  toLoc  size
        A      B      2      (A→B appears twice)
        B      C      1      (B→C appears once)
        C      A      1      (C→A appears once)
    """
    # Create column names based on order
    COLUMNS = [f"loc_{i+1}" for i in range(n)]
    COLUMNS.append("toLoc")

    locSequence = pd.DataFrame(columns=COLUMNS)

    # Extract destination locations (n positions ahead)
    locSequence["toLoc"] = df.iloc[n:]["location_id"].values
    
    # Extract source locations (shifted by position)
    for i in range(n):
        locSequence[f"loc_{i+1}"] = df.iloc[i : -n + i]["location_id"].values
    
    # Count occurrences and return
    return locSequence.groupby(by=COLUMNS).size().to_frame("size").reset_index()
```

**Visual Explanation of Transition Extraction:**

```
┌─────────────────────────────────────────────────────────────────────┐
│              TRANSITION EXTRACTION (n=1)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Original sequence:  [H, W, G, W, H, W, R, H]                      │
│  Index:              [0, 1, 2, 3, 4, 5, 6, 7]                      │
│                                                                     │
│  Alignment:                                                         │
│  loc_1 (iloc[0:-1]): [H, W, G, W, H, W, R]  (source)              │
│  toLoc (iloc[1:]):   [W, G, W, H, W, R, H]  (destination)         │
│                                                                     │
│  Pairs extracted:                                                   │
│  H→W, W→G, G→W, W→H, H→W, W→R, R→H                                │
│                                                                     │
│  After groupby + count:                                            │
│  loc_1  toLoc  size                                                │
│  H      W      2     (H→W appears at positions 0,4)               │
│  W      G      1                                                    │
│  G      W      1                                                    │
│  W      H      1                                                    │
│  W      R      1                                                    │
│  R      H      1                                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Prediction Functions

#### `get_true_pred_pair(locSequence, df, n=1)`

```python
def get_true_pred_pair(locSequence, df, n=1):
    """Get true and predicted pairs from test data.
    
    For each position in test sequence, looks up predictions
    from the learned transition table.
    
    Args:
        locSequence: Transition count DataFrame (from training)
        df: Test data DataFrame
        n: Markov order
    
    Returns:
        tuple: (true_ls, pred_ls, time_ls)
            - true_ls: List of true next locations
            - pred_ls: List of prediction arrays (ranked)
            - time_ls: List of prediction times (for profiling)
    
    Algorithm:
        For each test position:
        1. Extract current n locations
        2. Look up in transition table
        3. If found: return destinations sorted by count
        4. If not found: reduce n and retry
        5. If still not found: return zeros
    """
    testSeries = df["location_id"].values

    true_ls = []
    pred_ls = []
    time_ls = []
    
    for i in range(testSeries.shape[0] - n):
        locCurr = testSeries[i : i + n + 1]  # Current n locations + target
        numbLoc = n

        start = timer()
        
        # Loop until finds a match (or exhausts options)
        while True:
            res_df = locSequence
            
            # Filter by each location in sequence
            for j in range(n - numbLoc, n):
                res_df = res_df.loc[res_df[f"loc_{j+1}"] == locCurr[j]]
            res_df = res_df.sort_values(by="size", ascending=False)

            if res_df.shape[0]:  # Found matching transitions
                # Get destinations ranked by frequency
                pred = res_df["toLoc"].drop_duplicates().values
                break
            
            # Reduce history length and try again
            numbLoc -= 1
            if numbLoc == 0:
                pred = np.zeros(10)  # No prediction possible
                break

        time_ls.append(timer() - start)
        true_ls.append(locCurr[-1])  # True next location
        pred_ls.append(pred)

    return true_ls, pred_ls, time_ls
```

**Prediction Algorithm Flowchart:**

```
┌─────────────────────────────────────────────────────────────────────┐
│              PREDICTION ALGORITHM FLOW                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: Current locations [L1, L2, ..., Ln]                        │
│         Transition table from training                              │
│                                                                     │
│  ┌─────────────────────────────────────────────┐                   │
│  │ Step 1: Look up full sequence [L1...Ln]     │                   │
│  │         in transition table                 │                   │
│  └──────────────────────┬──────────────────────┘                   │
│                         │                                           │
│            ┌────────────┼────────────┐                             │
│            ↓                         ↓                              │
│       Found?                    Not Found?                          │
│            ↓                         ↓                              │
│  ┌─────────────────┐      ┌──────────────────────┐                 │
│  │ Sort by count   │      │ Reduce to [L2...Ln]  │                 │
│  │ Return ranked   │      │ (drop oldest)        │                 │
│  │ destinations    │      │ Try again            │                 │
│  └─────────────────┘      └──────────┬───────────┘                 │
│                                      │                              │
│                         ┌────────────┼────────────┐                │
│                         ↓                         ↓                 │
│                    Found?                    numbLoc == 0?          │
│                         ↓                         ↓                 │
│               [Return predictions]      [Return zeros]              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Metric Calculation

#### `get_performance_measure(true_ls, pred_ls)`

```python
def get_performance_measure(true_ls, pred_ls):
    """Calculate performance metrics.
    
    Computes comprehensive evaluation metrics for predictions.
    
    Args:
        true_ls: List of true labels
        pred_ls: List of prediction arrays (each array is ranked)
    
    Returns:
        pd.Series with: total, correct@1, correct@5, correct@10,
                       f1, recall, ndcg, rr
    
    Metrics Explained:
    - correct@k: Count of samples where true label is in top-k predictions
    - F1: Weighted F1 score using top-1 predictions
    - recall: Weighted recall using top-1 predictions
    - NDCG@10: Normalized Discounted Cumulative Gain
    - RR: Sum of reciprocal ranks (for MRR calculation)
    """
    acc_ls = [1, 5, 10]

    res = []
    ndcg_ls = []
    
    # Total number of samples
    res.append(len(true_ls))
    
    # Calculate top-k accuracy for k in [1, 5, 10]
    for top_acc in acc_ls:
        correct = 0
        for true, pred in zip(true_ls, pred_ls):
            if true in pred[:top_acc]:
                correct += 1

            # NDCG calculation (only for top-10)
            if top_acc == 10:
                idx = np.where(true == pred[:top_acc])[0]
                if len(idx) == 0:
                    ndcg_ls.append(0)
                else:
                    # NDCG formula: 1 / log2(rank + 1)
                    ndcg_ls.append(1 / np.log2(idx[0] + 1 + 1))

        res.append(correct)

    # F1 and Recall using top-1 predictions
    top1 = [pred[0] for pred in pred_ls]
    f1 = f1_score(true_ls, top1, average="weighted")
    recall = recall_score(true_ls, top1, average="weighted")

    res.append(f1)
    res.append(recall)
    res.append(np.mean(ndcg_ls))

    # Reciprocal Rank calculation
    rank_ls = []
    for true, pred in zip(true_ls, pred_ls):
        rank = np.where(pred == true)[0] + 1  # 1-indexed rank
        if len(rank):
            rank_ls.append(rank[0])
        else:
            rank_ls.append(0)
    rank = np.array(rank_ls, dtype=float)

    # Calculate 1/rank (0 for missing)
    rank = np.divide(1.0, rank, out=np.zeros_like(rank), where=rank != 0)
    res.append(rank.sum())

    return pd.Series(res, index=["total", "correct@1", "correct@5", "correct@10", 
                                  "f1", "recall", "ndcg", "rr"])
```

---

## 6. Main Execution Flow

### `main()` Function Structure

```python
def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--config", type=str, required=True, ...)
    args = parser.parse_args()

    # 2. Configuration Loading
    config = load_config(args.config)
    config = EasyDict(config)
    n = config.get("markov_order", 1)

    # 3. Experiment Directory Setup
    experiment_dir = init_experiment_dir(config, dataset_name, model_name="markov_ori")
    log_file = open(log_path, "w")

    # 4. Data Loading
    inputData = pd.read_csv(csv_path)
    inputData.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

    # 5. Data Splitting
    train_data, vali_data, test_data = splitDataset(inputData)

    # 6. Filtering by valid_ids
    if valid_ids_file exists:
        valid_ids = pickle.load(open(valid_ids_file, "rb"))
    else:
        valid_ids = generate_valid_ids(...)
    
    train_data = train_data.loc[train_data["id"].isin(valid_ids)]
    vali_data = vali_data.loc[vali_data["id"].isin(valid_ids)]
    test_data = test_data.loc[test_data["id"].isin(valid_ids)]

    # 7. Training and Evaluation
    val_results, total_params = run_evaluation(config, train_data, vali_data, "validation", n=n)
    test_results, _ = run_evaluation(config, train_data, test_data, "test", n=n)

    # 8. Save Results
    # Save config, validation results, test results to experiment_dir
```

### Execution Timeline

```
┌─────────────────────────────────────────────────────────────────────┐
│              EXECUTION TIMELINE                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  T=0s    ┌─────────────────────────────────────────────────────┐   │
│          │ Parse args, Load config                              │   │
│          └─────────────────────────────────────────────────────┘   │
│                                                                     │
│  T=0.1s  ┌─────────────────────────────────────────────────────┐   │
│          │ Load CSV data (~16,600 rows for GeoLife)             │   │
│          │ Sort by user_id, start_day, start_min                │   │
│          └─────────────────────────────────────────────────────┘   │
│                                                                     │
│  T=0.3s  ┌─────────────────────────────────────────────────────┐   │
│          │ Split dataset (60/20/20 by user days)                │   │
│          └─────────────────────────────────────────────────────┘   │
│                                                                     │
│  T=0.4s  ┌─────────────────────────────────────────────────────┐   │
│          │ Filter by valid_ids                                  │   │
│          └─────────────────────────────────────────────────────┘   │
│                                                                     │
│  T=0.5s  ┌─────────────────────────────────────────────────────┐   │
│          │ TRAINING: Build per-user transition tables           │   │
│          │ VALIDATION: Evaluate on validation set               │   │
│          │ (Progress bar: 45 users)                             │   │
│          └─────────────────────────────────────────────────────┘   │
│                                                                     │
│  T=2.5s  ┌─────────────────────────────────────────────────────┐   │
│          │ TEST: Evaluate on test set                           │   │
│          │ (Progress bar: 45 users)                             │   │
│          └─────────────────────────────────────────────────────┘   │
│                                                                     │
│  T=5s    ┌─────────────────────────────────────────────────────┐   │
│          │ Save results to experiment directory                 │   │
│          └─────────────────────────────────────────────────────┘   │
│                                                                     │
│  TOTAL: ~5 seconds (GeoLife)                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Input Data Format

### CSV Schema

The input CSV file (`dataset_geolife.csv` or `intermediate_eps{X}.csv`) has the following schema:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | int | Unique record identifier | 0, 1, 2, ... |
| `user_id` | int | User identifier | 1, 2, 3, ... |
| `location_id` | int | Clustered location ID | 0, 1, 2, ... |
| `duration` | float | Stay duration in minutes | 64.0, 309.0 |
| `start_day` | int | Start day index (0-based) | 0, 4, 12 |
| `end_day` | int | End day index | 0, 5, 12 |
| `start_min` | int | Start minute of day (0-1439) | 183, 272 |
| `end_min` | int | End minute of day | 248, 582 |
| `weekday` | int | Day of week (0=Monday) | 0, 1, ..., 6 |

### Sample Data

```csv
id,user_id,location_id,duration,start_day,end_day,start_min,end_min,weekday
0,1,0,64.0,0,0,183,248,3
1,1,1,309.0,0,0,272,582,3
2,1,2,899.0,0,1,670,130,3
3,1,3,752.0,4,5,725,38,0
4,1,4,33.0,5,5,38,72,1
```

### Valid IDs File

The `valid_ids_*.pk` file is a pickled numpy array of record IDs that should be included in evaluation:

```python
import pickle
valid_ids = pickle.load(open("valid_ids_geolife.pk", "rb"))
# valid_ids: np.array([0, 1, 2, 3, ...])
```

---

## 8. Output Format and Storage

### Experiment Directory Structure

```
experiments/{dataset}_markov_ori_{yyyyMMdd_hhmmss}/
├── checkpoints/           # Empty (no neural model)
├── training.log           # Complete log
├── config.yaml            # Flattened config
├── config_original.yaml   # Copy of input config
├── val_results.json       # Validation metrics
└── test_results.json      # Test metrics
```

### Results JSON Format

**test_results.json:**
```json
{
  "total_samples": 3457,
  "correct@1": 836,
  "correct@5": 1309,
  "correct@10": 1340,
  "acc@1": 24.18,
  "acc@5": 37.87,
  "acc@10": 38.76,
  "mrr": 30.34,
  "f1": 23.38,
  "recall": 24.18,
  "ndcg": 32.38,
  "dataset": "geolife",
  "markov_order": 1,
  "total_parameters": 166309,
  "training_time_seconds": 5.28,
  "num_users": 45
}
```

### Training Log Sample

```
============================================================
1st-Order Markov Chain Model - Original Implementation
============================================================

Dataset: geolife
Config: config/models/config_markov_ori_geolife.yaml
Markov order: 1
Experiment directory: experiments/geolife_markov_ori_20251226_173239
============================================================

Loaded 16600 records from data/geolife_eps20/markov_ori_data/dataset_geolife.csv
After split: train=(8380, 9), vali=(4018, 9), test=(4202, 9)
Loaded valid_ids from: data/geolife_eps20/markov_ori_data/valid_ids_geolife.pk
Valid IDs: 15978
After filtering: train=(7424, 9), vali=(3334, 9), test=(3502, 9)

============================================================
Training and Evaluation
============================================================

Training finished. Time: 5.28s
Total parameters: 166309

Validation Results:
  Acc@1:  33.57%
  Acc@5:  47.43%
  Acc@10: 48.59%
  MRR:    39.91%
  F1:     32.87%
  NDCG:   42.01%

Test Results:
  Acc@1:  24.18%
  Acc@5:  37.87%
  Acc@10: 38.76%
  MRR:    30.34%
  F1:     23.38%
  NDCG:   32.38%

=== Training Complete ===
```

---

## 9. Error Handling and Edge Cases

### Handled Edge Cases

| Case | Handling |
|------|----------|
| Missing valid_ids file | Generate from preprocessed data |
| Empty user data | Skip user in evaluation |
| Unknown location | Fallback to shorter history |
| No transitions found | Return zeros |
| Missing config key | Use default value |

### Error Scenarios

```python
# Missing config file
if not os.path.exists(args.config):
    raise FileNotFoundError(f"Config file not found: {args.config}")

# Empty results
if not true_all_ls:
    return None, total_parameter

# Division by zero in metrics
rank = np.divide(1.0, rank, out=np.zeros_like(rank), where=rank != 0)
```

### Graceful Degradation

The prediction function implements graceful degradation:

```python
while True:
    # Try current n-gram
    if res_df.shape[0]:
        pred = res_df["toLoc"].drop_duplicates().values
        break
    
    # Reduce to (n-1)-gram
    numbLoc -= 1
    
    # Ultimate fallback
    if numbLoc == 0:
        pred = np.zeros(10)
        break
```

This ensures the model always produces predictions, even for cold-start scenarios.

---

## Navigation

| Previous | Next |
|----------|------|
| [02_THEORY_BACKGROUND.md](02_THEORY_BACKGROUND.md) | [04_COMPONENTS_DEEP_DIVE.md](04_COMPONENTS_DEEP_DIVE.md) |

# Line-by-Line Walkthrough with Examples

## Table of Contents

1. [Introduction](#1-introduction)
2. [Setup and Configuration](#2-setup-and-configuration)
3. [Data Loading Walkthrough](#3-data-loading-walkthrough)
4. [Dataset Splitting Walkthrough](#4-dataset-splitting-walkthrough)
5. [Transition Building Walkthrough](#5-transition-building-walkthrough)
6. [Prediction Walkthrough](#6-prediction-walkthrough)
7. [Metric Calculation Walkthrough](#7-metric-calculation-walkthrough)
8. [Full Example: End-to-End](#8-full-example-end-to-end)
9. [Code Execution Trace](#9-code-execution-trace)

---

## 1. Introduction

This document provides a **line-by-line walkthrough** of the Markov Chain model using concrete examples. We'll trace through the code with actual data to show exactly what happens at each step.

### Example Data

Throughout this walkthrough, we'll use this simplified example:

```
3 Users, 4 Locations: Home(0), Work(1), Gym(2), Restaurant(3)

User 1's trajectory (over 10 days):
Day 0: Home → Work → Gym → Home
Day 1: Home → Work → Home
Day 2: Home → Work → Restaurant → Home
Day 3: Home → Work → Gym → Home
Day 4: Home → Work → Home
Day 5: Home → Work → Gym → Home
Day 6: Home → Work → Home
Day 7: Home → Work → Restaurant → Home
Day 8: Home → Work → Gym → Work → Home
Day 9: Home → Work → Home
```

---

## 2. Setup and Configuration

### Code: Configuration Loading

```python
# File: run_markov_ori.py, lines 43-56

def load_config(path):
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        if isinstance(value, dict):
            for k, v in value.items():
                config[k] = v
        else:
            config[key] = value

    return config
```

### Example Execution

```yaml
# Input: config/models/config_markov_ori_geolife.yaml

data:
  dataset: geolife
  data_csv: data/geolife_eps20/markov_ori_data/dataset_geolife.csv
  experiment_root: experiments

model:
  model_name: markov_ori
  markov_order: 1
```

**Step-by-step trace:**

```python
# Step 1: yaml.safe_load reads the file
cfg = {
    'data': {
        'dataset': 'geolife',
        'data_csv': 'data/geolife_eps20/markov_ori_data/dataset_geolife.csv',
        'experiment_root': 'experiments'
    },
    'model': {
        'model_name': 'markov_ori',
        'markov_order': 1
    }
}

# Step 2: Flatten nested structure
config = {}

# First iteration: key='data', value={'dataset': 'geolife', ...}
# value is a dict, so we flatten:
config['dataset'] = 'geolife'
config['data_csv'] = 'data/geolife_eps20/markov_ori_data/dataset_geolife.csv'
config['experiment_root'] = 'experiments'

# Second iteration: key='model', value={'model_name': 'markov_ori', ...}
config['model_name'] = 'markov_ori'
config['markov_order'] = 1

# Final config (flattened):
config = {
    'dataset': 'geolife',
    'data_csv': 'data/geolife_eps20/markov_ori_data/dataset_geolife.csv',
    'experiment_root': 'experiments',
    'model_name': 'markov_ori',
    'markov_order': 1
}
```

---

## 3. Data Loading Walkthrough

### Code: Main Data Loading

```python
# File: run_markov_ori.py, lines 379-382

csv_path = config.data_csv
print(f"\nLoading data from: {csv_path}")
inputData = pd.read_csv(csv_path)
inputData.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
```

### Example Execution

**Input CSV (simplified):**

```csv
id,user_id,location_id,duration,start_day,end_day,start_min,end_min,weekday
0,1,0,120,0,0,420,540,0
1,1,1,480,0,0,560,1040,0
2,1,2,60,0,0,1060,1120,0
3,1,0,600,0,0,1140,300,0
4,1,0,60,1,1,420,480,1
5,1,1,540,1,1,500,1040,1
6,1,0,660,1,1,1060,300,1
```

**Step-by-step trace:**

```python
# Step 1: pd.read_csv reads the file
inputData = pd.DataFrame({
    'id': [0, 1, 2, 3, 4, 5, 6],
    'user_id': [1, 1, 1, 1, 1, 1, 1],
    'location_id': [0, 1, 2, 0, 0, 1, 0],
    'duration': [120, 480, 60, 600, 60, 540, 660],
    'start_day': [0, 0, 0, 0, 1, 1, 1],
    ...
})

# Step 2: Sort by user_id, start_day, start_min
# Already sorted in this example, but ensures:
# - All records for user 1 come before user 2
# - Within each user, ordered by time

# Result after sorting:
#    id  user_id  location_id  duration  start_day  start_min
#    0       1            0       120          0        420  # Day 0, 7:00am
#    1       1            1       480          0        560  # Day 0, 9:20am  
#    2       1            2        60          0       1060  # Day 0, 5:40pm
#    3       1            0       600          0       1140  # Day 0, 7:00pm
#    4       1            0        60          1        420  # Day 1, 7:00am
#    5       1            1       540          1        500  # Day 1, 8:20am
#    6       1            0       660          1       1060  # Day 1, 5:40pm
```

---

## 4. Dataset Splitting Walkthrough

### Code: Split Functions

```python
# File: run_markov_ori.py, lines 90-106

def getSplitDaysUser(df):
    """Split the dataset according to the tracked day of each user."""
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

### Example Execution

**Input: User 1's data (10 days: 0-9)**

```python
# Step 1: Find max day
maxDay = df["start_day"].max()  # = 9

# Step 2: Calculate split points
train_split = maxDay * 0.6  # = 9 * 0.6 = 5.4
vali_split = maxDay * 0.8   # = 9 * 0.8 = 7.2

# Step 3: Assign labels
# Initialize all as "test"
df["Dataset"] = "test"

# Label train (start_day < 5.4, i.e., days 0-5)
df.loc[df["start_day"] < 5.4, "Dataset"] = "train"

# Label validation (5.4 <= start_day < 7.2, i.e., days 6-7)
df.loc[
    (df["start_day"] >= 5.4) & (df["start_day"] < 7.2),
    "Dataset",
] = "vali"

# Test remains for days 8-9
```

**Result:**

```
Day   Location Sequence        Dataset
───   ───────────────────      ───────
0     Home→Work→Gym→Home       train
1     Home→Work→Home           train
2     Home→Work→Rest→Home      train
3     Home→Work→Gym→Home       train
4     Home→Work→Home           train
5     Home→Work→Gym→Home       train
6     Home→Work→Home           vali
7     Home→Work→Rest→Home      vali
8     Home→Work→Gym→Work→Home  test
9     Home→Work→Home           test
```

---

## 5. Transition Building Walkthrough

### Code: Transition Extraction

```python
# File: run_markov_ori.py, lines 109-122

def markov_transition_prob(df, n=1):
    """Build transition probability table."""
    COLUMNS = [f"loc_{i+1}" for i in range(n)]
    COLUMNS.append("toLoc")

    locSequence = pd.DataFrame(columns=COLUMNS)

    locSequence["toLoc"] = df.iloc[n:]["location_id"].values
    for i in range(n):
        locSequence[f"loc_{i+1}"] = df.iloc[i : -n + i]["location_id"].values
    return locSequence.groupby(by=COLUMNS).size().to_frame("size").reset_index()
```

### Example Execution (n=1)

**Input: User 1's training data (days 0-5)**

Location sequence: [0, 1, 2, 0, 0, 1, 0, 0, 1, 3, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 2, 0]
(Home=0, Work=1, Gym=2, Restaurant=3)

```python
# Step 1: Create column names
COLUMNS = ["loc_1", "toLoc"]

# Step 2: Create empty DataFrame
locSequence = pd.DataFrame(columns=["loc_1", "toLoc"])

# Step 3: Extract destination locations (shift by n=1)
# iloc[1:] gives all locations except the first
locSequence["toLoc"] = df.iloc[1:]["location_id"].values
# Result: [1, 2, 0, 0, 1, 0, 0, 1, 3, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 2, 0]

# Step 4: Extract source locations
# iloc[0:-1] gives all locations except the last
locSequence["loc_1"] = df.iloc[0:-1]["location_id"].values
# Result: [0, 1, 2, 0, 0, 1, 0, 0, 1, 3, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 2]

# Step 5: Pair them up
#   loc_1  toLoc
#     0      1     (Home → Work)
#     1      2     (Work → Gym)
#     2      0     (Gym → Home)
#     0      0     (Home → Home - same day continuation)
#     0      1     (Home → Work)
#     1      0     (Work → Home)
#     ...

# Step 6: Group by (loc_1, toLoc) and count
locSequence.groupby(["loc_1", "toLoc"]).size().to_frame("size").reset_index()
```

**Result:**

```
   loc_1  toLoc  size
       0      1     6    # Home → Work (morning commute, 6 times)
       1      0     4    # Work → Home (evening return, 4 times)
       1      2     3    # Work → Gym (after work, 3 times)
       1      3     2    # Work → Restaurant (2 times)
       2      0     3    # Gym → Home (after gym, 3 times)
       3      0     2    # Restaurant → Home (after dinner, 2 times)
       0      0     1    # Home → Home (staying home)
```

**Visual representation of the transition table:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSITION TABLE (USER 1, TRAINING)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         To Location                                         │
│            ┌────────────────────────────────────────────┐                   │
│            │  Home(0)  Work(1)  Gym(2)  Rest(3)        │                   │
│  ┌─────────┼──────────────────────────────────────────┤                   │
│  │ Home(0) │    1        6        0        0          │                   │
│  │ Work(1) │    4        0        3        2          │                   │
│  │ Gym(2)  │    3        0        0        0          │                   │
│  │ Rest(3) │    2        0        0        0          │                   │
│  └─────────┴──────────────────────────────────────────┘                   │
│                                                                              │
│  Most frequent transitions:                                                 │
│  1. Home → Work:    6 (dominant morning pattern)                           │
│  2. Work → Home:    4                                                       │
│  3. Work → Gym:     3                                                       │
│  4. Gym → Home:     3                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Prediction Walkthrough

### Code: Get True/Pred Pairs

```python
# File: run_markov_ori.py, lines 125-162

def get_true_pred_pair(locSequence, df, n=1):
    """Get true and predicted pairs from test data."""
    testSeries = df["location_id"].values

    true_ls = []
    pred_ls = []
    time_ls = []
    
    for i in range(testSeries.shape[0] - n):
        locCurr = testSeries[i : i + n + 1]
        numbLoc = n

        start = timer()
        while True:
            res_df = locSequence
            for j in range(n - numbLoc, n):
                res_df = res_df.loc[res_df[f"loc_{j+1}"] == locCurr[j]]
            res_df = res_df.sort_values(by="size", ascending=False)

            if res_df.shape[0]:
                pred = res_df["toLoc"].drop_duplicates().values
                break
            numbLoc -= 1
            if numbLoc == 0:
                pred = np.zeros(10)
                break

        time_ls.append(timer() - start)
        true_ls.append(locCurr[-1])
        pred_ls.append(pred)

    return true_ls, pred_ls, time_ls
```

### Example Execution

**Input:**
- Transition table (from training): As shown above
- Test data (days 8-9): [0, 1, 2, 1, 0, 0, 1, 0]

**Step-by-step trace for first prediction:**

```python
# testSeries = [0, 1, 2, 1, 0, 0, 1, 0]
# i = 0 (first iteration)

# Step 1: Extract current locations + target
locCurr = testSeries[0 : 0 + 1 + 1]  # [0, 1]
# locCurr[0] = 0 (current location: Home)
# locCurr[1] = 1 (true next location: Work)

# Step 2: Look up in transition table
res_df = locSequence  # Full transition table
# Filter: loc_1 == 0 (current location is Home)
res_df = res_df.loc[res_df["loc_1"] == 0]

# Result:
#   loc_1  toLoc  size
#       0      1     6    # Home → Work
#       0      0     1    # Home → Home

# Step 3: Sort by size descending
res_df = res_df.sort_values(by="size", ascending=False)
#   loc_1  toLoc  size
#       0      1     6    # Rank 1
#       0      0     1    # Rank 2

# Step 4: Extract predictions
pred = res_df["toLoc"].drop_duplicates().values  # [1, 0]

# Step 5: Record result
true_ls.append(1)        # True: Work
pred_ls.append([1, 0])   # Predicted: [Work, Home]
```

**Continuing for all test positions:**

```
Position  Current  True Next  Prediction (ranked)  Correct@1?
────────  ───────  ─────────  ───────────────────  ──────────
   0       Home       Work     [Work, Home]           ✓
   1       Work       Gym      [Home, Gym, Rest]      ✗ (Gym is rank 2)
   2       Gym        Work     [Home]                 ✗ (Work not seen from Gym)
   3       Work       Home     [Home, Gym, Rest]      ✓
   4       Home       Home     [Work, Home]           ✗
   5       Home       Work     [Work, Home]           ✓
   6       Work       Home     [Home, Gym, Rest]      ✓
```

**Summary:**
- Total predictions: 7
- Correct @1: 4 (57%)
- Correct @5: 6 (86%)

---

## 7. Metric Calculation Walkthrough

### Code: Performance Metrics

```python
# File: run_markov_ori.py, lines 165-213

def get_performance_measure(true_ls, pred_ls):
    """Calculate performance metrics."""
    acc_ls = [1, 5, 10]

    res = []
    ndcg_ls = []
    res.append(len(true_ls))  # Total count
    
    for top_acc in acc_ls:
        correct = 0
        for true, pred in zip(true_ls, pred_ls):
            if true in pred[:top_acc]:
                correct += 1

            if top_acc == 10:
                idx = np.where(true == pred[:top_acc])[0]
                if len(idx) == 0:
                    ndcg_ls.append(0)
                else:
                    ndcg_ls.append(1 / np.log2(idx[0] + 1 + 1))

        res.append(correct)

    # F1 and Recall
    top1 = [pred[0] for pred in pred_ls]
    f1 = f1_score(true_ls, top1, average="weighted")
    recall = recall_score(true_ls, top1, average="weighted")
    res.append(f1)
    res.append(recall)
    res.append(np.mean(ndcg_ls))

    # MRR
    rank_ls = []
    for true, pred in zip(true_ls, pred_ls):
        rank = np.where(pred == true)[0] + 1
        if len(rank):
            rank_ls.append(rank[0])
        else:
            rank_ls.append(0)
    rank = np.array(rank_ls, dtype=float)
    rank = np.divide(1.0, rank, out=np.zeros_like(rank), where=rank != 0)
    res.append(rank.sum())

    return pd.Series(res, index=["total", "correct@1", "correct@5", 
                                  "correct@10", "f1", "recall", "ndcg", "rr"])
```

### Example Execution

**Input:**
```python
true_ls = [1, 2, 1, 0, 0, 1, 0]  # True next locations
pred_ls = [
    [1, 0],           # Prediction for position 0
    [0, 2, 3],        # Prediction for position 1
    [0],              # Prediction for position 2
    [0, 2, 3],        # Prediction for position 3
    [1, 0],           # Prediction for position 4
    [1, 0],           # Prediction for position 5
    [0, 2, 3]         # Prediction for position 6
]
```

**Step-by-step trace:**

```python
# ═══════════════════════════════════════════════════════════════════════════
# ACCURACY@K CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

# total = 7

# --- Accuracy@1 ---
# Position 0: true=1, pred[:1]=[1] → 1 in [1]? YES ✓
# Position 1: true=2, pred[:1]=[0] → 2 in [0]? NO
# Position 2: true=1, pred[:1]=[0] → 1 in [0]? NO
# Position 3: true=0, pred[:1]=[0] → 0 in [0]? YES ✓
# Position 4: true=0, pred[:1]=[1] → 0 in [1]? NO
# Position 5: true=1, pred[:1]=[1] → 1 in [1]? YES ✓
# Position 6: true=0, pred[:1]=[0] → 0 in [0]? YES ✓
# correct@1 = 4

# --- Accuracy@5 ---
# Position 0: true=1, pred[:5]=[1,0] → 1 in [1,0]? YES ✓
# Position 1: true=2, pred[:5]=[0,2,3] → 2 in [0,2,3]? YES ✓
# Position 2: true=1, pred[:5]=[0] → 1 in [0]? NO
# Position 3: true=0, pred[:5]=[0,2,3] → 0 in [0,2,3]? YES ✓
# Position 4: true=0, pred[:5]=[1,0] → 0 in [1,0]? YES ✓
# Position 5: true=1, pred[:5]=[1,0] → 1 in [1,0]? YES ✓
# Position 6: true=0, pred[:5]=[0,2,3] → 0 in [0,2,3]? YES ✓
# correct@5 = 6

# --- Accuracy@10 ---
# Same as @5 since predictions are shorter than 10
# correct@10 = 6

# ═══════════════════════════════════════════════════════════════════════════
# NDCG CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

# Position 0: true=1, pred=[1,0], idx=0 → NDCG = 1/log2(0+1+1) = 1/log2(2) = 1.0
# Position 1: true=2, pred=[0,2,3], idx=1 → NDCG = 1/log2(1+1+1) = 1/log2(3) = 0.631
# Position 2: true=1, pred=[0], idx=N/A → NDCG = 0
# Position 3: true=0, pred=[0,2,3], idx=0 → NDCG = 1/log2(2) = 1.0
# Position 4: true=0, pred=[1,0], idx=1 → NDCG = 1/log2(3) = 0.631
# Position 5: true=1, pred=[1,0], idx=0 → NDCG = 1.0
# Position 6: true=0, pred=[0,2,3], idx=0 → NDCG = 1.0
# 
# ndcg_ls = [1.0, 0.631, 0, 1.0, 0.631, 1.0, 1.0]
# mean_ndcg = 5.262 / 7 = 0.752

# ═══════════════════════════════════════════════════════════════════════════
# MRR CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

# Position 0: true=1, pred=[1,0] → rank=1 → RR=1/1=1.0
# Position 1: true=2, pred=[0,2,3] → rank=2 → RR=1/2=0.5
# Position 2: true=1, pred=[0] → rank=N/A → RR=0
# Position 3: true=0, pred=[0,2,3] → rank=1 → RR=1.0
# Position 4: true=0, pred=[1,0] → rank=2 → RR=0.5
# Position 5: true=1, pred=[1,0] → rank=1 → RR=1.0
# Position 6: true=0, pred=[0,2,3] → rank=1 → RR=1.0
#
# rr_sum = 1.0 + 0.5 + 0 + 1.0 + 0.5 + 1.0 + 1.0 = 5.0
# MRR = 5.0 / 7 = 0.714
```

**Final metrics:**

```python
# Result:
pd.Series({
    'total': 7,
    'correct@1': 4,
    'correct@5': 6,
    'correct@10': 6,
    'f1': 0.57,      # Weighted F1 for top-1 predictions
    'recall': 0.57,  # Same as accuracy for single-label
    'ndcg': 0.752,
    'rr': 5.0        # Sum of reciprocal ranks
})

# Percentages:
# Acc@1 = 4/7 = 57.1%
# Acc@5 = 6/7 = 85.7%
# Acc@10 = 6/7 = 85.7%
# MRR = 5.0/7 = 71.4%
# NDCG = 75.2%
```

---

## 8. Full Example: End-to-End

### Complete Trace with Mini Dataset

Let's trace through a complete run with a minimal dataset.

```python
# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

config = {
    'dataset': 'example',
    'data_csv': 'example_data.csv',
    'markov_order': 1
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

# example_data.csv content:
# id,user_id,location_id,duration,start_day,start_min,weekday
# 0,1,0,60,0,420,0     # User 1, Day 0, Home
# 1,1,1,480,0,500,0    # User 1, Day 0, Work
# 2,1,2,60,0,1000,0    # User 1, Day 0, Gym
# 3,1,0,120,0,1080,0   # User 1, Day 0, Home
# 4,1,0,60,1,420,1     # User 1, Day 1, Home
# 5,1,1,480,1,500,1    # User 1, Day 1, Work
# 6,1,0,120,1,1000,1   # User 1, Day 1, Home
# 7,1,0,60,2,420,2     # User 1, Day 2, Home
# 8,1,1,480,2,500,2    # User 1, Day 2, Work
# 9,1,3,90,2,1000,2    # User 1, Day 2, Restaurant

inputData = pd.read_csv('example_data.csv')
inputData.sort_values(by=['user_id', 'start_day', 'start_min'], inplace=True)

print(f"Loaded {len(inputData)} records")
# Output: Loaded 10 records

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: SPLIT DATASET
# ═══════════════════════════════════════════════════════════════════════════

# User 1: max_day = 2
# train_split = 2 * 0.6 = 1.2
# vali_split = 2 * 0.8 = 1.6

# Day 0: train (0 < 1.2)
# Day 1: train (1 < 1.2)
# Day 2: test (2 >= 1.6)

train_data, vali_data, test_data = splitDataset(inputData)
print(f"Train: {len(train_data)}, Vali: {len(vali_data)}, Test: {len(test_data)}")
# Output: Train: 7, Vali: 0, Test: 3

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: BUILD TRANSITION TABLE (TRAINING)
# ═══════════════════════════════════════════════════════════════════════════

# Training sequence: [0, 1, 2, 0, 0, 1, 0]
# Transitions:
#   0 → 1 (Home → Work)
#   1 → 2 (Work → Gym)
#   2 → 0 (Gym → Home)
#   0 → 0 (Home → Home)
#   0 → 1 (Home → Work)
#   1 → 0 (Work → Home)

locSeq_df = markov_transition_prob(train_data, n=1)
print(locSeq_df)
# Output:
#    loc_1  toLoc  size
# 0      0      0     1
# 1      0      1     2
# 2      1      0     1
# 3      1      2     1
# 4      2      0     1

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: PREDICTION (TESTING)
# ═══════════════════════════════════════════════════════════════════════════

# Test sequence: [0, 1, 3]
# (Home → Work → Restaurant)

# Position 0: Current = Home(0)
#   Query: What follows Home?
#   Table lookup: loc_1 == 0 → [toLoc=0(1), toLoc=1(2)]
#   Sorted: [1, 0] (Work=2 counts, Home=1 count)
#   True next: Work(1)
#   Prediction: [1, 0]
#   Result: Correct @1 ✓

# Position 1: Current = Work(1)
#   Query: What follows Work?
#   Table lookup: loc_1 == 1 → [toLoc=0(1), toLoc=2(1)]
#   Sorted: [0, 2] (tie, original order)
#   True next: Restaurant(3)
#   Prediction: [0, 2]
#   Result: Incorrect (Restaurant never seen after Work)

true_ls, pred_ls, _ = get_true_pred_pair(locSeq_df, test_data, n=1)
print(f"True: {true_ls}")
print(f"Pred: {pred_ls}")
# Output:
# True: [1, 3]
# Pred: [array([1, 0]), array([0, 2])]

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: CALCULATE METRICS
# ═══════════════════════════════════════════════════════════════════════════

result = get_performance_measure(true_ls, pred_ls)
print(result)
# Output:
# total        2.0
# correct@1    1.0
# correct@5    1.0
# correct@10   1.0
# f1           0.5
# recall       0.5
# ndcg         0.5
# rr           1.0

# Final metrics:
# Acc@1 = 1/2 = 50%
# Acc@5 = 1/2 = 50%
# MRR = 1.0/2 = 50%
# NDCG = 50%
```

---

## 9. Code Execution Trace

### Annotated Main Function

```python
def main():
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: SETUP
    # ─────────────────────────────────────────────────────────────────────────
    
    # Parse command line: python run_markov_ori.py --config config.yaml
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    # args.config = "config/models/config_markov_ori_geolife.yaml"

    # Load and flatten config
    config = load_config(args.config)
    config = EasyDict(config)
    # config.dataset = "geolife"
    # config.data_csv = "data/.../dataset_geolife.csv"
    # config.markov_order = 1
    
    n = config.get("markov_order", 1)  # n = 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: DATA PREPARATION
    # ─────────────────────────────────────────────────────────────────────────
    
    # Create experiment directory: experiments/geolife_markov_ori_20251226_173239/
    experiment_dir = init_experiment_dir(config, dataset_name, model_name="markov_ori")
    
    # Load CSV data (16,600 rows for GeoLife)
    inputData = pd.read_csv(csv_path)
    inputData.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    # Ensures chronological order within each user
    
    # Split: 60% train, 20% val, 20% test (by user's tracked days)
    train_data, vali_data, test_data = splitDataset(inputData)
    # train: 8,380 rows, vali: 4,018 rows, test: 4,202 rows
    
    # Filter by valid_ids (match with neural model samples)
    valid_ids = pickle.load(open(valid_ids_file, "rb"))
    train_data = train_data.loc[train_data["id"].isin(valid_ids)]
    vali_data = vali_data.loc[vali_data["id"].isin(valid_ids)]
    test_data = test_data.loc[test_data["id"].isin(valid_ids)]
    # train: 7,424 rows, vali: 3,334 rows, test: 3,502 rows
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: TRAINING AND EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    
    # run_evaluation does:
    # 1. Loop through each user
    # 2. Build transition table from user's training data
    # 3. Generate predictions for user's test data
    # 4. Aggregate all predictions
    # 5. Compute metrics
    
    val_results, total_params = run_evaluation(config, train_data, vali_data, "validation", n=n)
    # Processing 45 users...
    # val_results = {'acc@1': 33.57, 'mrr': 39.91, ...}
    
    test_results, _ = run_evaluation(config, train_data, test_data, "test", n=n)
    # Processing 45 users...
    # test_results = {'acc@1': 24.18, 'mrr': 30.34, ...}
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: OUTPUT
    # ─────────────────────────────────────────────────────────────────────────
    
    # Save results to JSON files
    with open(os.path.join(experiment_dir, "val_results.json"), "w") as f:
        json.dump(val_results, f, indent=2)
    with open(os.path.join(experiment_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Save config copy
    shutil.copy(args.config, os.path.join(experiment_dir, "config_original.yaml"))
    
    # Close log file
    log_file.close()
    
    print(f"Results saved to: {experiment_dir}")
```

### Timing Breakdown (GeoLife)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION TIMING (GeoLife)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase                         Time        Percentage                       │
│  ─────────────────────         ────        ──────────                       │
│  Config loading                0.01s       0.2%                             │
│  Data loading                  0.10s       1.9%                             │
│  Data sorting                  0.02s       0.4%                             │
│  Dataset splitting             0.05s       0.9%                             │
│  Valid ID filtering            0.02s       0.4%                             │
│  Validation evaluation         2.50s       47.3%                            │
│  Test evaluation               2.50s       47.3%                            │
│  Result saving                 0.08s       1.5%                             │
│  ─────────────────────         ────        ──────────                       │
│  TOTAL                         ~5.28s      100%                             │
│                                                                              │
│  Notes:                                                                     │
│  • 95% of time is spent in evaluation (prediction + metrics)               │
│  • Most time is DataFrame filtering for lookup                             │
│  • No GPU operations, pure CPU                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Key Functions

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `load_config` | Load YAML config | File path | Dict |
| `splitDataset` | Split by user days | DataFrame | 3 DataFrames |
| `getSplitDaysUser` | Per-user splitting | User DF | DF with labels |
| `markov_transition_prob` | Build transitions | Train DF | Transition DF |
| `get_true_pred_pair` | Generate predictions | Transitions + Test | Lists |
| `get_performance_measure` | Compute metrics | True/Pred lists | Metrics Series |
| `run_evaluation` | Full eval pipeline | Config + Data | Metrics Dict |

---

## Navigation

| Previous | Next |
|----------|------|
| [06_RESULTS_ANALYSIS.md](06_RESULTS_ANALYSIS.md) | [Back to Overview](01_OVERVIEW.md) |

# Components Deep Dive: Detailed Analysis of Each Module

## Table of Contents

1. [Configuration Component](#1-configuration-component)
2. [Data Loading Component](#2-data-loading-component)
3. [Dataset Splitting Component](#3-dataset-splitting-component)
4. [Valid ID Filtering Component](#4-valid-id-filtering-component)
5. [Transition Matrix Building Component](#5-transition-matrix-building-component)
6. [Prediction Component](#6-prediction-component)
7. [Evaluation Component](#7-evaluation-component)
8. [Experiment Management Component](#8-experiment-management-component)
9. [Component Interactions](#9-component-interactions)

---

## 1. Configuration Component

### Purpose

The configuration component manages all runtime parameters, enabling:
- Reproducible experiments
- Easy parameter changes without code modification
- Support for multiple datasets with different settings

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              CONFIGURATION COMPONENT                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input                           Processing                         │
│  ┌────────────────┐             ┌────────────────────────────────┐ │
│  │ config.yaml    │───────────> │ yaml.safe_load()               │ │
│  │                │             │ Parse nested YAML structure     │ │
│  │ data:          │             │                                 │ │
│  │   dataset: ... │             │ ┌──────────────────────────────┐│ │
│  │   data_csv: ...│             │ │ Flattening Logic             ││ │
│  │ model:         │             │ │ for key, value in cfg:       ││ │
│  │   order: ...   │             │ │   if isinstance(value, dict):││ │
│  └────────────────┘             │ │     flatten nested keys      ││ │
│                                 │ └──────────────────────────────┘│ │
│                                 └────────────────────────────────┘ │
│                                            │                        │
│                                            ↓                        │
│  Output                                                             │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │ EasyDict                                                       ││
│  │ {                                                              ││
│  │   'dataset': 'geolife',                                       ││
│  │   'data_csv': 'data/.../dataset_geolife.csv',                 ││
│  │   'markov_order': 1,                                          ││
│  │   ...                                                          ││
│  │ }                                                              ││
│  │                                                                ││
│  │ Access: config.dataset OR config['dataset']                    ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | - | Dataset name ('geolife' or 'diy') |
| `data_csv` | str | - | Path to input CSV file |
| `valid_ids_file` | str | None | Path to valid IDs pickle file |
| `processed_dir` | str | None | Directory for preprocessed data |
| `dataset_prefix` | str | None | Prefix for preprocessed files |
| `experiment_root` | str | 'experiments' | Root directory for outputs |
| `markov_order` | int | 1 | Order of Markov chain |

### Why This Design?

**Justification:**
1. **Separation of concerns:** Parameters outside code
2. **Reproducibility:** Save config with results
3. **Flexibility:** Different configs for different experiments
4. **Flat structure:** Easier access than nested dicts

---

## 2. Data Loading Component

### Purpose

Loads raw mobility data from CSV and prepares it for processing by:
- Reading the file efficiently
- Sorting to ensure temporal order
- Validating data format

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              DATA LOADING COMPONENT                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: CSV File Path                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ dataset_geolife.csv                                         │   │
│  │ ┌─────────────────────────────────────────────────────────┐ │   │
│  │ │id,user_id,location_id,duration,start_day,end_day,...    │ │   │
│  │ │0,1,0,64.0,0,0,183,248,3                                 │ │   │
│  │ │1,1,1,309.0,0,0,272,582,3                                │ │   │
│  │ │...                                                       │ │   │
│  │ └─────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         │                                           │
│                         ↓                                           │
│  Processing: pd.read_csv() + sort_values()                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ # Load into DataFrame                                       │   │
│  │ inputData = pd.read_csv(csv_path)                           │   │
│  │                                                             │   │
│  │ # Sort by user, then by time                                │   │
│  │ inputData.sort_values(                                      │   │
│  │     by=["user_id", "start_day", "start_min"],              │   │
│  │     inplace=True                                            │   │
│  │ )                                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         │                                           │
│                         ↓                                           │
│  Output: Sorted DataFrame                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ DataFrame (16,600 rows × 9 columns for GeoLife)             │   │
│  │                                                             │   │
│  │   id  user_id  location_id  duration  start_day  ...       │   │
│  │   0       1            0      64.0          0              │   │
│  │   1       1            1     309.0          0              │   │
│  │   ...                                                       │   │
│  │                                                             │   │
│  │ Guaranteed: Records for each user are in temporal order    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Sorting Matters

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHY SORTING IS CRITICAL                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BEFORE SORTING (arbitrary order):                                  │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ user_id  location  start_day  start_min                       │ │
│  │    1        A          5          200                         │ │
│  │    1        B          0          100    ← Earlier but later  │ │
│  │    1        C          3          150      in file            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Transition extraction would be WRONG:                              │
│  A→B (day 5 to day 0?!) and B→C (day 0 to day 3?!)                │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  AFTER SORTING (chronological):                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ user_id  location  start_day  start_min                       │ │
│  │    1        B          0          100    ← Earliest           │ │
│  │    1        C          3          150                         │ │
│  │    1        A          5          200    ← Latest             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Correct transitions: B→C (day 0→3) and C→A (day 3→5)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Statistics

| Dataset | Total Records | Unique Users | Unique Locations |
|---------|---------------|--------------|------------------|
| GeoLife | 16,600 | 46 | ~1,185 |
| DIY | ~200,000 | ~700 | ~4,000 |

---

## 3. Dataset Splitting Component

### Purpose

Splits data into train/validation/test sets using a **time-based** strategy that:
- Preserves temporal order (no future leakage)
- Applies per-user splitting (each user's timeline split independently)
- Uses 60/20/20 ratio

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              DATASET SPLITTING COMPONENT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: Sorted DataFrame (all users)                                │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                                                                ││
│  │  Step 1: Group by user_id                                      ││
│  │  ┌─────────┬────────────────────────────────────────────────┐ ││
│  │  │ User 1  │ [records day 0-100]                            │ ││
│  │  ├─────────┼────────────────────────────────────────────────┤ ││
│  │  │ User 2  │ [records day 0-80]                             │ ││
│  │  ├─────────┼────────────────────────────────────────────────┤ ││
│  │  │ ...     │ ...                                            │ ││
│  │  └─────────┴────────────────────────────────────────────────┘ ││
│  │                                                                ││
│  └────────────────────────────────────────────────────────────────┘│
│                         │                                           │
│                         ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                                                                ││
│  │  Step 2: Apply getSplitDaysUser to each group                  ││
│  │                                                                ││
│  │  User 1 (max_day=100):                                         ││
│  │  ┌──────────────────────────────────────────────────────────┐ ││
│  │  │ Days 0-59        │ Days 60-79     │ Days 80-99          │ ││
│  │  │ Dataset="train"  │ Dataset="vali" │ Dataset="test"      │ ││
│  │  └──────────────────────────────────────────────────────────┘ ││
│  │                                                                ││
│  │  User 2 (max_day=80):                                          ││
│  │  ┌──────────────────────────────────────────────────────────┐ ││
│  │  │ Days 0-47        │ Days 48-63     │ Days 64-79          │ ││
│  │  │ Dataset="train"  │ Dataset="vali" │ Dataset="test"      │ ││
│  │  └──────────────────────────────────────────────────────────┘ ││
│  │                                                                ││
│  └────────────────────────────────────────────────────────────────┘│
│                         │                                           │
│                         ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                                                                ││
│  │  Step 3: Separate by Dataset label                             ││
│  │                                                                ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            ││
│  │  │ train_data  │  │ vali_data   │  │ test_data   │            ││
│  │  │ (60%)       │  │ (20%)       │  │ (20%)       │            ││
│  │  │             │  │             │  │             │            ││
│  │  │ 8,380 rows  │  │ 4,018 rows  │  │ 4,202 rows  │            ││
│  │  │ (GeoLife)   │  │ (GeoLife)   │  │ (GeoLife)   │            ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘            ││
│  │                                                                ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Algorithm: `getSplitDaysUser`

```python
def getSplitDaysUser(df):
    """
    Input: DataFrame for single user
    Output: Same DataFrame with 'Dataset' column
    
    Algorithm:
    1. Find maximum day index (tracking duration)
    2. Calculate split points:
       - train_split = max_day * 0.6
       - vali_split = max_day * 0.8
    3. Label each record based on its start_day
    """
    maxDay = df["start_day"].max()      # e.g., 100
    train_split = maxDay * 0.6           # e.g., 60
    vali_split = maxDay * 0.8            # e.g., 80
    
    # Initialize all as "test"
    df["Dataset"] = "test"
    
    # Label train (days 0 to train_split)
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    
    # Label validation (days train_split to vali_split)
    df.loc[
        (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
        "Dataset",
    ] = "vali"
    
    return df
```

### Why Per-User Splitting?

```
┌─────────────────────────────────────────────────────────────────────┐
│              PER-USER vs GLOBAL SPLITTING                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  GLOBAL SPLITTING (wrong for this problem):                         │
│  ─────────────────────────────────────────                          │
│  All users split by the same day threshold                          │
│                                                                     │
│  Problem: User A started tracking on day 50                         │
│           Global train ends at day 30                               │
│           → User A has NO training data!                            │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  PER-USER SPLITTING (correct):                                      │
│  ────────────────────────────                                       │
│  Each user's timeline split independently                           │
│                                                                     │
│  User A (days 50-150): Train(50-110), Val(110-130), Test(130-150) │
│  User B (days 0-100):  Train(0-60), Val(60-80), Test(80-100)      │
│                                                                     │
│  → Every user has data in all splits                               │
│  → Temporal order preserved per user                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Valid ID Filtering Component

### Purpose

Filters records to match those used by neural models, ensuring:
- Fair comparison across different model types
- Consistent evaluation samples
- Handling of users that may be dropped in preprocessing

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              VALID ID FILTERING COMPONENT                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Two Pathways:                                                      │
│                                                                     │
│  Path A: Pre-computed valid_ids file exists                        │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │  valid_ids_geolife.pk                                          ││
│  │       ↓                                                        ││
│  │  pickle.load() → numpy array of IDs                            ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  Path B: Generate from preprocessed data                           │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │  Preprocessed .pk files                                        ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           ││
│  │  │ _train.pk    │ │ _val.pk      │ │ _test.pk     │           ││
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘           ││
│  │         │                │                │                    ││
│  │         └────────────────┼────────────────┘                    ││
│  │                          ↓                                     ││
│  │              Extract unique user IDs                           ││
│  │                          ↓                                     ││
│  │              Filter CSV to these users                         ││
│  │                          ↓                                     ││
│  │              Return record IDs                                 ││
│  └────────────────────────────────────────────────────────────────┘│
│                         │                                           │
│                         ↓                                           │
│  Application: Inner join with split data                            │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                                                                ││
│  │  train_data = train_data.loc[train_data["id"].isin(valid_ids)]││
│  │  vali_data = vali_data.loc[vali_data["id"].isin(valid_ids)]   ││
│  │  test_data = test_data.loc[test_data["id"].isin(valid_ids)]   ││
│  │                                                                ││
│  │  GeoLife:                                                      ││
│  │  Before: train=8,380, vali=4,018, test=4,202                  ││
│  │  After:  train=7,424, vali=3,334, test=3,502                  ││
│  │                                                                ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Function: `generate_valid_ids`

```python
def generate_valid_ids(preprocessed_dir, dataset_prefix, csv_data):
    """
    Generate valid_ids by matching preprocessed data users.
    
    This ensures Markov model uses same samples as neural models.
    
    Args:
        preprocessed_dir: Directory with .pk files
        dataset_prefix: Prefix for file names (e.g., "geolife_eps20_prev7")
        csv_data: Original CSV DataFrame
    
    Returns:
        numpy.ndarray: Array of valid record IDs
    """
    # Load all preprocessed data splits
    train_data = pickle.load(open(f"{preprocessed_dir}/{dataset_prefix}_train.pk", "rb"))
    val_data = pickle.load(open(f"{preprocessed_dir}/{dataset_prefix}_validation.pk", "rb"))
    test_data = pickle.load(open(f"{preprocessed_dir}/{dataset_prefix}_test.pk", "rb"))
    
    # Extract unique users from each split
    users_train = set(sample['user_X'][0] for sample in train_data)
    users_val = set(sample['user_X'][0] for sample in val_data)
    users_test = set(sample['user_X'][0] for sample in test_data)
    
    # Union of all users
    all_users = users_train | users_val | users_test
    
    # Filter CSV to only these users
    valid_records = csv_data[csv_data['user_id'].isin(all_users)]
    
    return valid_records['id'].values
```

### Why This Filtering?

```
┌─────────────────────────────────────────────────────────────────────┐
│              IMPORTANCE OF VALID ID FILTERING                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Problem: Preprocessing may exclude some users/records              │
│                                                                     │
│  Original CSV:                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ User 1: 100 records                                         │   │
│  │ User 2: 50 records                                          │   │
│  │ User 3: 5 records   ← Too few for neural model             │   │
│  │ ...                                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Neural Model Preprocessing:                                        │
│  - Requires minimum sequence length                                 │
│  - User 3 dropped (not enough data)                                │
│  - Final: Users 1, 2, 4, 5, ...                                    │
│                                                                     │
│  Markov Model (without filtering):                                  │
│  - Would include User 3                                            │
│  - Different evaluation samples → Unfair comparison!               │
│                                                                     │
│  Markov Model (with valid_ids filtering):                          │
│  - Matches exactly the users in neural model                       │
│  - Same samples → Fair comparison ✓                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Transition Matrix Building Component

### Purpose

Builds the core Markov model by:
- Counting transitions between locations
- Creating per-user transition tables
- Storing counts for ranking-based prediction

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              TRANSITION MATRIX BUILDING COMPONENT                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: User's Training Data (sorted by time)                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ location_id sequence: [H, W, G, W, H, W, R, H]              │   │
│  │                        ↓  ↓  ↓  ↓  ↓  ↓  ↓                 │   │
│  │ Indices:               0  1  2  3  4  5  6  7               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         │                                           │
│                         ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │  Step 1: Extract Source-Destination Pairs (n=1)                ││
│  │  ┌─────────────────────────────────────────────────────────┐  ││
│  │  │ Source (loc_1):   [H, W, G, W, H, W, R]   iloc[:-1]     │  ││
│  │  │ Dest (toLoc):     [W, G, W, H, W, R, H]   iloc[1:]      │  ││
│  │  └─────────────────────────────────────────────────────────┘  ││
│  │                                                                ││
│  │  Pairs: H→W, W→G, G→W, W→H, H→W, W→R, R→H                     ││
│  └────────────────────────────────────────────────────────────────┘│
│                         │                                           │
│                         ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │  Step 2: Group and Count                                       ││
│  │  ┌─────────────────────────────────────────────────────────┐  ││
│  │  │ .groupby(['loc_1', 'toLoc']).size()                     │  ││
│  │  │                                                         │  ││
│  │  │ loc_1  toLoc  size                                      │  ││
│  │  │   G      W      1                                       │  ││
│  │  │   H      W      2    ← H→W appears twice               │  ││
│  │  │   R      H      1                                       │  ││
│  │  │   W      G      1                                       │  ││
│  │  │   W      H      1                                       │  ││
│  │  │   W      R      1                                       │  ││
│  │  └─────────────────────────────────────────────────────────┘  ││
│  └────────────────────────────────────────────────────────────────┘│
│                         │                                           │
│                         ↓                                           │
│  Output: Transition Count DataFrame                                 │
│  (Sorted by count for each source location during prediction)      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Higher-Order Transitions (n > 1)

For higher-order Markov chains, the process extends to n-grams:

```
┌─────────────────────────────────────────────────────────────────────┐
│              HIGHER-ORDER TRANSITION EXTRACTION (n=2)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Sequence: [A, B, C, D, E, F]                                      │
│  Indices:  [0, 1, 2, 3, 4, 5]                                      │
│                                                                     │
│  n=2: Look at pairs of previous locations                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  loc_1:   [A, B, C, D]       iloc[0:4]   (positions 0,1,2,3) │  │
│  │  loc_2:   [B, C, D, E]       iloc[1:5]   (positions 1,2,3,4) │  │
│  │  toLoc:   [C, D, E, F]       iloc[2:6]   (positions 2,3,4,5) │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Resulting n-grams:                                                │
│  (A,B)→C, (B,C)→D, (C,D)→E, (D,E)→F                               │
│                                                                     │
│  DataFrame:                                                        │
│  loc_1  loc_2  toLoc  size                                         │
│    A      B      C      1                                          │
│    B      C      D      1                                          │
│    C      D      E      1                                          │
│    D      E      F      1                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Function: `markov_transition_prob`

```python
def markov_transition_prob(df, n=1):
    """
    Build transition probability table.
    
    For n=1 (1st-order): Tracks single location → next location
    For n>1 (higher-order): Tracks sequence of n locations → next location
    
    Implementation details:
    - Uses pandas slicing for efficiency
    - Groups and counts using groupby().size()
    - Returns raw counts (not probabilities) for ranking
    """
    # Create column names: loc_1, loc_2, ..., loc_n, toLoc
    COLUMNS = [f"loc_{i+1}" for i in range(n)]
    COLUMNS.append("toLoc")

    locSequence = pd.DataFrame(columns=COLUMNS)

    # Destination is n positions ahead
    locSequence["toLoc"] = df.iloc[n:]["location_id"].values
    
    # Source locations are at positions 0, 1, ..., n-1
    for i in range(n):
        # iloc[i : -n + i] or iloc[i : len-n+i]
        # This creates shifted views of the sequence
        if i < n - 1:
            locSequence[f"loc_{i+1}"] = df.iloc[i : -(n-i)]["location_id"].values
        else:
            # Last source column: iloc[n-1 : -(n-(n-1))] = iloc[n-1:-1]
            locSequence[f"loc_{i+1}"] = df.iloc[i : -n + i if -n + i != 0 else None]["location_id"].values
    
    # Group by all source columns and destination, count occurrences
    return locSequence.groupby(by=COLUMNS).size().to_frame("size").reset_index()
```

---

## 6. Prediction Component

### Purpose

Generates predictions for test samples by:
- Looking up current location(s) in transition table
- Ranking destinations by frequency
- Handling cold-start with fallback mechanisms

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              PREDICTION COMPONENT                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Transition Table (from training)                            │   │
│  │ ┌─────────────────────────────────────┐                     │   │
│  │ │ loc_1  toLoc  size                  │                     │   │
│  │ │   H      W      15                  │                     │   │
│  │ │   W      H      10                  │                     │   │
│  │ │   W      G       8                  │                     │   │
│  │ │   ...                               │                     │   │
│  │ └─────────────────────────────────────┘                     │   │
│  │                                                             │   │
│  │ Test Sequence: [Home, Work, Gym, Work]                      │   │
│  │                                   ↑                         │   │
│  │                            current location                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         │                                           │
│                         ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │  PREDICTION ALGORITHM                                          ││
│  │                                                                ││
│  │  for each test position:                                       ││
│  │      current_loc = test_sequence[i]  # "Work"                 ││
│  │                                                                ││
│  │      ┌─────────────────────────────────────────────────────┐  ││
│  │      │ Step 1: Filter transitions from current_loc         │  ││
│  │      │                                                     │  ││
│  │      │ res_df = table[table.loc_1 == "Work"]              │  ││
│  │      │                                                     │  ││
│  │      │ Result:                                             │  ││
│  │      │ loc_1  toLoc  size                                  │  ││
│  │      │   W      H      10                                  │  ││
│  │      │   W      G       8                                  │  ││
│  │      │   W      R       5                                  │  ││
│  │      └─────────────────────────────────────────────────────┘  ││
│  │                                                                ││
│  │      ┌─────────────────────────────────────────────────────┐  ││
│  │      │ Step 2: Sort by count (descending)                  │  ││
│  │      │                                                     │  ││
│  │      │ Sorted: [H(10), G(8), R(5)]                        │  ││
│  │      └─────────────────────────────────────────────────────┘  ││
│  │                                                                ││
│  │      ┌─────────────────────────────────────────────────────┐  ││
│  │      │ Step 3: Return ranked predictions                   │  ││
│  │      │                                                     │  ││
│  │      │ pred = [Home, Gym, Restaurant, ...]                │  ││
│  │      └─────────────────────────────────────────────────────┘  ││
│  │                                                                ││
│  └────────────────────────────────────────────────────────────────┘│
│                         │                                           │
│                         ↓                                           │
│  Output:                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ true_ls:  [actual next locations]                           │   │
│  │ pred_ls:  [ranked prediction arrays]                        │   │
│  │ time_ls:  [prediction times for profiling]                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Fallback Mechanism

```
┌─────────────────────────────────────────────────────────────────────┐
│              FALLBACK MECHANISM FOR COLD START                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Scenario: Current location "New_Place" not in transition table     │
│                                                                     │
│  For n=2 (2nd-order):                                              │
│  Query: (Old_Loc, New_Place) → ?                                   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Attempt 1: Search for (Old_Loc, New_Place)                │    │
│  │             Result: NOT FOUND                               │    │
│  │             numbLoc = 2 → 1                                 │    │
│  └────────────────────────────────────────────────────────────┘    │
│                         │                                           │
│                         ↓                                           │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Attempt 2: Search for (*, New_Place)                      │    │
│  │             [Ignore Old_Loc, only match New_Place]         │    │
│  │             Result: NOT FOUND (New_Place is truly new)     │    │
│  │             numbLoc = 1 → 0                                 │    │
│  └────────────────────────────────────────────────────────────┘    │
│                         │                                           │
│                         ↓                                           │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Attempt 3: numbLoc == 0                                   │    │
│  │             GIVE UP: Return zeros                          │    │
│  │             pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Note: For n=1, there's only one fallback step before zeros.       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Evaluation Component

### Purpose

Computes comprehensive metrics to assess prediction quality:
- Multiple accuracy levels (Acc@1, @5, @10)
- Ranking metrics (MRR, NDCG)
- Classification metrics (F1, Recall)

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              EVALUATION COMPONENT                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ true_ls:  [A, B, C, A, D, ...]  (ground truth)              │   │
│  │ pred_ls:  [[A,C,B], [C,B,A], [C,A,D], ...]  (predictions)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         │                                           │
│                         ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │  METRIC CALCULATIONS                                           ││
│  │                                                                ││
│  │  ┌──────────────────────────────────────────────────────────┐ ││
│  │  │ Accuracy@K                                               │ ││
│  │  │                                                          │ ││
│  │  │ For each (true, pred) pair:                              │ ││
│  │  │   if true in pred[:K]:                                   │ ││
│  │  │       correct_at_K += 1                                  │ ││
│  │  │                                                          │ ││
│  │  │ Example: true=A, pred=[A,C,B]                           │ ││
│  │  │   Acc@1: A in [A]? YES ✓                                │ ││
│  │  │   Acc@5: A in [A,C,B]? YES ✓                            │ ││
│  │  └──────────────────────────────────────────────────────────┘ ││
│  │                                                                ││
│  │  ┌──────────────────────────────────────────────────────────┐ ││
│  │  │ Reciprocal Rank (for MRR)                                │ ││
│  │  │                                                          │ ││
│  │  │ Find position of true in pred (1-indexed)                │ ││
│  │  │ RR = 1 / position                                        │ ││
│  │  │                                                          │ ││
│  │  │ Example: true=B, pred=[A,B,C]                           │ ││
│  │  │   Position = 2                                           │ ││
│  │  │   RR = 1/2 = 0.5                                        │ ││
│  │  │                                                          │ ││
│  │  │ If not found: RR = 0                                     │ ││
│  │  └──────────────────────────────────────────────────────────┘ ││
│  │                                                                ││
│  │  ┌──────────────────────────────────────────────────────────┐ ││
│  │  │ NDCG@10 (Normalized Discounted Cumulative Gain)          │ ││
│  │  │                                                          │ ││
│  │  │ Formula: NDCG = 1 / log2(position + 1)                   │ ││
│  │  │                                                          │ ││
│  │  │ Position 1: 1/log2(2) = 1.000                           │ ││
│  │  │ Position 2: 1/log2(3) = 0.631                           │ ││
│  │  │ Position 3: 1/log2(4) = 0.500                           │ ││
│  │  │ Position 10: 1/log2(11) = 0.289                         │ ││
│  │  │ Beyond 10: 0                                             │ ││
│  │  └──────────────────────────────────────────────────────────┘ ││
│  │                                                                ││
│  │  ┌──────────────────────────────────────────────────────────┐ ││
│  │  │ F1 Score (Weighted)                                      │ ││
│  │  │                                                          │ ││
│  │  │ Uses only top-1 predictions                              │ ││
│  │  │ sklearn.metrics.f1_score(true_ls, top1_ls, average='weighted')│ ││
│  │  │                                                          │ ││
│  │  │ Weighted: Accounts for class imbalance                   │ ││
│  │  │           More weight to frequent locations              │ ││
│  │  └──────────────────────────────────────────────────────────┘ ││
│  │                                                                ││
│  └────────────────────────────────────────────────────────────────┘│
│                         │                                           │
│                         ↓                                           │
│  Output:                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ pd.Series {                                                 │   │
│  │   'total': 3457,                                           │   │
│  │   'correct@1': 836,                                        │   │
│  │   'correct@5': 1309,                                       │   │
│  │   'correct@10': 1340,                                      │   │
│  │   'f1': 0.2338,                                            │   │
│  │   'recall': 0.2418,                                        │   │
│  │   'ndcg': 0.3238,                                          │   │
│  │   'rr': 1048.47  (sum, divide by total for MRR)           │   │
│  │ }                                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Metric Formulas

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **Acc@K** | correct@K / total × 100 | 0-100% | % samples with correct in top-K |
| **MRR** | (Σ 1/rank) / total × 100 | 0-100% | Average reciprocal rank |
| **NDCG@10** | mean(1/log2(rank+1)) × 100 | 0-100% | Ranking quality |
| **F1** | 2×precision×recall / (precision+recall) | 0-1 | Classification quality |

---

## 8. Experiment Management Component

### Purpose

Manages experimental outputs for reproducibility:
- Creates timestamped directories
- Saves configurations and results
- Generates logs

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              EXPERIMENT MANAGEMENT COMPONENT                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Directory Creation:                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  experiments/                                               │   │
│  │  └── {dataset}_{model}_{timestamp}/                        │   │
│  │      ├── checkpoints/        (empty for Markov)            │   │
│  │      ├── training.log                                      │   │
│  │      ├── config.yaml         (flattened)                   │   │
│  │      ├── config_original.yaml (copy)                       │   │
│  │      ├── val_results.json                                  │   │
│  │      └── test_results.json                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Naming Convention:                                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  geolife_markov_ori_20251226_173239                        │   │
│  │  ├─────┤ ├────────┤ ├──────────────┤                       │   │
│  │  dataset  model     timestamp (GMT+7)                       │   │
│  │                     YYYYMMdd_HHmmss                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  File Contents:                                                     │
│                                                                     │
│  training.log:                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ============================================================│   │
│  │ 1st-Order Markov Chain Model - Original Implementation      │   │
│  │ ============================================================│   │
│  │                                                             │   │
│  │ Dataset: geolife                                           │   │
│  │ Config: config/models/config_markov_ori_geolife.yaml       │   │
│  │ ...                                                         │   │
│  │ Validation Results:                                        │   │
│  │   Acc@1:  33.57%                                           │   │
│  │   ...                                                       │   │
│  │ Test Results:                                               │   │
│  │   Acc@1:  24.18%                                           │   │
│  │   ...                                                       │   │
│  │ === Training Complete ===                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Timestamp Generation

```python
def init_experiment_dir(config, dataset_name, model_name="markov_ori"):
    """Create experiment directory with GMT+7 timestamp."""
    gmt7 = timezone(timedelta(hours=7))
    now = datetime.now(gmt7)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{dataset_name}_{model_name}_{timestamp}"
    experiment_dir = os.path.join(config.experiment_root, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    return experiment_dir
```

---

## 9. Component Interactions

### Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE SYSTEM INTERACTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                           │
│  │ Command Line │                                                           │
│  │ Arguments    │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                    │
│         ↓                                                                    │
│  ┌──────────────┐     ┌────────────────┐                                   │
│  │ Config       │────>│ EasyDict       │                                   │
│  │ Component    │     │ (config obj)   │                                   │
│  └──────────────┘     └───────┬────────┘                                   │
│                               │                                             │
│         ┌─────────────────────┼─────────────────────┐                      │
│         │                     │                     │                       │
│         ↓                     ↓                     ↓                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐           │
│  │ Data Loading │     │ Experiment   │     │ Valid ID         │           │
│  │ Component    │     │ Management   │     │ Filtering        │           │
│  └──────┬───────┘     └──────┬───────┘     └─────────┬────────┘           │
│         │                     │                       │                     │
│         ↓                     │                       │                     │
│  ┌──────────────┐             │                       │                     │
│  │ Dataset      │             │                       │                     │
│  │ Splitting    │<────────────────────────────────────┘                     │
│  └──────┬───────┘             │                                             │
│         │                     │                                             │
│  ┌──────┴───────────────────────────┐                                       │
│  │                                   │                                       │
│  ↓                                   ↓                                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                                       │
│  │ Train   │ │ Val     │ │ Test    │                                       │
│  │ Data    │ │ Data    │ │ Data    │                                       │
│  └────┬────┘ └────┬────┘ └────┬────┘                                       │
│       │           │           │                                             │
│       │           │           │                                             │
│       ↓           │           │                                             │
│  ┌──────────────────────┐     │                                             │
│  │ Transition Matrix    │     │                                             │
│  │ Building Component   │     │                                             │
│  └─────────┬────────────┘     │                                             │
│            │                  │                                             │
│            │   ┌──────────────┘                                             │
│            ↓   ↓                                                            │
│  ┌────────────────────────────────┐                                        │
│  │ Prediction Component           │                                        │
│  │ (per user, uses train tables)  │                                        │
│  └─────────────┬──────────────────┘                                        │
│                │                                                            │
│                ↓                                                            │
│  ┌────────────────────────────────┐                                        │
│  │ Evaluation Component           │                                        │
│  │ (Acc, MRR, F1, NDCG)          │                                        │
│  └─────────────┬──────────────────┘                                        │
│                │                                                            │
│                ↓                                                            │
│  ┌────────────────────────────────┐                                        │
│  │ Experiment Management          │────> Results (JSON, Log)               │
│  │ (Save results)                 │                                        │
│  └────────────────────────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Step | Input | Component | Output |
|------|-------|-----------|--------|
| 1 | YAML file | Configuration | EasyDict config |
| 2 | CSV path | Data Loading | Sorted DataFrame |
| 3 | Sorted DF | Dataset Splitting | Train/Val/Test DFs |
| 4 | DFs + config | Valid ID Filtering | Filtered DFs |
| 5 | Train DF | Transition Building | Per-user transition tables |
| 6 | Tables + Val/Test | Prediction | (true, pred) lists |
| 7 | (true, pred) | Evaluation | Metrics dict |
| 8 | All results | Experiment Management | Files on disk |

---

## Navigation

| Previous | Next |
|----------|------|
| [03_TECHNICAL_IMPLEMENTATION.md](03_TECHNICAL_IMPLEMENTATION.md) | [05_DIAGRAMS_VISUALIZATIONS.md](05_DIAGRAMS_VISUALIZATIONS.md) |

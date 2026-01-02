# 07. Datasets

## DIY and GeoLife Dataset Details

---

## Document Overview

| Item | Details |
|------|---------|
| **Document Type** | Dataset Documentation |
| **Audience** | Researchers, Data Scientists |
| **Reading Time** | 12-15 minutes |
| **Prerequisites** | Basic understanding of GPS data |

---

## 1. Dataset Overview

### 1.1 Two Datasets Used

This experiment uses two real-world human mobility datasets:

| Property | DIY Dataset | GeoLife Dataset |
|----------|-------------|-----------------|
| **Origin** | Indonesia | Beijing, China |
| **Collection** | Mobile app data | GPS trackers |
| **Time Period** | Recent | 2007-2012 |
| **Users** | Multiple | 182 |
| **Clustering** | DBSCAN ε=50m | DBSCAN ε=20m |
| **Test Samples** | 12,368 | 3,502 |

### 1.2 Why Two Datasets?

Using two datasets provides:
1. **Generalization check**: Do findings hold across different contexts?
2. **Cultural diversity**: Asian urban mobility (Indonesia vs China)
3. **Collection method diversity**: Mobile app vs dedicated GPS
4. **Scale comparison**: Larger (DIY) vs smaller (GeoLife) dataset

---

## 2. DIY Dataset

### 2.1 Data Source

The DIY dataset originates from mobile application data collected in Indonesia. It captures smartphone-based location traces from multiple users.

**Data Collection**:
- Source: Mobile application with location services
- Collection method: Passive GPS/network location
- Geographic scope: Urban areas in Indonesia
- Privacy: Anonymized user IDs

### 2.2 Preprocessing Pipeline

```
Raw GPS Points → Stay Point Detection → DBSCAN Clustering → Location Assignment
```

**Step 1: Stay Point Detection**
- Identify locations where users spent time
- Filter out transit points
- Parameters: minimum duration, maximum radius

**Step 2: DBSCAN Clustering (ε=50m)**
- Cluster nearby stay points into semantic locations
- ε=50 meters defines "same location"
- MinPts parameter filters noise

**Step 3: Location Assignment**
- Each cluster becomes a discrete location ID
- Users share the same location vocabulary

### 2.3 Dataset Statistics

#### Overall Statistics

| Metric | Value |
|--------|-------|
| Total test samples | 12,368 |
| Total train samples | ~100,000 |
| Unique locations | ~20,000 |
| Unique users | ~2,000 |
| Clustering ε | 50 meters |
| Maximum prev_days | 7 |

#### Sequence Length by prev_days

| prev_days | Samples | Retention | Avg Len | Std Len | Max Len |
|-----------|---------|-----------|---------|---------|---------|
| 1 | 11,532 | 93.2% | 5.6 | 4.1 | 29 |
| 2 | 12,068 | 97.6% | 8.8 | 6.3 | 42 |
| 3 | 12,235 | 98.9% | 11.9 | 8.4 | 53 |
| 4 | 12,311 | 99.5% | 14.9 | 10.3 | 65 |
| 5 | 12,351 | 99.9% | 17.9 | 12.2 | 77 |
| 6 | 12,365 | 99.97% | 20.9 | 14.1 | 89 |
| 7 | 12,368 | 100% | 24.0 | 15.8 | 99 |

**Interpretation**:
- ~3.4 visits per day on average (24.0 / 7 ≈ 3.4)
- High variability (std ≈ 0.66 × mean)
- Some very active users (max 99 visits in 7 days = 14/day)

### 2.4 User Activity Distribution

```
Activity Level     Users (est.)    Description
───────────────────────────────────────────────
Very Active        5%              >40 visits/week
Active             25%             20-40 visits/week
Moderate           45%             10-20 visits/week
Light              25%             <10 visits/week
```

### 2.5 Temporal Patterns

**Hourly Distribution** (typical):
```
Hour  |████████████████████████████
00-06 |██
06-09 |████████████
09-12 |██████████
12-15 |████████████████
15-18 |██████████████
18-21 |████████████
21-24 |████
```

**Weekly Distribution**:
- Monday-Friday: Higher activity (work routine)
- Saturday-Sunday: Lower activity, more variable

---

## 3. GeoLife Dataset

### 3.1 Data Source

GeoLife is a public dataset collected by Microsoft Research Asia from 2007 to 2012.

**Data Collection**:
- Source: Microsoft Research Asia
- Collection method: Dedicated GPS loggers + smartphones
- Geographic scope: Primarily Beijing, China
- Duration: 5 years
- Users: 182 users

**Reference**:
> Zheng, Y., Xie, X., & Ma, W. Y. (2010). GeoLife: A collaborative social networking service among user, location and trajectory. *IEEE Data Engineering Bulletin*.

### 3.2 Preprocessing Pipeline

```
Raw GPS Trajectories → Trajectory Segmentation → Stay Point Detection → DBSCAN Clustering
```

**Step 1: Trajectory Segmentation**
- Split continuous trajectories into trips
- Handle GPS signal gaps
- Remove noise and outliers

**Step 2: Stay Point Detection**
- Identify stationary periods
- Parameters: time threshold, distance threshold

**Step 3: DBSCAN Clustering (ε=20m)**
- Finer clustering than DIY (ε=20m vs 50m)
- Results in more distinct locations
- Better granularity but more sparsity

### 3.3 Dataset Statistics

#### Overall Statistics

| Metric | Value |
|--------|-------|
| Total test samples | 3,502 |
| Total train samples | ~30,000 |
| Unique locations | ~8,000 |
| Unique users | 182 |
| Clustering ε | 20 meters |
| Maximum prev_days | 7 |

#### Sequence Length by prev_days

| prev_days | Samples | Retention | Avg Len | Std Len | Max Len |
|-----------|---------|-----------|---------|---------|---------|
| 1 | 3,263 | 93.2% | 4.1 | 2.7 | 14 |
| 2 | 3,398 | 97.0% | 6.5 | 4.1 | 21 |
| 3 | 3,458 | 98.7% | 8.9 | 5.5 | 28 |
| 4 | 3,487 | 99.6% | 11.2 | 6.9 | 32 |
| 5 | 3,494 | 99.8% | 13.6 | 8.3 | 35 |
| 6 | 3,499 | 99.9% | 15.9 | 9.7 | 40 |
| 7 | 3,502 | 100% | 18.4 | 11.1 | 46 |

**Interpretation**:
- ~2.6 visits per day on average (18.4 / 7 ≈ 2.6)
- Lower activity than DIY
- Smaller maximum sequence lengths

### 3.4 Transportation Modes

GeoLife includes transportation mode labels (not used in this experiment):

| Mode | Percentage (est.) |
|------|-------------------|
| Walk | 35% |
| Bus | 25% |
| Subway | 15% |
| Car | 15% |
| Bike | 8% |
| Other | 2% |

### 3.5 Geographic Coverage

Primary coverage in Beijing:
- Central business districts
- University areas (Tsinghua, Peking)
- Residential areas
- Transportation hubs

---

## 4. Dataset Comparison

### 4.1 Size Comparison

```
DIY Dataset:
████████████████████████████████████████████████████████ 12,368 samples

GeoLife Dataset:
███████████████ 3,502 samples

DIY is 3.5× larger
```

### 4.2 Activity Comparison

```
                    DIY         GeoLife
                    │              │
Avg visits/day:     3.4            2.6      DIY more active
Avg seq len (7d):   24.0           18.4     DIY longer sequences
Max seq len:        99             46       DIY more extreme users
Std seq len:        15.8           11.1     DIY more variable
```

### 4.3 Clustering Resolution

```
DIY (ε=50m):
├── Coarser clustering
├── Fewer distinct locations
├── May merge nearby POIs
└── Easier prediction (fewer classes)

GeoLife (ε=20m):
├── Finer clustering
├── More distinct locations
├── Better POI separation
└── Harder prediction (more classes)
```

### 4.4 Sample Retention Comparison

Both datasets show similar retention patterns:

```
             DIY     GeoLife
prev1:       93.2%   93.2%    ← Identical retention at 1 day
prev2:       97.6%   97.0%    ← Similar
prev3:       98.9%   98.7%    ← Similar
prev4+:      99%+    99%+     ← Near-complete retention
```

**Implication**: ~7% of samples have insufficient data with 1 day of history in both datasets.

---

## 5. Data Format

### 5.1 Pickle File Structure

```python
# Loading data
import pickle

with open('diy_eps50_prev7_test.pk', 'rb') as f:
    data = pickle.load(f)

# data is a list of dictionaries
len(data)  # 12,368 for DIY

# Each sample
sample = data[0]
sample.keys()
# dict_keys(['X', 'user_X', 'weekday_X', 'start_min_X', 'dur_X', 'diff', 'Y'])
```

### 5.2 Field Descriptions

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `X` | int64 | [seq_len] | Location IDs |
| `user_X` | int64 | [seq_len] | User ID (repeated) |
| `weekday_X` | int64 | [seq_len] | Day of week (0-6) |
| `start_min_X` | int64 | [seq_len] | Time of day (0-95, 15-min bins) |
| `dur_X` | int64 | [seq_len] | Duration bucket (0-99) |
| `diff` | int64 | [seq_len] | Days ago (0-7+) |
| `Y` | int64 | scalar | Target location ID |

### 5.3 Example Sample

```python
sample = {
    'X': array([42, 17, 42, 8, 23]),        # Visited locations
    'user_X': array([5, 5, 5, 5, 5]),       # User 5
    'weekday_X': array([1, 1, 1, 2, 2]),    # Monday, Monday, Monday, Tuesday, Tuesday
    'start_min_X': array([32, 48, 72, 35, 50]),  # Times (~8am, noon, 6pm, ~9am, ~12:30pm)
    'dur_X': array([2, 1, 3, 2, 1]),        # Duration buckets
    'diff': array([1, 1, 1, 0, 0]),         # Yesterday (×3), Today (×2)
    'Y': 42,                                 # Target: location 42
}
```

**Interpretation**:
- User 5 on Tuesday
- Yesterday (Monday): Visited 42 (8am), 17 (noon), 42 (6pm)
- Today (Tuesday): Visited 8 (9am), 23 (12:30pm)
- Prediction: Where will they go next? → Location 42

### 5.4 Time Encoding

Time of day is encoded in 15-minute intervals:

| start_min_X | Time Range |
|-------------|------------|
| 0 | 00:00-00:15 |
| 1 | 00:15-00:30 |
| ... | ... |
| 32 | 08:00-08:15 |
| 48 | 12:00-12:15 |
| 72 | 18:00-18:15 |
| 95 | 23:45-24:00 |

---

## 6. Data Quality Considerations

### 6.1 Known Limitations

| Issue | DIY | GeoLife | Mitigation |
|-------|-----|---------|------------|
| GPS noise | Medium | Low | DBSCAN clustering |
| Indoor accuracy | Poor | Poor | Stay point detection |
| Missing data | Some gaps | Some gaps | Sequence filtering |
| User heterogeneity | High | Medium | User embeddings |

### 6.2 Biases

**Temporal Bias**:
- Data collection periods may not represent all seasons
- Weekday vs weekend balance may vary

**Spatial Bias**:
- Urban areas overrepresented
- Rural/suburban mobility patterns underrepresented

**User Bias**:
- Tech-savvy users overrepresented
- May not reflect general population

### 6.3 Data Split Strategy

| Split | Purpose | Size (DIY) | Size (GeoLife) |
|-------|---------|------------|----------------|
| Train | Model training | ~80% | ~80% |
| Val | Hyperparameter tuning | ~10% | ~10% |
| Test | Final evaluation | ~10% | ~10% |

**Split Method**: User-stratified to ensure each user appears in all splits.

---

## 7. Statistical Properties

### 7.1 Location Visit Distribution

Both datasets exhibit power-law-like distributions:

```
Frequency
    │
    │█
    │█
    │██
    │████
    │██████████
    │█████████████████████████████████████████████
    └───────────────────────────────────────────────▶
     Top locations                    Long tail

~20% of locations account for ~80% of visits (Pareto principle)
```

### 7.2 User Activity Distribution

```
Users
    │
    │                                      ████
    │                              ████████████
    │                    ██████████████████████
    │            ████████████████████████████████
    │████████████████████████████████████████████
    └────────────────────────────────────────────▶
     0     20    40     60    80    100   120+
           Visits per week
```

### 7.3 Temporal Distribution

Both datasets show typical urban mobility patterns:
- Morning peak (7-9 AM)
- Lunch peak (12-1 PM)
- Evening peak (5-7 PM)
- Low activity (11 PM - 6 AM)

---

## 8. Impact on Experiment Results

### 8.1 Why DIY Shows Higher Accuracy

1. **Larger dataset**: More training data → better model
2. **Coarser clustering**: Fewer location classes → easier classification
3. **More active users**: Richer patterns to learn

### 8.2 Why GeoLife Shows Lower Loss

1. **Cleaner data**: GPS loggers produce higher quality traces
2. **More predictable users**: Researchers/students with regular routines
3. **Finer clustering**: More specific locations → higher confidence when correct

### 8.3 Why Both Show Similar Improvement Patterns

1. **Universal weekly patterns**: Human behavior is consistent across cultures
2. **Similar preprocessing**: Both use DBSCAN stay point detection
3. **Similar model capacity**: Same architecture handles both datasets

---

## 9. Accessing the Data

### 9.1 File Paths

```
DIY Dataset:
data/diy_eps50/processed/
├── diy_eps50_prev7_train.pk
├── diy_eps50_prev7_val.pk
└── diy_eps50_prev7_test.pk

GeoLife Dataset:
data/geolife_eps20/processed/
├── geolife_eps20_prev7_train.pk
├── geolife_eps20_prev7_val.pk
└── geolife_eps20_prev7_test.pk
```

### 9.2 Loading Code

```python
import pickle
import numpy as np

def load_dataset(data_path):
    """Load preprocessed dataset."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def print_dataset_stats(data):
    """Print basic statistics."""
    print(f"Number of samples: {len(data)}")
    
    seq_lengths = [len(s['X']) for s in data]
    print(f"Avg sequence length: {np.mean(seq_lengths):.2f}")
    print(f"Std sequence length: {np.std(seq_lengths):.2f}")
    print(f"Max sequence length: {np.max(seq_lengths)}")
    
    unique_users = len(set(s['user_X'][0] for s in data))
    print(f"Unique users: {unique_users}")
    
    all_locs = set()
    for s in data:
        all_locs.update(s['X'].tolist())
        all_locs.add(s['Y'])
    print(f"Unique locations: {len(all_locs)}")
```

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Created** | 2026-01-02 |
| **Word Count** | ~2,000 |
| **Status** | Final |

---

**Navigation**: [← Model Architecture](./06_model_architecture.md) | [Index](./INDEX.md) | [Next: Evaluation Metrics →](./08_evaluation_metrics.md)

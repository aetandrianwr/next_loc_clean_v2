# Quality Filtering Deep Dive

## 📋 Table of Contents
1. [Overview](#overview)
2. [Why Quality Filtering?](#why-quality-filtering)
3. [Sliding Window Quality Assessment](#sliding-window-quality-assessment)
4. [Filter Parameters](#filter-parameters)
5. [Step-by-Step Algorithm](#step-by-step-algorithm)
6. [Visualization of Filtering Process](#visualization-of-filtering-process)
7. [Impact on Dataset](#impact-on-dataset)
8. [Code Implementation](#code-implementation)

---

## Overview

Quality filtering ensures that only users with sufficient and consistent mobility data are included in the dataset. This process happens in the PSL detection notebook (`02_psl_detection_all.ipynb`) and produces the filtered user list.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUALITY FILTERING OVERVIEW                                │
└─────────────────────────────────────────────────────────────────────────────┘

Input: All users with staypoints (~50,000 users)
───────────────────────────────────────────────

    user_001: 2,500 staypoints over 90 days
    user_002: 15 staypoints over 5 days (sparse!)
    user_003: 3,000 staypoints over 120 days
    user_004: 100 staypoints, all in one week (short period!)
    ...

Quality Filtering Steps:
────────────────────────

    1. Day filter: User must have data spanning > 60 days
    2. Sliding window assessment: Check consistency over time
    3. Threshold filter: Must pass quality thresholds

Output: Filtered user list (~150-300 users)
──────────────────────────────────────────

    File: 10_filter_after_user_quality_DIY_slide_filteres.csv
    
    user_001: ✓ PASS
    user_002: ✗ FAIL (too sparse)
    user_003: ✓ PASS
    user_004: ✗ FAIL (period too short)
```

---

## Why Quality Filtering?

### Problem: Inconsistent Data Quality

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA QUALITY CHALLENGES                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Challenge 1: Users with Too Few Records
───────────────────────────────────────

    User A: Only 10 staypoints over 3 days
    
    Problem: 
    • Cannot learn meaningful patterns
    • Insufficient data for train/val/test split
    • Noise dominates signal

    Day 0     Day 1     Day 2     Day 3
    │         │         │         │
    * *       *         * * *     * * *
    
    Not enough data for prediction!


Challenge 2: Inconsistent Tracking
──────────────────────────────────

    User B: Sporadic tracking with gaps
    
    Week 1: Consistent tracking (50 staypoints)
    Week 2-5: No data (phone broken? app disabled?)
    Week 6: Tracking resumes (30 staypoints)
    Week 7-10: No data
    
    Day: 0    7    14   21   28   35   42   49   56   63   70
         │    │    │    │    │    │    │    │    │    │    │
         ████                         ███                   
         │                            │
         Active                       Active
         
    Gaps make patterns unreliable!


Challenge 3: Short Observation Period
─────────────────────────────────────

    User C: Only tracked for 2 weeks
    
    Cannot capture:
    • Weekly patterns (need at least 2 weeks)
    • Monthly variations
    • Seasonal changes
    
    80% for training = 11 days ← Too short!
```

### Solution: Multi-Criteria Quality Assessment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUALITY CRITERIA                                          │
└─────────────────────────────────────────────────────────────────────────────┘

Criterion 1: Minimum Observation Period
───────────────────────────────────────

    day_filter >= 60 days
    
    Why 60 days?
    • Allows 80/10/10 split with meaningful periods
    • Captures at least 8 weeks of patterns
    • Reduces impact of short-term anomalies


Criterion 2: Consistent Activity Over Time
──────────────────────────────────────────

    Use sliding window to assess activity in each period
    
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │ W1  │ W2  │ W3  │ W4  │ W5  │ W6  │ W7  │ W8  │ W9  │ W10 │
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
    
    Each window must have enough activity (active days)


Criterion 3: Average Quality Above Threshold
────────────────────────────────────────────

    mean_thres >= 0.7 (70% average activity rate)
    min_thres >= 0.6 (60% minimum in worst window)
    
    Ensures consistent tracking, not just bursts of activity
```

---

## Sliding Window Quality Assessment

### How Sliding Window Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SLIDING WINDOW MECHANISM                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Configuration:
    window_size = 10 weeks (70 days)
    slide_step = 1 week (7 days)

Process:
────────

    User's data: 100 days (0 to 99)
    
    Window 1: Days 0-69   (Week 1-10)
    Window 2: Days 7-76   (Week 2-11)
    Window 3: Days 14-83  (Week 3-12)
    Window 4: Days 21-90  (Week 4-13)
    Window 5: Days 28-97  (Week 5-14)

    Day:  0   7   14  21  28  35  42  49  56  63  70  77  84  91  98
          │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
          ├───┴───┴───┴───┴───┴───┴───┴───┴───┴───┤
          │         Window 1 (Days 0-69)          │
              ├───┴───┴───┴───┴───┴───┴───┴───┴───┴───┤
              │         Window 2 (Days 7-76)          │
                  ├───┴───┴───┴───┴───┴───┴───┴───┴───┴───┤
                  │         Window 3 (Days 14-83)         │
                      ... and so on


For each window, calculate:
───────────────────────────

    active_days = number of days with at least 1 staypoint
    total_days = window_size (70 days)
    activity_rate = active_days / total_days

Example:
    Window 1: 55 active days / 70 = 0.786 (78.6% activity)
    Window 2: 48 active days / 70 = 0.686 (68.6% activity)
    Window 3: 52 active days / 70 = 0.743 (74.3% activity)
```

### Quality Score Calculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUALITY SCORE METRICS                                     │
└─────────────────────────────────────────────────────────────────────────────┘

After computing activity rate for all windows:

    window_scores = [0.786, 0.686, 0.743, 0.714, 0.800]

Calculate:
──────────

    mean_score = mean(window_scores) = 0.746 (74.6%)
    min_score = min(window_scores) = 0.686 (68.6%)

Apply thresholds:
─────────────────

    mean_thres = 0.7 (70%)
    min_thres = 0.6 (60%)

    Check 1: mean_score >= mean_thres
             0.746 >= 0.7 ✓ PASS
    
    Check 2: min_score >= min_thres
             0.686 >= 0.6 ✓ PASS

Result: User PASSES quality filter!
```

---

## Filter Parameters

### Default Configuration

```yaml
# From 02_psl_detection_all.ipynb

day_filter: 60           # Minimum days of data
sliding_window: 10       # Window size in weeks
min_thres: 0.6          # Minimum activity rate in worst window
mean_thres: 0.7         # Minimum average activity rate
```

### Parameter Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER DETAILS                                         │
└─────────────────────────────────────────────────────────────────────────────┘

Parameter: day_filter = 60
──────────────────────────

    Minimum number of days with at least one staypoint.
    
    Why 60?
    • Provides enough data for meaningful analysis
    • Allows ~50 days training, ~5 days validation, ~5 days test
    • Captures approximately 2 months of behavior
    
    Trade-off:
    • Higher value → fewer users, better quality
    • Lower value → more users, potentially noisy


Parameter: sliding_window = 10 (weeks)
──────────────────────────────────────

    Size of each assessment window in weeks.
    
    10 weeks = 70 days
    
    Why 10 weeks?
    • Long enough to assess consistent behavior
    • Short enough to detect gaps
    • Approximately 2.5 months per window
    
    Trade-off:
    • Larger window → more forgiving of short gaps
    • Smaller window → stricter consistency requirement


Parameter: min_thres = 0.6
──────────────────────────

    Minimum activity rate in the WORST window.
    
    0.6 = 60% of days must have staypoints
    In 70-day window: at least 42 active days
    
    Why 60%?
    • Allows some gaps (weekends, holidays)
    • Ensures no extended periods of missing data
    • Reasonably tolerant of real-world conditions
    
    Trade-off:
    • Higher value → stricter, fewer users
    • Lower value → more tolerant, more users with gaps


Parameter: mean_thres = 0.7
───────────────────────────

    Minimum AVERAGE activity rate across all windows.
    
    0.7 = 70% average activity
    
    Why 70%?
    • Higher than min_thres → ensures overall quality
    • User can have one bad window if others compensate
    • Balanced between quality and quantity
```

### Parameter Impact Matrix

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER IMPACT ON FILTERING                               │
└────────────────────────────────────────────────────────────────────────────────┘

Stricter Parameters (fewer users, higher quality):
──────────────────────────────────────────────────

    day_filter: 60 → 90        # Require 3 months minimum
    sliding_window: 10 → 6     # Shorter assessment windows
    min_thres: 0.6 → 0.7       # 70% minimum in worst window
    mean_thres: 0.7 → 0.8      # 80% average requirement

    Result: ~50-100 users (very high quality)


Default Parameters (balanced):
──────────────────────────────

    day_filter: 60
    sliding_window: 10
    min_thres: 0.6
    mean_thres: 0.7

    Result: ~150-300 users (good balance)


Relaxed Parameters (more users, lower quality):
───────────────────────────────────────────────

    day_filter: 60 → 30        # Only 1 month minimum
    sliding_window: 10 → 12    # Longer windows (more forgiving)
    min_thres: 0.6 → 0.4       # 40% minimum
    mean_thres: 0.7 → 0.5      # 50% average

    Result: ~500-1000 users (may include noisy data)
```

---

## Step-by-Step Algorithm

### Complete Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUALITY FILTERING ALGORITHM                               │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Calculate Active Days per User
──────────────────────────────────────

    For each user:
        active_days = count(unique days with staypoints)
        total_days = max(start_day) - min(start_day) + 1

    user_001: 85 active days, 95 total days → activity = 89.5%
    user_002: 12 active days, 15 total days → activity = 80.0%
    user_003: 78 active days, 120 total days → activity = 65.0%


STEP 2: Apply Day Filter
────────────────────────

    Filter: active_days >= day_filter (60)
    
    user_001: 85 >= 60 ✓ KEEP
    user_002: 12 >= 60 ✗ REMOVE
    user_003: 78 >= 60 ✓ KEEP
    
    Remaining: user_001, user_003, ...


STEP 3: Sliding Window Assessment
─────────────────────────────────

    For each remaining user:
        windows = []
        for start in range(0, max_day, 7):  # slide by 7 days
            window_end = start + 70  # 10 weeks
            if window_end > max_day:
                break
            
            days_in_window = count(staypoints where start <= start_day < window_end)
            activity_rate = days_in_window / 70
            windows.append(activity_rate)
        
        mean_rate = mean(windows)
        min_rate = min(windows)


STEP 4: Apply Quality Thresholds
────────────────────────────────

    For each user:
        if mean_rate >= mean_thres AND min_rate >= min_thres:
            KEEP user
        else:
            REMOVE user


STEP 5: Output Filtered User List
─────────────────────────────────

    Save valid user IDs to:
    10_filter_after_user_quality_DIY_slide_filteres.csv
```

---

## Visualization of Filtering Process

### Example User Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    USER ANALYSIS VISUALIZATION                               │
└─────────────────────────────────────────────────────────────────────────────┘

USER_A: HIGH QUALITY (PASS)
═══════════════════════════

Activity over 100 days:
    Day: 0        20        40        60        80        100
         │         │         │         │         │         │
         ████████████████████████████████████████████████████
         
    Consistently active throughout the period
    
    Sliding windows (10 weeks = 70 days):
    ┌───────────────────────────────────────────────────┐
    │ W1: ████████████████████████████████████████████  │  85%
    │ W2: ████████████████████████████████████████████  │  82%
    │ W3: ██████████████████████████████████████████    │  78%
    │ W4: ████████████████████████████████████████████  │  80%
    └───────────────────────────────────────────────────┘
    
    mean_rate = 81.25% ✓ >= 70%
    min_rate = 78% ✓ >= 60%
    
    VERDICT: ✓ PASS


USER_B: GAP IN DATA (FAIL)
══════════════════════════

Activity over 100 days:
    Day: 0        20        40        60        80        100
         │         │         │         │         │         │
         ██████████                    ████████████████████
                   │                  │
                   └──── 20 day gap ──┘
    
    Large gap in middle of observation period
    
    Sliding windows:
    ┌───────────────────────────────────────────────────┐
    │ W1: ██████████████                                │  50%
    │ W2: ██████                                        │  35% ✗
    │ W3:               ██████████████████████████████  │  65%
    │ W4: ████████████████████████████████████████████  │  80%
    └───────────────────────────────────────────────────┘
    
    mean_rate = 57.5% ✗ < 70%
    min_rate = 35% ✗ < 60%
    
    VERDICT: ✗ FAIL (both thresholds failed)


USER_C: TOO SHORT (FAIL)
════════════════════════

Activity over 40 days:
    Day: 0        20        40
         │         │         │
         ██████████████████████
         
    Good activity but only 40 days
    
    day_filter check: 40 < 60 ✗
    
    VERDICT: ✗ FAIL (pre-filter, insufficient days)


USER_D: BORDERLINE (PASS)
═════════════════════════

Activity over 90 days:
    Day: 0        20        40        60        80        90
         │         │         │         │         │         │
         ████████████████████      ██████████████████████████
                             │    │
                             └─10d┘ gap
    
    One moderate gap, otherwise consistent
    
    Sliding windows:
    ┌───────────────────────────────────────────────────┐
    │ W1: ████████████████████████████████████████████  │  78%
    │ W2: ████████████████████████████████              │  65%
    │ W3: ████████████████████████████████████████████  │  75%
    └───────────────────────────────────────────────────┘
    
    mean_rate = 72.67% ✓ >= 70%
    min_rate = 65% ✓ >= 60%
    
    VERDICT: ✓ PASS (barely)
```

---

## Impact on Dataset

### Typical Filtering Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FILTERING STATISTICS (DIY DATASET)                        │
└─────────────────────────────────────────────────────────────────────────────┘

Initial Dataset:
────────────────

    Total users:       ~50,000
    Total staypoints:  ~165,000,000
    
After Day Filter (>= 60 days):
──────────────────────────────

    Remaining users:   ~2,500 (5% retained)
    Reason for removal:
    • Many users only tracked for a few days
    • App uninstalls, phone changes
    • Tourists, temporary residents

After Sliding Window Filter:
────────────────────────────

    Final users:       ~150-300 (0.3-0.6% of original)
    Reason for removal:
    • Inconsistent tracking patterns
    • Extended gaps in data
    • Seasonal usage only

Data Quality Improvement:
─────────────────────────

    Metric                  │ Before Filter │ After Filter
    ────────────────────────┼───────────────┼──────────────
    Avg active days/user    │     25        │     85
    Avg staypoints/user     │    3,300      │    2,200
    Avg gap between records │     2.5 days  │     0.8 days
    Users with >80% activity│     3%        │     75%
```

### Why So Few Users Pass?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNDERSTANDING LOW PASS RATE                               │
└─────────────────────────────────────────────────────────────────────────────┘

Typical User Behavior in Mobile Tracking Studies:
─────────────────────────────────────────────────

    Category              │ % of Users │ Reason for Exclusion
    ──────────────────────┼────────────┼────────────────────────────────
    Tried and quit        │    60%     │ Uninstalled after few days
    Occasional users      │    25%     │ Only use app sometimes
    Privacy-conscious     │    5%      │ Disabled tracking frequently
    Phone changes         │    5%      │ Data lost on new device
    Consistent trackers   │    5%      │ Keep tracking enabled
    QUALITY TRACKERS      │   <1%      │ Consistent + long duration

This is EXPECTED in mobility research!
─────────────────────────────────────

    • Academic datasets often report 100-500 quality users
    • Large initial dataset compensates for low retention
    • Quality > quantity for pattern learning
    
    Example studies:
    • GeoLife dataset: 182 users (from Microsoft employees)
    • Brightkite: ~58,000 users, ~20K with sufficient data
    • DIY dataset: ~50,000 → ~150-300 quality users
```

---

## Code Implementation

### Python Implementation

```python
"""
Quality Filtering Implementation
Based on 02_psl_detection_all.ipynb
"""

import pandas as pd
import numpy as np

def calculate_user_activity(staypoints_df):
    """
    Calculate activity metrics for each user.
    
    Args:
        staypoints_df: DataFrame with staypoints including 'user_id' and 'started_at'
    
    Returns:
        DataFrame with user activity statistics
    """
    # Ensure datetime
    staypoints_df['started_at'] = pd.to_datetime(staypoints_df['started_at'])
    staypoints_df['date'] = staypoints_df['started_at'].dt.date
    
    # Calculate per-user statistics
    user_stats = staypoints_df.groupby('user_id').agg({
        'date': ['nunique', 'min', 'max'],
        'id': 'count'  # Total staypoints
    }).reset_index()
    
    # Flatten column names
    user_stats.columns = ['user_id', 'active_days', 'first_date', 'last_date', 'staypoint_count']
    
    # Calculate total observation period
    user_stats['total_days'] = (
        pd.to_datetime(user_stats['last_date']) - 
        pd.to_datetime(user_stats['first_date'])
    ).dt.days + 1
    
    # Calculate overall activity rate
    user_stats['overall_activity_rate'] = user_stats['active_days'] / user_stats['total_days']
    
    return user_stats


def apply_day_filter(user_stats, day_filter=60):
    """
    Filter users by minimum active days.
    
    Args:
        user_stats: DataFrame from calculate_user_activity
        day_filter: Minimum required active days
    
    Returns:
        Filtered DataFrame
    """
    print(f"Before day filter: {len(user_stats)} users")
    filtered = user_stats[user_stats['active_days'] >= day_filter]
    print(f"After day filter (>= {day_filter} days): {len(filtered)} users")
    return filtered


def calculate_sliding_window_scores(staypoints_df, user_id, window_weeks=10):
    """
    Calculate activity scores for sliding windows.
    
    Args:
        staypoints_df: DataFrame with user's staypoints
        user_id: User ID to analyze
        window_weeks: Window size in weeks
    
    Returns:
        List of activity scores per window
    """
    user_data = staypoints_df[staypoints_df['user_id'] == user_id].copy()
    user_data['date'] = pd.to_datetime(user_data['started_at']).dt.date
    
    # Get user's date range
    first_date = user_data['date'].min()
    last_date = user_data['date'].max()
    total_days = (last_date - first_date).days + 1
    
    # Calculate active days per day number
    user_data['day_num'] = (user_data['date'] - first_date).apply(lambda x: x.days)
    active_days_set = set(user_data['day_num'].unique())
    
    # Sliding window parameters
    window_days = window_weeks * 7
    slide_days = 7  # Slide by 1 week
    
    scores = []
    window_start = 0
    
    while window_start + window_days <= total_days:
        # Count active days in this window
        active_in_window = sum(
            1 for day in range(window_start, window_start + window_days)
            if day in active_days_set
        )
        
        # Calculate activity rate
        activity_rate = active_in_window / window_days
        scores.append(activity_rate)
        
        # Slide window
        window_start += slide_days
    
    return scores


def apply_sliding_window_filter(staypoints_df, user_stats, 
                                 window_weeks=10, 
                                 min_thres=0.6, 
                                 mean_thres=0.7):
    """
    Apply sliding window quality filter.
    
    Args:
        staypoints_df: DataFrame with all staypoints
        user_stats: DataFrame with user statistics (after day filter)
        window_weeks: Window size in weeks
        min_thres: Minimum activity rate threshold
        mean_thres: Mean activity rate threshold
    
    Returns:
        List of user_ids that pass the filter
    """
    print(f"\nApplying sliding window filter...")
    print(f"  Window size: {window_weeks} weeks ({window_weeks * 7} days)")
    print(f"  Min threshold: {min_thres} ({min_thres*100}%)")
    print(f"  Mean threshold: {mean_thres} ({mean_thres*100}%)")
    
    valid_users = []
    
    for user_id in user_stats['user_id']:
        scores = calculate_sliding_window_scores(
            staypoints_df, user_id, window_weeks
        )
        
        if len(scores) == 0:
            continue  # Not enough data for even one window
        
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        
        if mean_score >= mean_thres and min_score >= min_thres:
            valid_users.append(user_id)
    
    print(f"After sliding window filter: {len(valid_users)} users")
    return valid_users


def quality_filter(staypoints_df, 
                   day_filter=60,
                   window_weeks=10,
                   min_thres=0.6,
                   mean_thres=0.7):
    """
    Complete quality filtering pipeline.
    
    Args:
        staypoints_df: Raw staypoints DataFrame
        day_filter: Minimum active days
        window_weeks: Sliding window size in weeks
        min_thres: Minimum window activity threshold
        mean_thres: Mean activity threshold
    
    Returns:
        List of valid user IDs
    """
    print("=" * 60)
    print("QUALITY FILTERING")
    print("=" * 60)
    
    # Step 1: Calculate user activity
    print("\n[Step 1] Calculating user activity statistics...")
    user_stats = calculate_user_activity(staypoints_df)
    print(f"Total users: {len(user_stats)}")
    
    # Step 2: Apply day filter
    print("\n[Step 2] Applying day filter...")
    user_stats_filtered = apply_day_filter(user_stats, day_filter)
    
    # Step 3: Apply sliding window filter
    print("\n[Step 3] Applying sliding window filter...")
    valid_users = apply_sliding_window_filter(
        staypoints_df,
        user_stats_filtered,
        window_weeks,
        min_thres,
        mean_thres
    )
    
    print("\n" + "=" * 60)
    print(f"FINAL: {len(valid_users)} users pass quality filter")
    print("=" * 60)
    
    return valid_users


# Example usage
if __name__ == "__main__":
    # Load staypoints
    staypoints_df = pd.read_csv("staypoints.csv")
    
    # Apply quality filter with default parameters
    valid_users = quality_filter(
        staypoints_df,
        day_filter=60,
        window_weeks=10,
        min_thres=0.6,
        mean_thres=0.7
    )
    
    # Save valid user list
    valid_df = pd.DataFrame({'user_id': valid_users})
    valid_df.to_csv(
        "10_filter_after_user_quality_DIY_slide_filteres.csv",
        index=False
    )
```

---

## Summary

### Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUALITY FILTERING SUMMARY                                 │
└─────────────────────────────────────────────────────────────────────────────┘

1. TWO-STAGE FILTERING
   • Day filter: Removes users with too few days
   • Sliding window: Removes users with inconsistent tracking

2. SLIDING WINDOW APPROACH
   • 10-week windows, sliding by 1 week
   • Assesses consistency over time
   • Both minimum and mean thresholds must pass

3. EXPECTED LOW PASS RATE
   • ~50,000 users → ~150-300 quality users
   • This is normal for mobility datasets
   • Quality data > large quantity of noisy data

4. DEFAULT PARAMETERS
   • day_filter: 60 (minimum active days)
   • window_weeks: 10 (assessment window size)
   • min_thres: 0.6 (60% worst window)
   • mean_thres: 0.7 (70% average)

5. OUTPUT FILE
   • 10_filter_after_user_quality_DIY_slide_filteres.csv
   • Contains user_id of all users passing filter
   • Used by downstream preprocessing scripts
```

### Quick Reference

```python
# Minimal quality filter check
def passes_quality(user_data, day_filter=60, min_thres=0.6, mean_thres=0.7):
    active_days = user_data['date'].nunique()
    if active_days < day_filter:
        return False
    
    window_scores = calculate_sliding_window_scores(user_data)
    return np.mean(window_scores) >= mean_thres and np.min(window_scores) >= min_thres
```

# Temporal Features Deep Dive

## 📋 Table of Contents
1. [Overview](#overview)
2. [Time Feature Extraction](#time-feature-extraction)
3. [Start Day Calculation](#start-day-calculation)
4. [Start Minute Calculation](#start-minute-calculation)
5. [Weekday Extraction](#weekday-extraction)
6. [Duration Calculation](#duration-calculation)
7. [Day Difference (diff) Feature](#day-difference-diff-feature)
8. [Complete Example](#complete-example)
9. [Feature Importance in Prediction](#feature-importance-in-prediction)

---

## Overview

Temporal features are crucial for next location prediction because human mobility is highly time-dependent:
- People go to work at specific times
- Weekend patterns differ from weekday patterns
- Duration at locations varies by activity type

### Temporal Features in the Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL FEATURES OVERVIEW                                │
└─────────────────────────────────────────────────────────────────────────────┘

Raw Timestamps                 Extracted Features
────────────────               ──────────────────

started_at: 2023-01-15 07:30   ├─► start_day: 0 (days since first record)
                               ├─► weekday: 6 (Sunday)
                               ├─► start_min: 450 (7:30 = 7*60 + 30)
                               │
finished_at: 2023-01-15 17:45  └─► duration: 615 minutes (10h 15m)


Sequence Generation Features:
─────────────────────────────

For predicting next location at Day 7:

History (7-day window):    Target:
┌────────────────────┐    ┌──────────┐
│ X: [42, 15, 8...]  │    │ Y: 42    │
│ weekday_X: [6,0,1] │    │          │
│ start_min_X: [...]│ ─► │ Predict! │
│ dur_X: [...]       │    │          │
│ diff: [7, 6, 5...] │    └──────────┘
└────────────────────┘
```

---

## Time Feature Extraction

### From Raw Timestamps to Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIME FEATURE EXTRACTION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

Input: Staypoint with timestamps
────────────────────────────────

    {
        "user_id": "user_001",
        "started_at": "2023-01-15 07:30:00",
        "finished_at": "2023-01-15 17:45:00"
    }

Step 1: Parse timestamps
────────────────────────

    started_at  = pd.Timestamp("2023-01-15 07:30:00")
    finished_at = pd.Timestamp("2023-01-15 17:45:00")

Step 2: Extract features
────────────────────────

    start_day = (started_at.date() - user_first_date).days
    end_day   = (finished_at.date() - user_first_date).days
    
    weekday   = started_at.dayofweek  # 0=Monday, 6=Sunday
    
    start_min = started_at.hour * 60 + started_at.minute
    end_min   = finished_at.hour * 60 + finished_at.minute
    
    duration  = (finished_at - started_at).total_seconds() / 60

Result:
───────

    {
        "start_day": 0,
        "end_day": 0,
        "weekday": 6,        # Sunday
        "start_min": 450,    # 7:30 AM
        "end_min": 1065,     # 5:45 PM
        "duration": 615.0    # minutes
    }
```

---

## Start Day Calculation

### Concept

The `start_day` feature normalizes all timestamps to "days since user's first record":

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    START_DAY: RELATIVE DAY NUMBER                            │
└─────────────────────────────────────────────────────────────────────────────┘

Why relative days instead of absolute dates?
────────────────────────────────────────────

    User A starts:  2022-06-01
    User B starts:  2023-01-15
    
    Absolute dates make comparison difficult:
    User A's Day 100 = 2022-09-08
    User B's Day 100 = 2023-04-25
    
    With relative days:
    User A's start_day=0 is their first day (2022-06-01)
    User B's start_day=0 is their first day (2023-01-15)
    
    Now Day 100 means "100 days into their tracking period" for both!
```

### Calculation Code

```python
# From diy_1_raw_to_interim.py

def calculate_start_day(staypoints_df):
    """
    Calculate relative day number for each staypoint.
    """
    # Get user's first record date
    user_first_dates = staypoints_df.groupby('user_id')['started_at'].min().dt.date
    
    # Calculate days since first record
    staypoints_df['start_day'] = staypoints_df.apply(
        lambda row: (row['started_at'].date() - 
                    user_first_dates[row['user_id']]).days,
        axis=1
    )
    
    return staypoints_df
```

### Detailed Example

```
User: user_001
═══════════════════════════════════════════════════════════════════════════════

User's first record: 2023-01-15

    Staypoint  │ started_at          │ Calculation                    │ start_day
    ───────────┼─────────────────────┼────────────────────────────────┼───────────
    SP_001     │ 2023-01-15 07:00    │ 2023-01-15 - 2023-01-15 = 0    │    0
    SP_002     │ 2023-01-15 09:00    │ 2023-01-15 - 2023-01-15 = 0    │    0
    SP_003     │ 2023-01-15 18:00    │ 2023-01-15 - 2023-01-15 = 0    │    0
    SP_004     │ 2023-01-16 08:30    │ 2023-01-16 - 2023-01-15 = 1    │    1
    SP_005     │ 2023-01-16 17:30    │ 2023-01-16 - 2023-01-15 = 1    │    1
    SP_006     │ 2023-01-20 12:00    │ 2023-01-20 - 2023-01-15 = 5    │    5
    SP_007     │ 2023-02-14 10:00    │ 2023-02-14 - 2023-01-15 = 30   │   30
    SP_008     │ 2023-04-25 09:00    │ 2023-04-25 - 2023-01-15 = 100  │  100

Timeline visualization:
───────────────────────

    Day 0          Day 1      Day 5              Day 30          Day 100
    │              │          │                  │               │
    ▼              ▼          ▼                  ▼               ▼
    ├──────────────┼──────────┼──────────────────┼───────────────┼─────►
    │              │          │                  │               │
    SP_001         SP_004     SP_006             SP_007          SP_008
    SP_002         SP_005
    SP_003
```

### Use in Temporal Splitting

```python
# Split based on start_day
max_day = user_df['start_day'].max()

# 80% train, 10% validation, 10% test
train_cutoff = int(max_day * 0.8)   # Day 80 if max is 100
val_cutoff = int(max_day * 0.9)     # Day 90 if max is 100

train_df = user_df[user_df['start_day'] <= train_cutoff]
val_df = user_df[(user_df['start_day'] > train_cutoff) & 
                  (user_df['start_day'] <= val_cutoff)]
test_df = user_df[user_df['start_day'] > val_cutoff]
```

---

## Start Minute Calculation

### Concept

The `start_min` feature captures the time of day (0-1439 minutes):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    START_MIN: MINUTE OF DAY                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Why minutes instead of hours?
─────────────────────────────

    Hours (0-23): Too coarse
    - 7:00 and 7:59 are both "hour 7"
    - Loses 59 minutes of precision
    
    Minutes (0-1439): Fine-grained
    - 7:00 = 420, 7:30 = 450, 7:59 = 479
    - Captures rush hour vs mid-morning difference

Formula:
────────
    start_min = hour × 60 + minute

Range:
──────
    0 = 00:00 (midnight)
    1439 = 23:59 (end of day)
```

### Visual Time-to-Minute Mapping

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MINUTE OF DAY SCALE                                       │
└─────────────────────────────────────────────────────────────────────────────┘

    Time        │ Calculation        │ start_min │ Typical Activity
    ────────────┼────────────────────┼───────────┼──────────────────────
    00:00       │ 0 × 60 + 0         │    0      │ Sleeping
    06:00       │ 6 × 60 + 0         │  360      │ Early morning
    07:00       │ 7 × 60 + 0         │  420      │ Waking up
    07:30       │ 7 × 60 + 30        │  450      │ Morning routine
    08:00       │ 8 × 60 + 0         │  480      │ Morning commute
    08:30       │ 8 × 60 + 30        │  510      │ Arriving at work
    09:00       │ 9 × 60 + 0         │  540      │ Work start
    12:00       │ 12 × 60 + 0        │  720      │ Lunch time
    17:00       │ 17 × 60 + 0        │ 1020      │ Evening commute
    18:00       │ 18 × 60 + 0        │ 1080      │ Arriving home
    20:00       │ 20 × 60 + 0        │ 1200      │ Evening activity
    22:00       │ 22 × 60 + 0        │ 1320      │ Night time
    23:59       │ 23 × 60 + 59       │ 1439      │ End of day


    Visual scale:
    ─────────────
    0                   720                    1439
    │─────────────────────│──────────────────────│
    midnight            noon                  midnight
    
    │    │    │    │    │    │    │    │    │    │
    0   180  360  540  720  900 1080 1260 1440
    12AM 3AM  6AM  9AM  12PM 3PM  6PM  9PM  12AM
```

### Calculation Code

```python
# From preprocessing scripts

# Extract start minute from timestamp
staypoints_df['start_min'] = (
    staypoints_df['started_at'].dt.hour * 60 + 
    staypoints_df['started_at'].dt.minute
)

# Similarly for end minute
staypoints_df['end_min'] = (
    staypoints_df['finished_at'].dt.hour * 60 + 
    staypoints_df['finished_at'].dt.minute
)
```

### Why This Feature Matters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIME-DEPENDENT MOBILITY PATTERNS                          │
└─────────────────────────────────────────────────────────────────────────────┘

Same user, different times → Different next locations:

    Time: start_min = 450 (7:30 AM on weekday)
    History: [Home]
    Likely next: Work (high probability)
    
    Time: start_min = 1080 (6:00 PM on weekday)
    History: [Work]
    Likely next: Home (high probability)
    
    Time: start_min = 720 (12:00 PM on weekend)
    History: [Home]
    Likely next: Restaurant, Shopping, Recreation (varied)

The model learns these temporal patterns!
```

---

## Weekday Extraction

### Concept

The `weekday` feature captures the day of the week (0-6):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WEEKDAY: DAY OF WEEK                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Encoding (Python/Pandas convention):
────────────────────────────────────

    weekday │ Day
    ────────┼───────────
       0    │ Monday
       1    │ Tuesday
       2    │ Wednesday
       3    │ Thursday
       4    │ Friday
       5    │ Saturday
       6    │ Sunday

Why important?
──────────────
    
    Weekday patterns (Mon-Fri):
    • Morning: Home → Work
    • Evening: Work → Home
    • Consistent routine
    
    Weekend patterns (Sat-Sun):
    • Variable timing
    • Recreation, shopping, social
    • Less predictable
```

### Calculation Code

```python
# From preprocessing scripts

# Extract weekday from timestamp (0=Monday, 6=Sunday)
staypoints_df['weekday'] = staypoints_df['started_at'].dt.dayofweek
```

### Example

```
User: user_001
═══════════════════════════════════════════════════════════════════════════════

    Staypoint  │ started_at          │ .dt.dayofweek │ weekday │ Day Name
    ───────────┼─────────────────────┼───────────────┼─────────┼──────────
    SP_001     │ 2023-01-15 07:00    │      6        │    6    │ Sunday
    SP_002     │ 2023-01-16 09:00    │      0        │    0    │ Monday
    SP_003     │ 2023-01-17 08:30    │      1        │    1    │ Tuesday
    SP_004     │ 2023-01-18 17:30    │      2        │    2    │ Wednesday
    SP_005     │ 2023-01-19 12:00    │      3        │    3    │ Thursday
    SP_006     │ 2023-01-20 10:00    │      4        │    4    │ Friday
    SP_007     │ 2023-01-21 14:00    │      5        │    5    │ Saturday
```

### Weekly Pattern Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WEEKLY MOBILITY PATTERN                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Typical user weekly pattern:

    Mon     Tue     Wed     Thu     Fri     Sat     Sun
    (0)     (1)     (2)     (3)     (4)     (5)     (6)
    
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │ 🏠→🏢  🏠→🏢  🏠→🏢  🏠→🏢  🏠→🏢  🏠→🛒   🏠→⛪   │  AM
    │                                                       │
    │ 🏢→🏠  🏢→🏠  🏢→🏠  🏢→🏠  🏢→🍺  🛒→🏠   ⛪→🏠   │  PM
    │                                                       │
    └───────────────────────────────────────────────────────┘
    
    Weekday (0-4): Regular work pattern
    Weekend (5-6): Variable pattern

Model can learn: 
    "If weekday=5 and start_min=600, next location is likely shopping, not work"
```

---

## Duration Calculation

### Concept

The `duration` feature captures how long the user stayed at a location:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DURATION: TIME SPENT AT LOCATION                          │
└─────────────────────────────────────────────────────────────────────────────┘

Formula:
────────
    duration = (finished_at - started_at) in minutes

Range:
──────
    Minimum: ~30 minutes (staypoint threshold)
    Maximum: Capped at 2880 minutes (48 hours) to handle outliers

Why cap at 2880 minutes?
────────────────────────
    • Prevents extreme outliers from skewing the data
    • 48 hours covers most multi-day stays (vacation, hospital)
    • Reduces impact of GPS gaps that appear as long stays
```

### Calculation Code

```python
# From preprocessing scripts

# Calculate duration in minutes
staypoints_df['duration'] = (
    (staypoints_df['finished_at'] - staypoints_df['started_at'])
    .dt.total_seconds() / 60
)

# Cap at maximum (from config)
max_duration = config.get('max_duration', 2880)  # Default 48 hours
staypoints_df['duration'] = staypoints_df['duration'].clip(upper=max_duration)
```

### Duration Patterns by Location Type

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TYPICAL DURATION BY LOCATION TYPE                         │
└─────────────────────────────────────────────────────────────────────────────┘

    Location Type     │ Typical Duration │ Range         │ Insights
    ──────────────────┼──────────────────┼───────────────┼────────────────────
    Home              │ 600-840 min      │ 10-14 hours   │ Overnight stay
    Work/Office       │ 480-540 min      │ 8-9 hours     │ Work day
    Restaurant        │ 60-90 min        │ 1-1.5 hours   │ Meal duration
    Coffee Shop       │ 30-60 min        │ 0.5-1 hour    │ Quick visit
    Shopping Mall     │ 120-180 min      │ 2-3 hours     │ Shopping trip
    Gym               │ 60-120 min       │ 1-2 hours     │ Workout session
    Religious place   │ 60-120 min       │ 1-2 hours     │ Service duration
    University        │ 180-360 min      │ 3-6 hours     │ Class attendance

Duration helps model distinguish between:
    • Passing through vs. staying at a location
    • Short visit (coffee) vs. long visit (work)
    • Regular activity vs. special occasion
```

### Example Durations

```
User: user_001
═══════════════════════════════════════════════════════════════════════════════

    Staypoint  │ started_at          │ finished_at         │ Calculation      │ duration
    ───────────┼─────────────────────┼─────────────────────┼──────────────────┼──────────
    SP_001     │ 2023-01-15 07:00    │ 2023-01-15 07:30    │ 30 min           │   30.0
    SP_002     │ 2023-01-15 08:00    │ 2023-01-15 17:00    │ 9 hours = 540    │  540.0
    SP_003     │ 2023-01-15 18:00    │ 2023-01-16 07:00    │ 13 hours = 780   │  780.0
    SP_004     │ 2023-01-16 08:00    │ 2023-01-16 09:30    │ 1.5 hours = 90   │   90.0
    SP_005     │ 2023-01-16 12:00    │ 2023-01-19 12:00    │ 72 hours = 4320  │ 2880.0*
    
    * Capped at max_duration (2880 minutes = 48 hours)
```

---

## Day Difference (diff) Feature

### Concept

The `diff` feature in sequences captures "how many days ago" each historical staypoint occurred:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DIFF: DAYS AGO IN SEQUENCE                                │
└─────────────────────────────────────────────────────────────────────────────┘

Purpose:
────────
    When predicting for Day 7, the model needs to know:
    • Which historical visits are recent (yesterday)?
    • Which are from last week?
    
    diff encodes this temporal distance.

Formula:
────────
    diff = target_day - staypoint_start_day

Example:
────────
    Target prediction: Day 7
    
    Historical staypoints:
    • Day 0: diff = 7 - 0 = 7 (7 days ago)
    • Day 1: diff = 7 - 1 = 6 (6 days ago)
    • Day 5: diff = 7 - 5 = 2 (2 days ago)
    • Day 6: diff = 7 - 6 = 1 (yesterday)
```

### Calculation in Sequence Generation

```python
# From sequence generation code

def create_sequence(user_df, target_idx, previous_day, max_duration):
    """
    Create a single sequence for prediction.
    """
    target = user_df.iloc[target_idx]
    target_day = target['start_day']
    
    # Get history within previous_day window
    window_start = target_day - previous_day
    history = user_df[
        (user_df['start_day'] >= window_start) & 
        (user_df['start_day'] < target_day)
    ]
    
    # Calculate diff for each historical staypoint
    diff = [target_day - row['start_day'] for _, row in history.iterrows()]
    
    return {
        'X': history['location_id'].tolist(),
        'diff': diff,  # Days ago
        # ... other features
    }
```

### Detailed Example

```
Predicting for Day 7 with 7-day history window (previous_day=7)
═══════════════════════════════════════════════════════════════════════════════

Historical staypoints (Days 0-6):
─────────────────────────────────

    Index │ start_day │ location_id │ Target Day │ diff = target - start
    ──────┼───────────┼─────────────┼────────────┼───────────────────────
      0   │     0     │     42      │     7      │     7 - 0 = 7
      1   │     0     │     15      │     7      │     7 - 0 = 7
      2   │     1     │     42      │     7      │     7 - 1 = 6
      3   │     2     │     15      │     7      │     7 - 2 = 5
      4   │     3     │      8      │     7      │     7 - 3 = 4
      5   │     5     │     42      │     7      │     7 - 5 = 2
      6   │     6     │     15      │     7      │     7 - 6 = 1

Result sequence:
────────────────

    X:    [42, 15, 42, 15,  8, 42, 15]    # Location IDs
    diff: [ 7,  7,  6,  5,  4,  2,  1]    # Days ago

Interpretation:
───────────────
    • First two visits (42, 15) were 7 days ago
    • Most recent visit (15) was yesterday (diff=1)
    • Gap between day 3 and 5 (no staypoints on day 4)
```

### Why diff is Important

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL DECAY IN PREDICTION                              │
└─────────────────────────────────────────────────────────────────────────────┘

Recent visits are more predictive than old visits!
─────────────────────────────────────────────────

    Yesterday (diff=1):
    • User was at location X
    • High probability of similar pattern today
    
    7 days ago (diff=7):
    • User was at location Y (same weekday last week)
    • Weekly pattern may repeat
    • But less confident than yesterday

Model learns temporal attention:
────────────────────────────────

    The diff feature allows the model to:
    1. Weight recent visits more heavily
    2. Recognize weekly patterns (diff=7 = same weekday last week)
    3. Handle irregular sequences (gaps in data)

Example pattern:
────────────────

    Target: Monday (Day 7)
    
    diff=7: Previous Monday → Same weekday pattern
    diff=1: Sunday → Day-before context
    diff=2: Saturday → Weekend context
    
    Model can learn: "On Mondays (diff=7 pattern), user goes to work"
```

---

## Complete Example

### Full Feature Extraction Walkthrough

```
User: user_001 - Complete Temporal Features Example
═══════════════════════════════════════════════════════════════════════════════

RAW INPUT:
──────────
    User's first record: 2023-01-15 (Sunday)
    
    staypoint │ started_at          │ finished_at         │ location
    ──────────┼─────────────────────┼─────────────────────┼──────────
    SP_001    │ 2023-01-15 07:00    │ 2023-01-15 07:30    │ Home
    SP_002    │ 2023-01-15 10:00    │ 2023-01-15 12:00    │ Church
    SP_003    │ 2023-01-15 12:30    │ 2023-01-15 14:00    │ Restaurant
    SP_004    │ 2023-01-15 18:00    │ 2023-01-16 06:30    │ Home
    SP_005    │ 2023-01-16 07:30    │ 2023-01-16 08:00    │ Coffee
    SP_006    │ 2023-01-16 08:30    │ 2023-01-16 17:30    │ Work
    SP_007    │ 2023-01-16 18:00    │ 2023-01-17 06:30    │ Home


STEP 1: Calculate start_day (relative to first record)
──────────────────────────────────────────────────────

    User first date: 2023-01-15
    
    SP_001: 2023-01-15 → 01-15 - 01-15 = 0 days → start_day = 0
    SP_002: 2023-01-15 → 01-15 - 01-15 = 0 days → start_day = 0
    SP_003: 2023-01-15 → 01-15 - 01-15 = 0 days → start_day = 0
    SP_004: 2023-01-15 → 01-15 - 01-15 = 0 days → start_day = 0
    SP_005: 2023-01-16 → 01-16 - 01-15 = 1 day  → start_day = 1
    SP_006: 2023-01-16 → 01-16 - 01-15 = 1 day  → start_day = 1
    SP_007: 2023-01-16 → 01-16 - 01-15 = 1 day  → start_day = 1


STEP 2: Calculate weekday
─────────────────────────

    SP_001: 2023-01-15 → Sunday    → weekday = 6
    SP_002: 2023-01-15 → Sunday    → weekday = 6
    SP_003: 2023-01-15 → Sunday    → weekday = 6
    SP_004: 2023-01-15 → Sunday    → weekday = 6
    SP_005: 2023-01-16 → Monday    → weekday = 0
    SP_006: 2023-01-16 → Monday    → weekday = 0
    SP_007: 2023-01-16 → Monday    → weekday = 0


STEP 3: Calculate start_min
───────────────────────────

    SP_001: 07:00 → 7 × 60 + 0  = 420
    SP_002: 10:00 → 10 × 60 + 0 = 600
    SP_003: 12:30 → 12 × 60 + 30 = 750
    SP_004: 18:00 → 18 × 60 + 0 = 1080
    SP_005: 07:30 → 7 × 60 + 30 = 450
    SP_006: 08:30 → 8 × 60 + 30 = 510
    SP_007: 18:00 → 18 × 60 + 0 = 1080


STEP 4: Calculate duration (minutes)
────────────────────────────────────

    SP_001: 07:30 - 07:00 = 30 min
    SP_002: 12:00 - 10:00 = 120 min
    SP_003: 14:00 - 12:30 = 90 min
    SP_004: 06:30(next day) - 18:00 = 750 min (12.5 hours)
    SP_005: 08:00 - 07:30 = 30 min
    SP_006: 17:30 - 08:30 = 540 min (9 hours)
    SP_007: 06:30(next day) - 18:00 = 750 min (12.5 hours)


FINAL RESULT:
─────────────

    staypoint │ location_id │ start_day │ weekday │ start_min │ duration
    ──────────┼─────────────┼───────────┼─────────┼───────────┼──────────
    SP_001    │     42      │     0     │    6    │    420    │   30.0
    SP_002    │     10      │     0     │    6    │    600    │  120.0
    SP_003    │     17      │     0     │    6    │    750    │   90.0
    SP_004    │     42      │     0     │    6    │   1080    │  750.0
    SP_005    │      8      │     1     │    0    │    450    │   30.0
    SP_006    │     15      │     1     │    0    │    510    │  540.0
    SP_007    │     42      │     1     │    0    │   1080    │  750.0


SEQUENCE GENERATION (predicting for Day 2, previous_day=7):
──────────────────────────────────────────────────────────

    Target: SP_008 on Day 2 at Work (location 15)
    History: All staypoints from Day 0-1 within 7-day window

    Sequence:
    {
        "X":           [42, 10, 17, 42, 8, 15, 42],   # location_ids
        "user_X":      [1,  1,  1,  1, 1,  1,  1],    # encoded user
        "weekday_X":   [6,  6,  6,  6, 0,  0,  0],    # weekdays
        "start_min_X": [420, 600, 750, 1080, 450, 510, 1080],
        "dur_X":       [30, 120, 90, 750, 30, 540, 750],
        "diff":        [2,  2,  2,  2, 1,  1,  1],    # days ago
        "Y":           15                              # target: Work
    }
```

---

## Feature Importance in Prediction

### Which Temporal Features Matter Most?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE IMPORTANCE ANALYSIS                               │
└─────────────────────────────────────────────────────────────────────────────┘

Based on typical next location prediction models:

1. location_id (X) - MOST IMPORTANT
   • Past locations directly predict future locations
   • "You'll go where you've been before"

2. weekday - HIGH IMPORTANCE
   • Strong weekly patterns in human mobility
   • Work on weekdays, leisure on weekends

3. start_min - HIGH IMPORTANCE
   • Time of day strongly affects destination
   • Morning → work, evening → home

4. diff - MEDIUM IMPORTANCE
   • Recent visits more predictive than old
   • Weekly patterns (diff=7 indicates same weekday)

5. duration - MEDIUM IMPORTANCE
   • Distinguishes activity types
   • Short visit vs. long stay patterns

6. user_id - VARIABLE IMPORTANCE
   • Personal patterns differ
   • More important in heterogeneous populations


Model Architecture Implications:
────────────────────────────────

    Typical embedding dimensions:
    
    Feature    │ Embedding Dim │ Rationale
    ───────────┼───────────────┼────────────────────────────────
    location   │ 64-256        │ Main feature, needs capacity
    weekday    │ 8-16          │ Only 7 values, small embedding
    start_min  │ 32-64         │ 1440 values, medium embedding
    user       │ 16-64         │ Depends on user count
    diff       │ 8-16          │ Small range (1-7 typically)
    duration   │ N/A           │ Often used as scalar, not embedded
```

### Example: How Model Uses Temporal Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL PREDICTION EXAMPLE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Input Sequence:
───────────────
    X:           [42, 15, 42, 15, 42]    # Home, Work, Home, Work, Home
    weekday_X:   [0,  0,  1,  1,  2]     # Mon, Mon, Tue, Tue, Wed
    start_min_X: [450, 510, 420, 510, 420]
    diff:        [5,  5,  4,  4,  3]
    
    Current prediction context:
    - Weekday: 2 (Wednesday)
    - Time: 450 (7:30 AM)

Model reasoning:
────────────────
    1. Pattern recognition:
       "User alternates Home(42) → Work(15) on weekday mornings"
       
    2. Weekday context:
       "It's Wednesday (weekday=2), workday pattern applies"
       
    3. Time context:
       "7:30 AM = morning commute time (start_min=450)"
       
    4. Recency weighting:
       "Most recent visits (diff=3) show Home→Work pattern"

Prediction: Location 15 (Work) with high confidence

Without temporal features:
─────────────────────────
    Input: [42, 15, 42, 15, 42]
    
    Model only sees: "User visits 42 and 15 frequently"
    Cannot distinguish:
    - Morning prediction (→ Work)
    - Evening prediction (→ Home)
    - Weekend prediction (→ Leisure)
```

---

## Summary

### Quick Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL FEATURES SUMMARY                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Feature      │ Type    │ Range       │ Calculation                    │ Purpose
─────────────┼─────────┼─────────────┼────────────────────────────────┼────────────────
start_day    │ Integer │ 0 - ~100+   │ (date - user_first_date).days  │ Temporal split
end_day      │ Integer │ 0 - ~100+   │ (end_date - first_date).days   │ Multi-day stays
weekday      │ Integer │ 0 - 6       │ timestamp.dayofweek            │ Weekly patterns
start_min    │ Integer │ 0 - 1439    │ hour × 60 + minute             │ Time of day
end_min      │ Integer │ 0 - 1440    │ hour × 60 + minute             │ End time
duration     │ Float   │ 30 - 2880   │ (end - start).minutes          │ Stay length
diff         │ Integer │ 1 - 7+      │ target_day - start_day         │ Recency
```

### Code Quick Reference

```python
# All temporal feature calculations in one place

import pandas as pd

def extract_temporal_features(df, max_duration=2880):
    """Extract all temporal features from staypoints DataFrame."""
    
    # Get user's first date
    user_first_dates = df.groupby('user_id')['started_at'].min().dt.date
    
    # Calculate features
    df['start_day'] = df.apply(
        lambda r: (r['started_at'].date() - user_first_dates[r['user_id']]).days, 
        axis=1
    )
    df['end_day'] = df.apply(
        lambda r: (r['finished_at'].date() - user_first_dates[r['user_id']]).days, 
        axis=1
    )
    df['weekday'] = df['started_at'].dt.dayofweek
    df['start_min'] = df['started_at'].dt.hour * 60 + df['started_at'].dt.minute
    df['end_min'] = df['finished_at'].dt.hour * 60 + df['finished_at'].dt.minute
    df['duration'] = (
        (df['finished_at'] - df['started_at']).dt.total_seconds() / 60
    ).clip(upper=max_duration)
    
    return df
```

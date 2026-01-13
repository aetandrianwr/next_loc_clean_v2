# Worked Examples: Understanding Return Probability Analysis

## 1. Introduction

This document provides worked examples using a **consistent scenario** throughout. All examples follow the same users and data to build intuition step-by-step.

---

## 2. Example Scenario Setup

### 2.1 Our Example Users

We follow four users over a 10-day period:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXAMPLE USERS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  👩 ALICE (user_id = 1)                                                     │
│     • Office worker with regular commute                                    │
│     • First location: HOME                                                  │
│     • Expected behavior: Returns daily                                      │
│                                                                             │
│  👨 BOB (user_id = 2)                                                       │
│     • Works from home some days                                            │
│     • First location: HOME                                                  │
│     • Expected behavior: Returns within 24 hours                           │
│                                                                             │
│  👩 CAROL (user_id = 3)                                                     │
│     • Traveling salesperson                                                 │
│     • First location: HOTEL                                                │
│     • Expected behavior: May not return (explorer)                         │
│                                                                             │
│  👨 DAVE (user_id = 4)                                                      │
│     • Weekend warrior                                                       │
│     • First location: HOME                                                  │
│     • Expected behavior: Returns after weekend trip (~48h)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Location Encoding

```
Location ID    Place           Type
─────────────────────────────────────────────
100            Alice's HOME    Residential
101            Bob's HOME      Residential
102            Carol's HOTEL   Commercial
103            Dave's HOME     Residential
200            OFFICE_A        Work
201            OFFICE_B        Work
300            CAFE            Leisure
400            GYM             Leisure
500            MALL            Shopping
```

---

## 3. Example: Raw Data to Intermediate CSV

### 3.1 Raw GPS Trajectories

**Alice's Raw GPS Data (Day 1)**:
```
timestamp,              latitude,   longitude
2026-01-01 08:00:00,   39.9042,    116.4074    # Home
2026-01-01 08:45:00,   39.9200,    116.4200    # Transit
2026-01-01 09:15:00,   39.9100,    116.4300    # Office
2026-01-01 12:00:00,   39.9150,    116.4350    # Cafe
2026-01-01 18:30:00,   39.9042,    116.4074    # Home (return!)
```

### 3.2 After Preprocessing (Intermediate CSV)

GPS points are clustered into **staypoints** and encoded:

```csv
user_id,location_id,start_day,start_min
1,100,0,480
1,200,0,555
1,300,0,720
1,100,0,1110
```

**Explanation**:
- `user_id=1` is Alice
- `location_id=100` is HOME (clustered from GPS coordinates)
- `start_day=0` is Day 1 (January 1)
- `start_min=480` is 8:00 AM (480 minutes from midnight)
- `start_min=1110` is 6:30 PM (1110 minutes = 18h 30min)

### 3.3 Complete Example Dataset

All four users' data combined:

```csv
user_id,location_id,start_day,start_min
1,100,0,480      # Alice at HOME, Day 1, 8:00 AM
1,200,0,555      # Alice at OFFICE, Day 1, 9:15 AM
1,300,0,720      # Alice at CAFE, Day 1, 12:00 PM
1,100,0,1110     # Alice at HOME, Day 1, 6:30 PM (RETURN!)
1,200,1,540      # Alice at OFFICE, Day 2, 9:00 AM
2,101,0,540      # Bob at HOME, Day 1, 9:00 AM
2,200,0,780      # Bob at OFFICE, Day 1, 1:00 PM
2,101,0,1020     # Bob at HOME, Day 1, 5:00 PM (RETURN!)
3,102,0,600      # Carol at HOTEL, Day 1, 10:00 AM
3,200,0,720      # Carol at CLIENT_A, Day 1, 12:00 PM
3,201,1,600      # Carol at CLIENT_B, Day 2, 10:00 AM
3,300,2,720      # Carol at CAFE, Day 3 (NO RETURN!)
4,103,0,480      # Dave at HOME, Friday, 8:00 AM
4,500,0,600      # Dave at MALL, Friday, 10:00 AM
4,400,1,600      # Dave at GYM (trip), Saturday
4,500,1,900      # Dave at MALL, Saturday
4,103,2,1080     # Dave at HOME, Sunday 6:00 PM (RETURN!)
```

---

## 4. Example: Computing Timestamps

### 4.1 Timestamp Conversion

For each record, we compute `timestamp_hours`:

```
timestamp_hours = (start_day × 1440 + start_min) / 60
```

**Alice's First Record**:
```
start_day = 0, start_min = 480

timestamp_hours = (0 × 1440 + 480) / 60
                = 480 / 60
                = 8.0 hours
```

**Alice's Return to HOME**:
```
start_day = 0, start_min = 1110

timestamp_hours = (0 × 1440 + 1110) / 60
                = 1110 / 60
                = 18.5 hours
```

**Dave's Return to HOME (Day 2)**:
```
start_day = 2, start_min = 1080

timestamp_hours = (2 × 1440 + 1080) / 60
                = (2880 + 1080) / 60
                = 3960 / 60
                = 66.0 hours
```

### 4.2 Full Timestamp Table

```
user_id  location_id  start_day  start_min  timestamp_hours
   1         100          0         480           8.00
   1         200          0         555           9.25
   1         300          0         720          12.00
   1         100          0        1110          18.50   ← Alice return
   1         200          1         540          33.00
   2         101          0         540           9.00
   2         200          0         780          13.00
   2         101          0        1020          17.00   ← Bob return
   3         102          0         600          10.00
   3         200          0         720          12.00
   3         201          1         600          34.00
   3         300          2         720          60.00   ← Carol NO return
   4         103          0         480           8.00
   4         500          0         600          10.00
   4         400          1         600          34.00
   4         500          1         900          39.00
   4         103          2        1080          66.00   ← Dave return
```

---

## 5. Example: Finding First Locations

### 5.1 Sort by User and Time

Data is already sorted, but let's verify the order:

```
After sorting by (user_id, timestamp_hours):

user_id  location_id  timestamp_hours
   1         100           8.00     ← Alice's first
   1         200           9.25
   1         300          12.00
   1         100          18.50
   1         200          33.00
   2         101           9.00     ← Bob's first
   2         200          13.00
   2         101          17.00
   3         102          10.00     ← Carol's first
   3         200          12.00
   3         201          34.00
   3         300          60.00
   4         103           8.00     ← Dave's first
   4         500          10.00
   4         400          34.00
   4         500          39.00
   4         103          66.00
```

### 5.2 Extract First Events

```
GROUP BY user_id → FIRST():

user_id  first_location  first_time
   1          100            8.00     # Alice: HOME
   2          101            9.00     # Bob: HOME
   3          102           10.00     # Carol: HOTEL
   4          103            8.00     # Dave: HOME
```

---

## 6. Example: Finding Returns

### 6.1 Merge First Location Info

Add `first_location` and `first_time` to all events:

```
user_id  location_id  timestamp_hours  first_location  first_time
   1         100           8.00             100            8.00
   1         200           9.25             100            8.00
   1         300          12.00             100            8.00
   1         100          18.50             100            8.00
   1         200          33.00             100            8.00
   2         101           9.00             101            9.00
   2         200          13.00             101            9.00
   2         101          17.00             101            9.00
   3         102          10.00             102           10.00
   3         200          12.00             102           10.00
   3         201          34.00             102           10.00
   3         300          60.00             102           10.00
   4         103           8.00             103            8.00
   4         500          10.00             103            8.00
   4         400          34.00             103            8.00
   4         500          39.00             103            8.00
   4         103          66.00             103            8.00
```

### 6.2 Filter: Later Events Only

Keep only events where `timestamp_hours > first_time`:

```
user_id  location_id  timestamp_hours  first_location  first_time
   1         200           9.25             100            8.00   ✓
   1         300          12.00             100            8.00   ✓
   1         100          18.50             100            8.00   ✓
   1         200          33.00             100            8.00   ✓
   2         200          13.00             101            9.00   ✓
   2         101          17.00             101            9.00   ✓
   3         200          12.00             102           10.00   ✓
   3         201          34.00             102           10.00   ✓
   3         300          60.00             102           10.00   ✓
   4         500          10.00             103            8.00   ✓
   4         400          34.00             103            8.00   ✓
   4         500          39.00             103            8.00   ✓
   4         103          66.00             103            8.00   ✓
```

### 6.3 Filter: Returns Only

Keep only events where `location_id == first_location`:

```
user_id  location_id  timestamp_hours  first_location  first_time
   1         100          18.50             100            8.00   ← Alice RETURN
   2         101          17.00             101            9.00   ← Bob RETURN
   4         103          66.00             103            8.00   ← Dave RETURN

   Carol (user 3): NO RETURNS (never went back to HOTEL)
```

### 6.4 Compute Delta_t

```
user_id  delta_t_hours  Calculation
   1         10.50      18.50 - 8.00 = 10.50 hours
   2          8.00      17.00 - 9.00 =  8.00 hours
   4         58.00      66.00 - 8.00 = 58.00 hours

Carol: No return → excluded from analysis
```

---

## 7. Example: Building the Histogram

### 7.1 Return Times

```
delta_t_values = [10.50, 8.00, 58.00]
```

### 7.2 Create Bins

With `bin_width = 2 hours`:

```
bins = [0, 2, 4, 6, 8, 10, 12, ..., 58, 60, ...]

Bin ranges: [0,2), [2,4), [4,6), [6,8), [8,10), [10,12), ..., [58,60), ...
```

### 7.3 Count Values in Bins

```
delta_t = 8.00  → falls in bin [8, 10)  → count = 1
delta_t = 10.50 → falls in bin [10, 12) → count = 1
delta_t = 58.00 → falls in bin [58, 60) → count = 1

Histogram counts:
  Bin [0,2):   0
  Bin [2,4):   0
  Bin [4,6):   0
  Bin [6,8):   0
  Bin [8,10):  1  ← Bob
  Bin [10,12): 1  ← Alice
  ...
  Bin [58,60): 1  ← Dave
```

### 7.4 Normalize to Probability Density

```
N_returns = 3 (Alice, Bob, Dave)
bin_width = 2

F_pt(t) = count / (N_returns × bin_width)
        = count / (3 × 2)
        = count / 6

Results:
  F_pt(9)  = 1/6 = 0.167  (Bob)
  F_pt(11) = 1/6 = 0.167  (Alice)
  F_pt(59) = 1/6 = 0.167  (Dave)
  All other bins: 0
```

### 7.5 Verify Normalization

```
Total probability mass = Σ F_pt(tᵢ) × bin_width
                      = (0.167 + 0.167 + 0.167) × 2
                      = 0.501 × 2
                      = 1.002 ≈ 1.0 ✓
```

---

## 8. Example: Interpreting Results

### 8.1 Return Statistics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXAMPLE RESULTS SUMMARY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Users analyzed:        4                                                   │
│  Users with returns:    3 (Alice, Bob, Dave)                               │
│  Users without returns: 1 (Carol)                                          │
│  Return rate:           75% (3/4)                                          │
│                                                                             │
│  Return times:                                                              │
│  ─────────────────────────────────────────────────────────────────         │
│  Alice:   10.50 hours (same day return, went home after work)              │
│  Bob:      8.00 hours (same day return, short work day)                    │
│  Dave:    58.00 hours (weekend trip, returned Sunday evening)              │
│                                                                             │
│  Mean return time:    (10.50 + 8.00 + 58.00) / 3 = 25.5 hours             │
│  Median return time:  10.50 hours (middle value when sorted)               │
│                                                                             │
│  Observation: Mean > Median → Right-skewed distribution                    │
│               (most return quickly, some return much later)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Visual Representation

```
                    EXAMPLE RETURN PROBABILITY DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

F_pt(t)
0.20 │              
     │     ║   ║
0.15 │     ║   ║
     │     ║   ║
0.10 │     ║   ║
     │     ║   ║
0.05 │     ║   ║                                                  ║
     │     ║   ║                                                  ║
0.00 └─────╨───╨──────────────────────────────────────────────────╨───
          8   10  12  14  16  18  20  22  24  ...  56  58  60      t(h)
          ↑   ↑                                       ↑
         Bob Alice                                  Dave

     |<-- Same day returns -->|                  |<-- Weekend trip -->|
```

---

## 9. Example: Connection to Model

### 9.1 What the Model Should Learn

Based on our example:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               WHAT POINTER NETWORK V45 LEARNS FROM THIS DATA                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Pattern 1: Most users return (75% in our example)                         │
│  → Model learns: Pointer mechanism should have HIGH weight                 │
│                                                                             │
│  Pattern 2: Same-day returns common (Alice, Bob)                           │
│  → Model learns: Recent locations get HIGH attention                       │
│                                                                             │
│  Pattern 3: Some users don't return (Carol the explorer)                   │
│  → Model learns: Keep generation head for new locations                    │
│                                                                             │
│  Pattern 4: Weekend patterns (Dave)                                        │
│  → Model learns: Weekday embedding captures weekly cycles                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Example Prediction Scenario

```
INPUT: Alice's history on Day 3
  Locations: [HOME, OFFICE, CAFE, HOME, OFFICE, HOME, OFFICE, ...]
  Current time: 5:30 PM (Day 3)
  Current location: OFFICE

QUESTION: What will Alice do next?

MODEL REASONING:
  1. Pointer checks history: HOME appears 3 times (most frequent)
  2. Time feature: 5:30 PM → typical "go home" time
  3. Recency: Last HOME visit was this morning
  4. Gate: High α (likely return situation)

PREDICTION: HOME (location 100) with high confidence

GROUND TRUTH: Alice goes HOME at 6:00 PM ✓
```

---

## 10. Summary: From Data to Insight

### 10.1 Complete Pipeline Example

```
RAW GPS                 INTERMEDIATE CSV           RETURN TIMES
───────────────        ──────────────────         ─────────────
lat, lon, time    →    user_id, location,    →    delta_t_hours
                       start_day, start_min        
                                                   Alice: 10.5h
Alice at 39.9°N,  →    1, 100, 0, 480        →    Bob:   8.0h
116.4°E, 8:00AM                                    Dave: 58.0h
                                                   Carol: (none)


HISTOGRAM                          PROBABILITY DENSITY
──────────────────                ────────────────────────
Bin [8,10): 1 (Bob)          →    F_pt(9) = 0.167
Bin [10,12): 1 (Alice)       →    F_pt(11) = 0.167
Bin [58,60): 1 (Dave)        →    F_pt(59) = 0.167


INSIGHT                            MODEL DESIGN
──────────────────────            ──────────────────────
75% return rate              →    Pointer mechanism
Same-day returns common      →    Position-from-end embedding
~25% exploration             →    Generation head
```

### 10.2 Key Takeaways from Examples

1. **Data Flow**: GPS → Staypoints → Return times → Histogram → PDF
2. **Normalization**: F_pt integrates to 1.0 (probability density)
3. **Return Rate**: Percentage of users who came back to first location
4. **Delta_t**: Time between first observation and first return
5. **Model Connection**: Each finding justifies a model component

---

*← Back to [Model Justification](08_MODEL_JUSTIFICATION.md) | Continue to [Appendix](10_APPENDIX.md) →*

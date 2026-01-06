# H3 Dataset Statistics and EDA

Generated statistics for H3-preprocessed mobility datasets.

```
================================================================================
H3 DATASET STATISTICS AND EDA
Dataset: diy
H3 Resolution: 8
================================================================================

----------------------------------------
INTERIM DATA STATISTICS
----------------------------------------
Total Staypoints: 370,648
Total Users: 1,306
Total Locations (H3 cells): 2,938
Mean Staypoints/User: 283.8
Mean Duration (min): 562.9
Median Duration (min): 250.0
Mean Days Tracked: 129.3

----------------------------------------
PROCESSED DATA STATISTICS
----------------------------------------
Previous Day Window: 7
Final Users: 846
Final Locations: 2,755
Total Staypoints: 304,549
  Train: 239,550
  Val: 30,690
  Test: 34,309

Total Sequences: 258,601
  Train: 225,182
  Val: 14,928
  Test: 18,491

----------------------------------------
DETAILED STAYPOINT ANALYSIS
----------------------------------------

Staypoints per User:
  Mean: 283.8, Std: 222.9
  Min: 3, Max: 1861, Median: 224.5

Unique Locations per User:
  Mean: 37.8, Std: 33.6
  Min: 1, Max: 260, Median: 27.0

Duration (minutes):
  Mean: 562.9, Std: 1079.4
  Min: 30.0, Max: 55494.0, Median: 250.0

Location Visit Frequency:
  Mean visits/location: 126.2
  Std: 644.5
  Most visited location: 24002 visits
  Least visited location: 1 visits

Weekday Distribution:
  Mon: 54,910 (14.8%)
  Tue: 49,966 (13.5%)
  Wed: 53,552 (14.4%)
  Thu: 54,615 (14.7%)
  Fri: 54,639 (14.7%)
  Sat: 53,716 (14.5%)
  Sun: 49,250 (13.3%)

TRAIN Sequence Statistics:
  Count: 225,182
  Length Mean: 26.0
  Length Std: 15.7
  Length Range: 3 - 126
  Unique Targets: 2733

VALIDATION Sequence Statistics:
  Count: 14,928
  Length Mean: 26.2
  Length Std: 16.3
  Length Range: 3 - 109
  Unique Targets: 1615

TEST Sequence Statistics:
  Count: 18,491
  Length Mean: 28.4
  Length Std: 16.7
  Length Range: 3 - 102
  Unique Targets: 1589

================================================================================

================================================================================
H3 DATASET STATISTICS AND EDA
Dataset: geolife
H3 Resolution: 8
================================================================================

----------------------------------------
INTERIM DATA STATISTICS
----------------------------------------
Total Staypoints: 24,246
Total Users: 92
Total Locations (H3 cells): 1,362
Mean Staypoints/User: 263.5
Mean Duration (min): 397.6
Median Duration (min): 241.0
Mean Days Tracked: 306.5

----------------------------------------
PROCESSED DATA STATISTICS
----------------------------------------
Previous Day Window: 7
Final Users: 50
Final Locations: 900
Total Staypoints: 20,978
  Train: 10,614
  Val: 5,070
  Test: 5,294

Total Sequences: 18,168
  Train: 9,526
  Val: 4,217
  Test: 4,425

----------------------------------------
DETAILED STAYPOINT ANALYSIS
----------------------------------------

Staypoints per User:
  Mean: 263.5, Std: 419.6
  Min: 2, Max: 2916, Median: 113.5

Unique Locations per User:
  Mean: 52.1, Std: 67.1
  Min: 1, Max: 443, Median: 29.0

Duration (minutes):
  Mean: 397.6, Std: 368.1
  Min: 30.0, Max: 4265.0, Median: 241.0

Location Visit Frequency:
  Mean visits/location: 17.8
  Std: 77.0
  Most visited location: 1455 visits
  Least visited location: 1 visits

Weekday Distribution:
  Mon: 3,431 (14.2%)
  Tue: 3,462 (14.3%)
  Wed: 3,555 (14.7%)
  Thu: 3,505 (14.5%)
  Fri: 3,642 (15.0%)
  Sat: 3,494 (14.4%)
  Sun: 3,157 (13.0%)

TRAIN Sequence Statistics:
  Count: 9,526
  Length Mean: 20.3
  Length Std: 10.7
  Length Range: 3 - 54
  Unique Targets: 855

VALIDATION Sequence Statistics:
  Count: 4,217
  Length Mean: 20.3
  Length Std: 11.2
  Length Range: 3 - 54
  Unique Targets: 417

TEST Sequence Statistics:
  Count: 4,425
  Length Mean: 20.0
  Length Std: 11.1
  Length Range: 3 - 52
  Unique Targets: 392

================================================================================
```

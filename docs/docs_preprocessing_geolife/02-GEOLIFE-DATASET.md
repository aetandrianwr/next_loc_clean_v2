# GeoLife Dataset - Complete Reference Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Origin and Collection](#dataset-origin-and-collection)
3. [Raw Data Structure](#raw-data-structure)
4. [Data Format Specifications](#data-format-specifications)
5. [Sample Data Examples](#sample-data-examples)
6. [Data Characteristics](#data-characteristics)
7. [Why GeoLife for Next Location Prediction](#why-geolife-for-next-location-prediction)

---

## Introduction

The **GeoLife dataset** is one of the most widely used public GPS trajectory datasets in mobility research. It was collected by Microsoft Research Asia over a period of 5 years (April 2007 - August 2012) and contains GPS trajectories of 182 users in Beijing, China.

### Quick Facts

| Attribute | Value |
|-----------|-------|
| **Total Users** | 182 |
| **Total Trajectories** | 17,621 |
| **Total GPS Points** | ~24.9 million |
| **Collection Period** | April 2007 - August 2012 |
| **Collection Location** | Primarily Beijing, China |
| **Total Distance** | ~1,292,951 km |
| **Total Duration** | ~50,176 hours |
| **Average Sampling Rate** | 1-5 seconds (most dense) to minutes |

---

## Dataset Origin and Collection

### Collection Methodology

The GeoLife dataset was collected using various GPS devices carried by volunteers during their daily life activities. The participants included:

- Microsoft Research Asia employees
- University students
- Volunteers in Beijing

### Collection Devices

Different GPS logging devices were used, leading to varying sampling rates:
- Most trajectories: 1-5 second sampling interval
- Some trajectories: 1-2 minute intervals
- Various GPS loggers and smartphones

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION METHODOLOGY                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   [User carries GPS device] ────► [GPS records position] ────► [Data uploaded]  │
│                                                                                  │
│   Activities captured:                                                           │
│   ┌──────────────┬──────────────┬──────────────┬──────────────┐                 │
│   │     Work     │    Home      │   Shopping   │   Transit    │                 │
│   │   commute    │  activities  │    trips     │   (bus,      │                 │
│   │              │              │              │   subway)    │                 │
│   └──────────────┴──────────────┴──────────────┴──────────────┘                 │
│                                                                                  │
│   Sampling:                                                                      │
│   ●●●●●●●●●●●●●●●●●●●●  Dense sampling (1-5 seconds)                            │
│   ●    ●    ●    ●      Sparse sampling (1-2 minutes)                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Raw Data Structure

### Directory Layout

The raw GeoLife dataset follows a hierarchical folder structure:

```
raw_geolife/
└── Data/
    ├── 000/                          # User 000
    │   ├── Trajectory/               # GPS trajectory files
    │   │   ├── 20081023025304.plt    # Trajectory file
    │   │   ├── 20081024020959.plt
    │   │   ├── 20081026134407.plt
    │   │   └── ... (more .plt files)
    │   └── labels.txt                # Transportation mode labels (optional)
    │
    ├── 001/                          # User 001
    │   ├── Trajectory/
    │   │   └── ... 
    │   └── labels.txt
    │
    ├── 002/                          # User 002
    │   ├── Trajectory/
    │   │   └── ...
    │   └── labels.txt
    │
    └── ... (up to 181/)              # 182 total users (000-181)
```

### User ID Naming Convention

- User IDs are **three-digit strings** from `000` to `181`
- Total: 182 users
- Not all users have equal amount of data

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          USER DISTRIBUTION                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   User ID:  000   001   002   003   ...   100   101   ...   180   181           │
│                                                                                  │
│   Trajectories per user (varies significantly):                                  │
│   ████  ██  ████████  █  ...  ████████████  ████  ...  ██  ███                  │
│                                                                                  │
│   Some users have:                                                               │
│   - Dense tracking over many months                                              │
│   - Sparse tracking over few days                                                │
│                                                                                  │
│   Quality filtering is essential (covered in preprocessing)                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Format Specifications

### PLT File Format (Trajectory File)

Each `.plt` file contains a single trajectory (continuous GPS recording). The filename encodes the start time.

#### File Structure

```
Lines 1-6:    Header (metadata, skip when parsing)
Lines 7+:     GPS data points (one per line)
```

#### Header Lines (1-6)

```
Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,My Track,0,0,2,8421376
0
```

These lines contain metadata about the trajectory format and can be skipped during parsing.

#### Data Lines (7+)

Each data line has 7 comma-separated fields:

```
Latitude,Longitude,Reserved,Altitude,DateDays,Date,Time
```

| Field | Index | Description | Example |
|-------|-------|-------------|---------|
| Latitude | 0 | Decimal degrees (WGS84) | 39.984702 |
| Longitude | 1 | Decimal degrees (WGS84) | 116.318417 |
| Reserved | 2 | Always 0, unused | 0 |
| Altitude | 3 | Altitude in feet (can be -777 if invalid) | 492 |
| DateDays | 4 | Days since Dec 30, 1899 | 39744.1201157407 |
| Date | 5 | Date in YYYY-MM-DD format | 2008-10-23 |
| Time | 6 | Time in HH:MM:SS format | 02:53:04 |

#### Example PLT File Content

```
Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,My Track,0,0,2,8421376
0
39.984702,116.318417,0,492,39744.1201157407,2008-10-23,02:53:04
39.984683,116.318450,0,492,39744.1201388889,2008-10-23,02:53:06
39.984686,116.318417,0,492,39744.1201504630,2008-10-23,02:53:07
39.984688,116.318385,0,492,39744.1201620370,2008-10-23,02:53:08
39.984655,116.318263,0,492,39744.1201967593,2008-10-23,02:53:11
```

### Labels File Format (Transportation Mode)

Some users have a `labels.txt` file that provides transportation mode annotations.

#### Format

```
Start Time          End Time            Transportation Mode
2008/04/02 11:24:21,2008/04/02 11:50:45,bus
2008/04/03 01:07:03,2008/04/03 11:31:55,train
2008/04/03 11:32:18,2008/04/03 14:03:50,walk
```

| Field | Description | Example |
|-------|-------------|---------|
| Start Time | Start timestamp (YYYY/MM/DD HH:MM:SS) | 2008/04/02 11:24:21 |
| End Time | End timestamp | 2008/04/02 11:50:45 |
| Transportation Mode | Mode of transport | bus, train, walk, car, taxi, subway, bike |

**Note**: Not all users have labels, and labels are not used in the location prediction preprocessing.

---

## Sample Data Examples

### Example: Raw GPS Points from User 000

Let's trace through what the raw data looks like:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EXAMPLE: User 000, Trajectory 20081023025304                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  File: Data/000/Trajectory/20081023025304.plt                                   │
│                                                                                  │
│  Raw content (after header):                                                     │
│  ─────────────────────────────────────────────────────────────────────         │
│  39.984702,116.318417,0,492,39744.1201157407,2008-10-23,02:53:04               │
│  39.984683,116.318450,0,492,39744.1201388889,2008-10-23,02:53:06               │
│  39.984686,116.318417,0,492,39744.1201504630,2008-10-23,02:53:07               │
│  39.984688,116.318385,0,492,39744.1201620370,2008-10-23,02:53:08               │
│  39.984655,116.318263,0,492,39744.1201967593,2008-10-23,02:53:11               │
│                                                                                  │
│  Visualization (map approximation):                                              │
│  ─────────────────────────────────────────────────────────────────────         │
│                                                                                  │
│       N                                                                          │
│       ↑                                                                          │
│       │                                                                          │
│       │    ●──●    (GPS points very close together)                             │
│       │     ╲│                                                                   │
│       │      ●──●                                                                │
│       │         ╲                                                                │
│       │          ●                                                               │
│       └────────────────► E                                                       │
│                                                                                  │
│  Time span: 7 seconds (02:53:04 to 02:53:11)                                    │
│  Distance covered: ~16 meters                                                    │
│  Movement: User walking or slowly moving                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Example: Full Day of User Movement

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              EXAMPLE: Typical User Day (Conceptual Visualization)                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Timeline:                                                                       │
│  ───────────────────────────────────────────────────────────────────────────    │
│                                                                                  │
│  00:00     06:00     08:00     12:00     14:00     18:00     22:00    24:00     │
│    │─────────│─────────│─────────│─────────│─────────│─────────│─────────│      │
│              │         │         │         │         │                          │
│              │         │         │         │         │                          │
│    ┌─────────┐    ┌────┐    ┌────┐    ┌────┐    ┌────┐    ┌─────────┐          │
│    │  HOME   │    │Move│    │WORK│    │Move│    │WORK│    │  HOME   │          │
│    │(staying)│    │    │    │    │    │lunch│    │    │    │(staying)│          │
│    └─────────┘    └────┘    └────┘    └────┘    └────┘    └─────────┘          │
│                                                                                  │
│  GPS Data Density:                                                               │
│  ▓▓▓▓▓▓▓▓▓▓▓░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░▓▓▓▓▓▓▓▓▓         │
│  (dense at stops, sparse during movement)                                        │
│                                                                                  │
│  This day will generate:                                                         │
│  - Staypoints at HOME, WORK, LUNCH location                                      │
│  - Movement between locations (triplegs)                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Characteristics

### Geographic Distribution

The dataset is primarily collected in Beijing, China, with most trajectories concentrated in the Haidian District area near Microsoft Research Asia and Tsinghua University.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       GEOGRAPHIC DISTRIBUTION (Beijing)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                        Beijing City Map (Simplified)                             │
│                                                                                  │
│                    ┌────────────────────────────────────┐                       │
│                    │                                    │                       │
│                    │    ┌─────────────────┐            │                       │
│                    │    │   Haidian       │            │                       │
│                    │    │   District      │            │                       │
│                    │    │   ████████████  │ ← Most data│                       │
│                    │    │   ████████████  │   here     │                       │
│                    │    └─────────────────┘            │                       │
│                    │                                    │                       │
│                    │          ┌───────────┐            │                       │
│                    │          │  Central  │            │                       │
│                    │          │  Beijing  │            │                       │
│                    │          │   ████    │            │                       │
│                    │          └───────────┘            │                       │
│                    │                                    │                       │
│                    └────────────────────────────────────┘                       │
│                                                                                  │
│  Latitude range: ~39.6° to ~40.5° N                                             │
│  Longitude range: ~116.0° to ~117.0° E                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Temporal Distribution

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TEMPORAL DISTRIBUTION                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Data collection over time (2007-2012):                                         │
│                                                                                  │
│  2007    2008    2009    2010    2011    2012                                   │
│    │───────│───────│───────│───────│───────│                                    │
│         ████████████████████████████████████                                    │
│         │                            │                                          │
│         Most dense                   Less dense                                 │
│         collection                   collection                                 │
│                                                                                  │
│  User tracking duration varies:                                                  │
│  ─────────────────────────────────────────────────────────────────────          │
│  User A: ████████████████████████████████  (2+ years)                          │
│  User B: ████████  (few months)                                                 │
│  User C: ██████████████████  (1 year)                                           │
│  User D: ██  (few weeks)                                                        │
│                                                                                  │
│  This is why quality filtering is important!                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Quality Variations

The dataset has varying quality across users:

| Quality Aspect | Range | Notes |
|----------------|-------|-------|
| Tracking Duration | 1 day - 5 years | Highly variable |
| GPS Points per User | ~100 - 2M+ | Depends on device usage |
| Sampling Rate | 1 sec - 5 min | Different devices |
| Gaps in Data | None - Days | Device off periods |
| Geographic Coverage | Local - City-wide | Most concentrated in Haidian |

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DATA QUALITY OVERVIEW                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  User Quality Distribution (conceptual):                                         │
│                                                                                  │
│  Quality │                                                                       │
│  Score   │                                                                       │
│   High   │ ████                                                                  │
│          │ ████████                                                              │
│   Medium │ ████████████████████                                                  │
│          │ ████████████████████████████                                          │
│   Low    │ ████████████████████████████████████████                              │
│          └─────────────────────────────────────────────►                        │
│                        Number of Users                                           │
│                                                                                  │
│  Quality metrics include:                                                        │
│  • Number of tracking days                                                       │
│  • Tracking coverage (% of time with data)                                       │
│  • Consistency (regular vs sporadic tracking)                                    │
│                                                                                  │
│  Preprocessing filters keep only high-quality users:                             │
│  • Minimum 50 tracking days                                                      │
│  • Sliding window quality assessment                                             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Why GeoLife for Next Location Prediction

### Advantages

1. **Real-World Data**: Collected from actual daily activities, not synthetic
2. **Long-Term Tracking**: Up to 5 years of continuous data enables capturing routines
3. **Dense Sampling**: 1-5 second sampling provides accurate staypoint detection
4. **Public Availability**: Free to use for research purposes
5. **Rich Patterns**: Contains commuting, shopping, leisure, and other activities
6. **Benchmark Status**: Widely used in literature, enabling comparisons

### Challenges

1. **Quality Variation**: Not all users have equal data quality
2. **Geographic Bias**: Concentrated in Beijing
3. **Temporal Gaps**: Missing data periods
4. **Device Diversity**: Different sampling rates require preprocessing

### Academic References

The GeoLife dataset has been used in numerous research papers:

- **Original Paper**: "GeoLife: A Collaborative Social Networking Service among User, location and trajectory" (Zheng et al., 2010)
- **Mining Trajectories**: "Mining interesting locations and travel sequences from GPS trajectories" (Zheng et al., 2009)
- **Transportation Mode**: "Understanding transportation modes based on GPS data for web applications" (Zheng et al., 2008)

---

## Next Steps

Now that you understand the GeoLife dataset structure, proceed to:
- [03-CONFIGURATION.md](03-CONFIGURATION.md) - Learn about configuration parameters
- [04-SCRIPT1-RAW-TO-INTERIM.md](04-SCRIPT1-RAW-TO-INTERIM.md) - See how raw data is processed

---

*Documentation Version: 1.0*
*For PhD Research Reference*

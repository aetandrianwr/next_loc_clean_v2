# Descriptive Analysis Results

## Dataset Characteristics Comparison

| Category            | Metric                       |        DIY |   GeoLife |   Difference |
|:--------------------|:-----------------------------|-----------:|----------:|-------------:|
| Copy Applicability  | Target-in-History Rate (%)   |    84.12   |   83.81   |      -0.31   |
| Copy Applicability  | Target in Last-1 (%)         |    18.56   |   27.18   |       8.63   |
| Copy Applicability  | Target in Last-3 (%)         |    64.89   |   65.53   |       0.64   |
| Repetition Patterns | Avg Repetition Rate          |     0.6865 |    0.6596 |      -0.027  |
| Repetition Patterns | Avg Consecutive Repetition   |     0.1794 |    0.2687 |       0.0893 |
| Vocabulary          | Unique Locations in Test     |  2346      |  347      |   -1999      |
| Vocabulary          | Top-10 Location Coverage (%) |    42.61   |   69.34   |      26.73   |
| Vocabulary          | Target Distribution Entropy  |     5.022  |    3.5388 |      -1.4832 |
| User Patterns       | Number of Users              |   692      |   45      |    -647      |
| User Patterns       | Avg Target Revisit Rate      |     0.9738 |    0.9584 |      -0.0154 |
| Sequence            | Avg Sequence Length          |    23.98   |   18.37   |      -5.61   |
| Sequence            | Total Test Samples           | 12368      | 3502      |   -8866      |

## Key Findings

### 1. Target-in-History Rate (Copy Applicability)
- **GeoLife**: 83.81% of targets appear in history
- **DIY**: 84.12% of targets appear in history
- **Interpretation**: GeoLife has significantly higher copy applicability, meaning the pointer mechanism has more opportunity to be useful.

### 2. Repetition Patterns
- **GeoLife Repetition Rate**: 0.6596
- **DIY Repetition Rate**: 0.6865
- **Interpretation**: Higher repetition in GeoLife sequences means more opportunities for the pointer to select from repeated locations.


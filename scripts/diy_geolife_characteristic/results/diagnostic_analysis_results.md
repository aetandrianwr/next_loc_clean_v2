# Diagnostic Analysis Results

## Model-Based Analysis Summary

| Category           | Metric                    |       DIY |   GeoLife | Interpretation                                       |
|:-------------------|:--------------------------|----------:|----------:|:-----------------------------------------------------|
| Gate Behavior      | Mean Gate Value           |    0.7872 |    0.6267 | Higher = More Pointer Reliance                       |
| Gate Behavior      | Gate (Target in History)  |    0.803  |    0.6367 | Model should increase gate when target is in history |
| Component Accuracy | Pointer-Only Acc@1 (%)    |   56.53   |   51.63   | Pointer head performance alone                       |
| Component Accuracy | Generation-Only Acc@1 (%) |    5.64   |   12.19   | Generation head performance alone                    |
| Component Accuracy | Combined Acc@1 (%)        |   56.58   |   51.4    | Final combined performance                           |
| Component Accuracy | Pointer Advantage         |    0.4742 |    0.4366 | Avg(P_ptr - P_gen) on target                         |
| MRR Analysis       | Pointer MRR (%)           |   69.75   |   67.8    | Mean reciprocal rank of pointer                      |
| MRR Analysis       | Generation MRR (%)        |   10.09   |   18.73   | Mean reciprocal rank of generation                   |
| Vocabulary         | Unique Targets in Test    | 1713      |  315      | More targets = harder for generation                 |

## Key Insights

### 1. Pointer Performance Difference
- GeoLife pointer accuracy: 51.63%
- DIY pointer accuracy: 56.53%
- Difference: -4.91%

### 2. Generation Performance Difference
- GeoLife generation accuracy: 12.19%
- DIY generation accuracy: 5.64%
- Difference: 6.55%

### 3. Critical Insight: Why Pointer Matters More for GeoLife
The generation head performs significantly worse on GeoLife (12.19%) compared to DIY (5.64%). This makes the pointer mechanism relatively more important for GeoLife.

# Methodology and Experimental Design

## Comprehensive Documentation - Part 2

---

## Table of Contents

1. [Overview of Analysis Pipeline](#overview-of-analysis-pipeline)
2. [Data Sources](#data-sources)
3. [Model Architecture](#model-architecture)
4. [Analysis Scripts](#analysis-scripts)
5. [Visualization Standards](#visualization-standards)
6. [Reproducibility](#reproducibility)

---

## Overview of Analysis Pipeline

The analysis follows a systematic four-stage approach:

```
Stage 1: Descriptive Analysis
    ↓
Stage 2: Diagnostic Analysis (Model-Based)
    ↓
Stage 3: Hypothesis Testing
    ↓
Stage 4: Causal Experiments (Test Manipulation)
```

Each stage builds upon findings from the previous stage, progressively narrowing down the root cause of the differential ablation impact.

---

## Data Sources

### DIY Dataset

- **Full Name:** DIY Location Dataset
- **Preprocessing:** eps50 (50-meter clustering radius)
- **Sequence Configuration:** prev7 (7-day history window)
- **Test Set Size:** 12,368 samples
- **Number of Users:** 692
- **Train Vocabulary Size:** 7,017 locations
- **Test Unique Locations:** 2,346 locations
- **Geographic Context:** Urban mobility data

### GeoLife Dataset

- **Full Name:** GeoLife GPS Trajectory Dataset
- **Source:** Microsoft Research Asia
- **Preprocessing:** eps20 (20-meter clustering radius)
- **Sequence Configuration:** prev7 (7-day history window)
- **Test Set Size:** 3,502 samples
- **Number of Users:** 45
- **Train Vocabulary Size:** 1,156 locations
- **Test Unique Locations:** 347 locations
- **Geographic Context:** Beijing, China

### Data Format

Each sample contains:
```python
sample = {
    'X': np.array([...]),      # Input sequence (location IDs)
    'Y': int,                   # Target location ID
    'user_X': np.array([...]),  # User IDs
    'weekday_X': np.array([...]), # Weekday for each location
    'start_min_X': np.array([...]), # Start time (minutes)
    'dur_X': np.array([...]),   # Duration at each location
    'diff': np.array([...]),    # Time difference (recency)
}
```

### Data Paths

```
DIY Test:     data/diy_eps50/processed/diy_eps50_prev7_test.pk
DIY Train:    data/diy_eps50/processed/diy_eps50_prev7_train.pk
GeoLife Test: data/geolife_eps20/processed/geolife_eps20_prev7_test.pk
GeoLife Train: data/geolife_eps20/processed/geolife_eps20_prev7_train.pk
```

---

## Model Architecture

### PointerNetworkV45

The model is a transformer-based pointer-generator network with the following components:

#### 1. Input Embeddings
- **Location Embedding:** Maps location IDs to dense vectors
- **User Embedding:** Captures user-specific patterns
- **Temporal Embeddings:**
  - Time of day (15-minute buckets, 0-96)
  - Weekday (0-7)
  - Recency (days since visit, 0-8)
  - Duration (30-minute buckets, 0-99)
- **Position Embedding:** Position from end of sequence

#### 2. Transformer Encoder
- Multi-head self-attention layers
- Captures sequential dependencies and patterns
- Configuration varies by dataset (hyperparameter tuned)

#### 3. Pointer Mechanism
```python
# Query from context (last position)
query = self.pointer_query(context)  # [batch, d_model]

# Keys from encoded sequence
keys = self.pointer_key(encoded)  # [batch, seq_len, d_model]

# Attention scores with position bias
ptr_scores = (query @ keys.T) / sqrt(d_model) + position_bias
ptr_probs = softmax(ptr_scores)

# Scatter to vocabulary
ptr_dist.scatter_add_(input_locations, ptr_probs)
```

#### 4. Generation Head
```python
gen_logits = linear(context)  # [batch, num_locations]
gen_probs = softmax(gen_logits)
```

#### 5. Gate Mechanism
```python
gate = sigmoid(linear(context))  # [batch, 1], range [0, 1]
final_probs = gate * ptr_dist + (1 - gate) * gen_probs
```

### Model Checkpoints

| Dataset | Checkpoint Path |
|---------|-----------------|
| DIY | `experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt` |
| GeoLife | `experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt` |

### Configuration Files

| Dataset | Config Path |
|---------|-------------|
| DIY | `scripts/sci_hyperparam_tuning/configs/pointer_v45_diy_trial09.yaml` |
| GeoLife | `scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml` |

---

## Analysis Scripts

### Script 1: Descriptive Analysis (`01_descriptive_analysis.py`)

**Purpose:** Analyze dataset characteristics to identify fundamental differences.

**Key Analyses:**
1. **Target-in-History Rate:** Measures how often the target location appears in input sequence
2. **Repetition Patterns:** Measures location revisitation within sequences
3. **Vocabulary Utilization:** Analyzes unique locations and concentration
4. **User Patterns:** Examines per-user mobility diversity
5. **Sequence Characteristics:** Length distributions
6. **Temporal Patterns:** Time of day and weekday distributions

**Outputs:**
- `fig1_target_in_history.png/pdf`
- `fig2_repetition_patterns.png/pdf`
- `fig3_vocabulary_user_patterns.png/pdf`
- `fig4_radar_comparison.png/pdf`
- `descriptive_analysis_results.csv`
- `descriptive_analysis_results.md`
- `descriptive_analysis_full.json`

### Script 2: Diagnostic Analysis (`02_diagnostic_analysis.py`)

**Purpose:** Analyze trained model behavior to understand component-wise performance.

**Key Analyses:**
1. **Gate Value Analysis:** How the model weights pointer vs generation
2. **Component Accuracy:** Pointer-only, generation-only, and combined accuracy
3. **Pointer Attention Analysis:** Where does pointer focus in sequence?
4. **Per-sample Analysis:** When does pointer help vs hurt?

**Outputs:**
- `fig5_gate_analysis.png/pdf`
- `fig6_ptr_vs_gen.png/pdf`
- `fig7_vocabulary_effect.png/pdf`
- `diagnostic_analysis_results.csv`
- `diagnostic_analysis_results.md`
- `diagnostic_summary.json`

### Script 3: Hypothesis Testing (`03_hypothesis_testing.py`)

**Purpose:** Test specific hypotheses about root causes.

**Experiments:**
1. **Stratified Analysis:** Performance by target-in-history status
2. **Ablation Simulation:** Simulate pointer removal (gate=0)
3. **Generation Difficulty:** Analyze vocabulary size effects
4. **Root Cause Synthesis:** Connect all evidence

**Outputs:**
- `exp1_stratified_analysis.png/pdf`
- `exp2_ablation_simulation.png/pdf`
- `exp3_generation_difficulty.png/pdf`
- `fig_summary_root_cause.png`
- `exp4_root_cause_synthesis.csv`
- `hypothesis_testing_results.json`

### Script 4: Test Manipulation (`04_test_manipulation.py`)

**Purpose:** Causal experiments through test set manipulation.

**Experiments:**
1. **Target-in-History Ablation:** Evaluate on subsets where target is/isn't in history
2. **Recency Analysis:** Performance by target position (last 1, 2, 3, 5, 10)
3. **Pointer Necessity Proof:** Demonstrate equal applicability

**Outputs:**
- `exp5_target_in_history_ablation.png/pdf`
- `exp5_recency_effect.png/pdf`
- `final_summary.md`
- `final_summary_table.csv`
- `test_manipulation_results.json`

---

## Visualization Standards

### Publication Style Configuration (`publication_style.py`)

All figures follow a classic scientific publication style:

#### Visual Standards
- **Background:** White
- **Axis Box:** Black, all 4 sides visible
- **Tick Marks:** Inside, on all sides
- **Grid Lines:** None
- **Font:** Serif (Times New Roman style)
- **Line Width:** 1.5pt

#### Color Palette
| Element | Color |
|---------|-------|
| DIY | Blue (`#0000FF`) |
| GeoLife | Red (`#FF0000`) |
| Pointer | Green (`#00FF00`) |
| Generation | Purple (`#9b59b6`) |
| Combined | Orange (`#FFA500`) |

#### Marker Styles
| Dataset | Marker |
|---------|--------|
| DIY | Circle (○) |
| GeoLife | Square (□) |

#### Hatch Patterns (Bar Charts)
| Dataset | Hatch |
|---------|-------|
| DIY | `///` |
| GeoLife | `\\\` |

### Figure Naming Convention

```
fig{N}_{description}.{png|pdf}  # Main figures
exp{N}_{experiment_name}.{png|pdf}  # Experiment-specific figures
```

---

## Reproducibility

### Environment Setup

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

# Navigate to project root
cd /data/next_loc_clean_v2
```

### Running the Analysis

```bash
# Run all experiments in order
python scripts/diy_geolife_characteristic_v2/01_descriptive_analysis.py
python scripts/diy_geolife_characteristic_v2/02_diagnostic_analysis.py
python scripts/diy_geolife_characteristic_v2/03_hypothesis_testing.py
python scripts/diy_geolife_characteristic_v2/04_test_manipulation.py
```

### Dependencies

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm
- PyYAML

### Random Seed

All experiments use `SEED = 42` for reproducibility.

---

## Theoretical Framework

### Key Metrics Definitions

#### Target-in-History Rate
```
Rate = (Samples where Y ∈ X) / Total_Samples × 100%
```
Where Y is the target location and X is the input sequence.

#### Repetition Rate
```
Repetition_Rate = (len(X) - len(unique(X))) / len(X)
```
Measures how much of the sequence consists of repeated locations.

#### Gate Value
```
Gate ∈ [0, 1]
Higher gate → More pointer reliance
Lower gate → More generation reliance
```

#### Pointer Advantage
```
Pointer_Advantage = P_pointer(target) - P_generation(target)
```
Average probability difference on the correct target.

---

*This methodology ensures systematic, reproducible analysis with clear scientific standards.*

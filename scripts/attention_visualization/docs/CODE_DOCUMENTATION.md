# Code Documentation

## Technical Reference for Attention Visualization Scripts

This document provides detailed technical documentation for all code files in the attention visualization module.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [attention_extractor.py](#2-attention_extractorpy)
3. [run_attention_experiment.py](#3-run_attention_experimentpy)
4. [cross_dataset_comparison.py](#4-cross_dataset_comparisonpy)
5. [Dependencies and Requirements](#5-dependencies-and-requirements)
6. [Data Flow](#6-data-flow)
7. [Output Specification](#7-output-specification)

---

## 1. Module Overview

### 1.1 Purpose

The attention visualization module provides tools to:
- Extract attention weights from trained PointerNetworkV45 models
- Analyze attention patterns across samples and datasets
- Generate publication-quality visualizations
- Create statistical summaries and comparison tables

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Attention Visualization Module                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────┐    ┌─────────────────────────────┐  │
│  │  attention_extractor.py │    │  run_attention_experiment.py │  │
│  │  ─────────────────────  │    │  ──────────────────────────  │  │
│  │  • AttentionExtractor   │───▶│  • Model loading              │  │
│  │  • extract_batch_attn   │    │  • Experiment pipeline        │  │
│  │  • compute_statistics   │    │  • Visualization generation   │  │
│  └────────────────────────┘    │  • Table generation           │  │
│                                └─────────────────────────────┘  │
│                                              │                   │
│                                              ▼                   │
│                                ┌─────────────────────────────┐  │
│                                │ cross_dataset_comparison.py  │  │
│                                │ ──────────────────────────── │  │
│                                │ • Load experiment results    │  │
│                                │ • Generate comparison plots  │  │
│                                │ • Create summary tables      │  │
│                                └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. attention_extractor.py

### 2.1 Module Documentation

```python
"""
Attention Extraction Module for PointerNetworkV45.

This module provides comprehensive attention extraction capabilities for
the Pointer Network V45 model. It extracts multiple types of attention:

1. **Transformer Self-Attention**: Captures relationships between positions
   in the input sequence through multi-head self-attention.

2. **Pointer Attention**: The core mechanism that determines which historical
   locations to "copy" for the next prediction.

3. **Pointer-Generation Gate**: A scalar gate that balances between the
   pointer mechanism (copying from history) and generation (predicting from
   full vocabulary).

Scientific Significance:
- Self-attention reveals temporal dependencies in location sequences
- Pointer attention shows which historical visits influence predictions
- Gate values indicate when the model relies on repetitive vs. novel behavior
"""
```

### 2.2 Class: AttentionExtractor

```python
class AttentionExtractor:
    """
    Extracts and processes attention weights from PointerNetworkV45.
    
    Attributes:
        model (nn.Module): The PointerNetworkV45 model instance
        device (torch.device): Device for computation (cuda/cpu)
        hooks (List): List of registered forward hooks
        attention_weights (Dict): Captured attention weights
    
    Methods:
        register_hooks(): Register forward hooks on transformer layers
        clear_hooks(): Remove all registered hooks
        extract_attention(): Extract all attention components
        _compute_self_attention(): Manually compute self-attention weights
    """
```

#### 2.2.1 Constructor

```python
def __init__(self, model: nn.Module, device: torch.device):
    """
    Initialize the attention extractor.
    
    Args:
        model: PointerNetworkV45 model instance (must be on device)
        device: Torch device (cuda/cpu) for computations
    
    Example:
        model = PointerNetworkV45(...)
        model = model.to(device)
        extractor = AttentionExtractor(model, device)
    """
```

#### 2.2.2 extract_attention Method

```python
@torch.no_grad()
def extract_attention(
    self,
    x: torch.Tensor,
    x_dict: Dict[str, torch.Tensor],
    return_predictions: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extract all attention components from a forward pass.
    
    Args:
        x: Location sequence tensor [seq_len, batch_size]
        x_dict: Dictionary with temporal features:
            - 'user': User IDs [batch_size]
            - 'time': Time of day [seq_len, batch_size]
            - 'weekday': Day of week [seq_len, batch_size]
            - 'diff': Days ago [seq_len, batch_size]
            - 'duration': Visit duration [seq_len, batch_size]
            - 'len': Sequence lengths [batch_size]
        return_predictions: Whether to compute final predictions
        
    Returns:
        Dictionary containing:
            'self_attention': List[Tensor] - Per-layer self-attention
                Shape: [batch, num_heads, seq_len, seq_len] per layer
            'pointer_scores_raw': Tensor - Raw pointer scores before bias
                Shape: [batch, seq_len]
            'position_bias': Tensor - Learned position bias values
                Shape: [batch, seq_len]
            'pointer_probs': Tensor - Final pointer attention weights
                Shape: [batch, seq_len]
            'gate_values': Tensor - Pointer-generation gate outputs
                Shape: [batch, 1]
            'generation_probs': Tensor - Generation head probabilities
                Shape: [batch, num_locations]
            'predictions': Tensor - Log probabilities (if requested)
                Shape: [batch, num_locations]
            'final_probs': Tensor - Combined distribution
                Shape: [batch, num_locations]
            'pointer_distribution': Tensor - Pointer probs over locations
                Shape: [batch, num_locations]
            'input_sequence': Tensor - Input location sequence
                Shape: [batch, seq_len]
            'lengths': Tensor - Sequence lengths
                Shape: [batch]
            'mask': Tensor - Padding mask
                Shape: [batch, seq_len]
            'pos_from_end': Tensor - Position from end indices
                Shape: [batch, seq_len]
    """
```

#### 2.2.3 _compute_self_attention Method

```python
def _compute_self_attention(
    self,
    x: torch.Tensor,
    attn_module: nn.MultiheadAttention,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute self-attention weights manually.
    
    This method replicates PyTorch's MultiheadAttention computation
    to extract the intermediate attention weights.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        attn_module: PyTorch MultiheadAttention module
        mask: Padding mask [batch_size, seq_len]
        
    Returns:
        Attention weights [batch_size, num_heads, seq_len, seq_len]
    
    Implementation Details:
        1. Extract Q, K, V using in_proj weights
        2. Reshape for multi-head: [batch, heads, seq, head_dim]
        3. Compute scaled dot-product: QK^T / sqrt(d_k)
        4. Apply mask and softmax
        5. Handle NaN values from masked positions
    """
```

### 2.3 Function: extract_batch_attention

```python
def extract_batch_attention(
    extractor: AttentionExtractor,
    dataloader: DataLoader,
    num_samples: int = None,
    device: torch.device = None
) -> List[Dict]:
    """
    Extract attention for multiple samples from a dataloader.
    
    Args:
        extractor: Initialized AttentionExtractor instance
        dataloader: PyTorch DataLoader with test data
        num_samples: Maximum samples to extract (None for all)
        device: Torch device (uses extractor's device if None)
        
    Returns:
        List of dictionaries, one per sample, containing:
            'input_sequence': Tensor - Location sequence
            'length': int - Sequence length
            'pointer_probs': Tensor - Pointer attention
            'pointer_scores_raw': Tensor - Raw scores
            'position_bias': Tensor - Position bias
            'gate_value': float - Gate scalar
            'self_attention': List[Tensor] - Self-attention per layer
            'target': int - Target location ID
            'prediction': int - Predicted location ID
            'final_probs': Tensor - Final distribution
            'pointer_distribution': Tensor - Pointer distribution
            'generation_probs': Tensor - Generation distribution
    
    Example:
        results = extract_batch_attention(extractor, test_loader, num_samples=1000)
        print(f"Extracted {len(results)} samples")
    """
```

### 2.4 Function: compute_attention_statistics

```python
def compute_attention_statistics(
    attention_results: List[Dict]
) -> Dict[str, np.ndarray]:
    """
    Compute aggregate statistics over attention results.
    
    Args:
        attention_results: List of per-sample attention dictionaries
        
    Returns:
        Dictionary of statistics:
            'gate_mean': float - Mean gate value
            'gate_std': float - Gate standard deviation
            'gate_values': np.array - All gate values
            'pointer_entropy_mean': float - Mean pointer entropy
            'pointer_entropy_std': float - Entropy std dev
            'pointer_entropies': np.array - All entropy values
            'position_attention': np.array - Mean attention per position
            'position_counts': np.array - Sample count per position
            'accuracy': float - Prediction accuracy
            'correct_gate_mean': float - Gate mean for correct preds
            'incorrect_gate_mean': float - Gate mean for incorrect preds
    
    Computation Details:
        - Gate statistics: Direct mean/std over all samples
        - Entropy: -sum(p * log(p)) for each sample's pointer attention
        - Position attention: Averaged attention aligned to sequence end
        - Accuracy: Fraction where prediction == target
    """
```

---

## 3. run_attention_experiment.py

### 3.1 Module Documentation

```python
"""
Comprehensive Attention Visualization Experiment for PointerNetworkV45.

This experiment provides Nature Journal-standard scientific analysis of attention
mechanisms in the Pointer Network model for next location prediction.

Experiments Conducted:
======================
1. **Aggregate Attention Analysis**
   - Pointer attention distribution across sequence positions
   - Self-attention patterns in transformer layers
   - Pointer-Generation gate value distribution
   - Comparison between correct and incorrect predictions

2. **Sample-Level Analysis**
   - Top 10 samples with highest prediction confidence
   - Attention heatmaps for individual predictions
   - Position bias visualization
   - Multi-head attention decomposition

3. **Statistical Analysis**
   - Attention entropy analysis
   - Correlation between attention patterns and accuracy
   - Position bias effect quantification

Usage:
    python run_attention_experiment.py --dataset diy --seed 42
    python run_attention_experiment.py --dataset geolife --seed 42
"""
```

### 3.2 Configuration

```python
EXPERIMENT_CONFIGS = {
    'diy': {
        'experiment_dir': '/data/next_loc_clean_v2/experiments/diy_pointer_v45_...',
        'config_path': '/data/next_loc_clean_v2/scripts/.../pointer_v45_diy_trial09.yaml',
        'test_data': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_test.pk',
        'train_data': '/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_train.pk',
        'dataset_name': 'DIY',
        'full_name': 'DIY Check-in Dataset'
    },
    'geolife': {
        'experiment_dir': '/data/next_loc_clean_v2/experiments/geolife_pointer_v45_...',
        'config_path': '/data/next_loc_clean_v2/scripts/.../pointer_v45_geolife_trial01.yaml',
        'test_data': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_test.pk',
        'train_data': '/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_train.pk',
        'dataset_name': 'Geolife',
        'full_name': 'GeoLife GPS Trajectory Dataset'
    }
}
```

### 3.3 Function: load_model_and_data

```python
def load_model_and_data(
    config: Dict, 
    device: torch.device
) -> Tuple[nn.Module, DataLoader, Dict]:
    """
    Load trained model and test data.
    
    Args:
        config: Configuration dictionary with paths
        device: Torch device for model
        
    Returns:
        model: Loaded PointerNetworkV45 model in eval mode
        test_loader: DataLoader for test set (batch_size=64)
        info: Dictionary with:
            - num_locations: Total unique locations
            - num_users: Total unique users
            - max_seq_len: Maximum sequence length
            - test_size: Number of test samples
            - model_config: Model hyperparameters
    
    Implementation Notes:
        - Loads checkpoint from experiment_dir/checkpoints/best.pt
        - Infers max_seq_len from position_bias shape in checkpoint
        - Creates model with exact config used during training
        - Sets model to evaluation mode
    """
```

### 3.4 Function: select_best_samples

```python
def select_best_samples(
    attention_results: List[Dict],
    num_samples: int = 10
) -> Tuple[List[int], List[Dict]]:
    """
    Select best samples for visualization.
    
    Selection Criteria:
        1. Correct predictions only
        2. High prediction confidence (sorted by top probability)
        3. Diverse sequence lengths (first pass)
        4. Fill remaining with highest confidence (second pass)
    
    Args:
        attention_results: List of attention dictionaries
        num_samples: Number of samples to select
        
    Returns:
        indices: List of original indices in attention_results
        samples: List of selected sample dictionaries
    
    Algorithm:
        1. Filter to correct predictions only
        2. Sort by confidence (descending)
        3. First pass: Select diverse lengths from top 50
        4. Second pass: Fill with highest confidence
    """
```

### 3.5 Visualization Functions

#### plot_aggregate_pointer_attention

```python
def plot_aggregate_pointer_attention(
    attention_results: List[Dict],
    stats: Dict,
    output_dir: str,
    dataset_name: str
):
    """
    Create aggregate pointer attention visualization.
    
    Output: aggregate_pointer_attention.png/pdf
    
    Left Panel:
        - Bar chart of mean attention weight per position
        - X-axis: Position from sequence end (t-k)
        - Y-axis: Mean attention weight
        - Only shows positions with >= 10 samples
        - Annotation for most recent position
    
    Right Panel:
        - Histogram of attention entropy
        - X-axis: Entropy in nats
        - Y-axis: Number of samples
        - Red dashed line: Mean entropy
    """
```

#### plot_gate_analysis

```python
def plot_gate_analysis(
    attention_results: List[Dict],
    stats: Dict,
    output_dir: str,
    dataset_name: str
):
    """
    Analyze pointer-generation gate behavior.
    
    Output: gate_analysis.png/pdf
    
    Panel 1 (Left):
        - Histogram of gate value distribution
        - X-axis: Gate value [0, 1]
        - Y-axis: Count
        - Dashed line: Mean gate
    
    Panel 2 (Center):
        - Violin plot comparing correct vs incorrect
        - Green: Correct predictions
        - Red: Incorrect predictions
        - Shows mean and median
    
    Panel 3 (Right):
        - Gate value vs sequence length
        - X-axis: Sequence length
        - Y-axis: Mean gate value
        - Error bars: Standard deviation
        - Only lengths with >= 5 samples
    """
```

#### plot_self_attention_aggregate

```python
def plot_self_attention_aggregate(
    attention_results: List[Dict],
    output_dir: str,
    dataset_name: str,
    num_layers: int
):
    """
    Visualize aggregate self-attention patterns.
    
    Output: self_attention_aggregate.png/pdf
    
    Creates one heatmap per transformer layer:
        - X-axis: Key position (from end)
        - Y-axis: Query position (from end)
        - Color: Average attention weight
        - Max 15 positions shown
        - Head-averaged attention
        - Aligned to sequence end for comparison
    """
```

#### plot_sample_attention

```python
def plot_sample_attention(
    sample: Dict,
    sample_idx: int,
    output_dir: str,
    dataset_name: str,
    num_layers: int
):
    """
    Create detailed visualization for single sample.
    
    Output: sample_XX_attention.png/pdf
    
    Layout (GridSpec 3x3):
        Row 1, Col 0-1: Pointer attention bar chart
            - X-axis: Position (labeled with location IDs)
            - Y-axis: Attention weight
            - Color gradient: Yellow (low) to Red (high)
            - Black border on max attention position
            - Annotation: Gate value
        
        Row 1, Col 2: Score decomposition
            - Blue bars: Raw attention scores
            - Orange bars: Position bias
        
        Row 2, Col 0-2: Self-attention heatmaps
            - One per layer (up to 3)
            - Head-averaged
        
        Row 3, Col 0-2: Multi-head comparison
            - X-axis: Key position
            - Y-axis: Attention head
            - Shows head specialization
    """
```

#### plot_position_bias_analysis

```python
def plot_position_bias_analysis(
    model: nn.Module,
    output_dir: str,
    dataset_name: str,
    max_positions: int = 30
):
    """
    Analyze learned position bias parameter.
    
    Output: position_bias_analysis.png/pdf
    
    Left Panel:
        - Line plot of raw position bias values
        - X-axis: Position from end
        - Y-axis: Bias value
        - Horizontal line at 0
    
    Right Panel:
        - Bar chart showing attention distribution
        - Assumes equal base scores (isolates bias effect)
        - Shows softmax(position_bias) distribution
    """
```

### 3.6 Table Generation Functions

```python
def generate_statistics_table(
    stats: Dict,
    attention_results: List[Dict],
    output_dir: str,
    dataset_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate comprehensive statistics tables.
    
    Output Files:
        - attention_statistics.csv: Main metrics
        - attention_statistics.tex: LaTeX version
        - position_attention.csv: Position-wise attention
    
    Main Statistics Table Columns:
        - Metric: Name of statistic
        - Value: Computed value
    
    Position Attention Table Columns:
        - Position (from end): 0 to 14
        - Mean Attention: Average weight
        - Sample Count: Number of samples at that position
    """

def generate_sample_table(
    selected_samples: List[Dict],
    indices: List[int],
    output_dir: str,
    dataset_name: str
) -> pd.DataFrame:
    """
    Generate table for selected samples.
    
    Output Files:
        - selected_samples.csv
        - selected_samples.tex
    
    Columns:
        - Sample: Sample number (1-10)
        - Original Index: Index in full results
        - Sequence Length: Number of positions
        - Target Location: Ground truth
        - Prediction: Model prediction
        - Correct: Yes/No
        - Confidence: Probability of target
        - Gate Value: Pointer-gen gate
        - Max Pointer Attn: Maximum attention weight
    """
```

### 3.7 Main Experiment Function

```python
def run_experiment(dataset: str, seed: int = 42):
    """
    Run the complete attention visualization experiment.
    
    Pipeline:
        [1/6] Load model and test data
        [2/6] Extract attention for all test samples
        [3/6] Compute aggregate statistics
        [4/6] Select 10 best samples for visualization
        [5/6] Generate all visualizations
        [6/6] Generate statistical tables
    
    Output Structure:
        results/{dataset}/
        ├── aggregate_pointer_attention.png/pdf
        ├── attention_statistics.csv/tex
        ├── experiment_metadata.json
        ├── gate_analysis.png/pdf
        ├── position_attention.csv
        ├── position_bias_analysis.png/pdf
        ├── sample_01_attention.png/pdf
        ├── ... (10 sample files)
        ├── samples_overview.png/pdf
        ├── selected_samples.csv/tex
        └── self_attention_aggregate.png/pdf
    """
```

---

## 4. cross_dataset_comparison.py

### 4.1 Module Documentation

```python
"""
Cross-Dataset Comparison for Attention Visualization Experiment.

This script generates comparative visualizations and tables between
the DIY and Geolife datasets for the PointerNetworkV45 model.

Scientific Purpose:
===================
Compare attention mechanisms across datasets with different characteristics:
- DIY: Check-in based location data (urban mobility)
- Geolife: GPS trajectory data (continuous movement)

Output:
=======
- Comparative visualizations (PDF + PNG)
- Summary tables (CSV + LaTeX)
- Statistical comparison metrics
"""
```

### 4.2 Function: load_experiment_data

```python
def load_experiment_data(dataset: str) -> dict:
    """
    Load experiment results for a dataset.
    
    Args:
        dataset: 'diy' or 'geolife'
        
    Returns:
        Dictionary with:
            'metadata': JSON experiment metadata
            'statistics': DataFrame of attention statistics
            'position_attention': DataFrame of position-wise attention
            'selected_samples': DataFrame of selected sample info
    """
```

### 4.3 Function: create_comparison_table

```python
def create_comparison_table(
    diy_data: dict, 
    geolife_data: dict, 
    output_dir: Path
) -> pd.DataFrame:
    """
    Create comprehensive comparison table.
    
    Output Files:
        - cross_dataset_comparison.csv
        - cross_dataset_comparison.tex
    
    Metrics Compared:
        - Dataset name
        - Test Samples
        - Model Parameters
        - Prediction Accuracy (%)
        - Mean Gate Value
        - Gate Std Dev
        - Gate (Correct/Incorrect)
        - Mean Pointer Entropy
        - Most Recent Position Attention
        - Model architecture (d_model, nhead, num_layers)
    """
```

### 4.4 Function: plot_gate_comparison

```python
def plot_gate_comparison(
    diy_data: dict, 
    geolife_data: dict, 
    output_dir: Path
):
    """
    Create side-by-side gate value comparison.
    
    Output: cross_dataset_gate_comparison.png/pdf
    
    Left Panel:
        - Bar chart comparing mean gate values
        - Error bars: Standard deviation
        - Horizontal line at 0.5 (equal balance)
        - Annotations: Mean ± Std
    
    Right Panel:
        - Grouped bar chart of position attention
        - Green bars: DIY
        - Blue bars: Geolife
        - X-axis: Position from end (0-14)
    """
```

### 4.5 Function: plot_attention_pattern_comparison

```python
def plot_attention_pattern_comparison(
    diy_data: dict, 
    geolife_data: dict, 
    output_dir: Path
):
    """
    Compare attention patterns across datasets.
    
    Output: cross_dataset_attention_patterns.png/pdf
    
    Panel A (Top-Left): Recency Effect
        - Line plot of position attention
        - Shows decay from recent positions
    
    Panel B (Top-Right): Cumulative Attention
        - Cumulative sum for top-k positions
        - Shows attention concentration
    
    Panel C (Bottom-Left): Gate by Outcome
        - Grouped bars: Correct vs Incorrect
        - Compares gate differential
    
    Panel D (Bottom-Right): Summary Metrics
        - Normalized comparison (0-1 scale)
        - Metrics: Accuracy, Gate, Entropy, Recency
    """
```

### 4.6 Function: generate_summary_statistics

```python
def generate_summary_statistics(
    diy_data: dict, 
    geolife_data: dict, 
    output_dir: Path
) -> pd.DataFrame:
    """
    Generate key findings table.
    
    Output Files:
        - key_findings.csv
        - key_findings.tex
    
    Columns:
        - Finding: Description of finding
        - DIY: DIY dataset value
        - Geolife: Geolife dataset value
        - Interpretation: Scientific interpretation
    
    Findings:
        1. Higher pointer reliance
        2. Lower pointer entropy
        3. Stronger recency bias
        4. Better accuracy
        5. Gate differential
    """
```

---

## 5. Dependencies and Requirements

### 5.1 Python Dependencies

```python
# Core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Data Processing
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Utilities
import json
import yaml
import pickle
from pathlib import Path
from datetime import datetime
```

### 5.2 Internal Dependencies

```python
# Model
from src.models.proposed.pointer_v45 import PointerNetworkV45

# Training utilities
from src.training.train_pointer_v45 import NextLocationDataset, collate_fn, set_seed

# Attention extraction
from scripts.attention_visualization.attention_extractor import (
    AttentionExtractor,
    extract_batch_attention,
    compute_attention_statistics
)
```

### 5.3 Matplotlib Configuration

```python
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
```

---

## 6. Data Flow

### 6.1 Experiment Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          EXPERIMENT PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [Input]                                                                 │
│  ├── Trained Model Checkpoint (best.pt)                                  │
│  ├── Test Dataset (*.pk pickle files)                                    │
│  └── Model Configuration (*.yaml)                                        │
│                                                                          │
│  [Step 1: Load]                                                          │
│  load_model_and_data() ──▶ model, test_loader, info                     │
│                                                                          │
│  [Step 2: Extract]                                                       │
│  AttentionExtractor.extract_attention() ──▶ per-batch attention         │
│  extract_batch_attention() ──▶ all_results (List[Dict])                  │
│                                                                          │
│  [Step 3: Analyze]                                                       │
│  compute_attention_statistics() ──▶ stats (Dict)                        │
│  select_best_samples() ──▶ indices, samples                              │
│                                                                          │
│  [Step 4: Visualize]                                                     │
│  plot_aggregate_pointer_attention() ──▶ PNG/PDF                          │
│  plot_gate_analysis() ──▶ PNG/PDF                                        │
│  plot_self_attention_aggregate() ──▶ PNG/PDF                             │
│  plot_position_bias_analysis() ──▶ PNG/PDF                               │
│  plot_sample_attention() (×10) ──▶ PNG/PDF                               │
│  plot_combined_samples_overview() ──▶ PNG/PDF                            │
│                                                                          │
│  [Step 5: Export]                                                        │
│  generate_statistics_table() ──▶ CSV/TEX                                 │
│  generate_sample_table() ──▶ CSV/TEX                                     │
│  Save metadata ──▶ JSON                                                  │
│                                                                          │
│  [Output]                                                                │
│  └── results/{dataset}/*.png/*.pdf/*.csv/*.tex/*.json                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Attention Data Structure

```python
# Per-sample attention dictionary structure
sample_attention = {
    # Input data
    'input_sequence': torch.Tensor,      # [seq_len] location IDs
    'length': int,                        # Actual sequence length
    
    # Pointer attention
    'pointer_probs': torch.Tensor,        # [seq_len] attention weights
    'pointer_scores_raw': torch.Tensor,   # [seq_len] raw scores
    'position_bias': torch.Tensor,        # [seq_len] bias values
    
    # Self-attention
    'self_attention': List[torch.Tensor], # Per-layer [heads, seq, seq]
    
    # Gate and distributions
    'gate_value': float,                  # Pointer-gen gate scalar
    'final_probs': torch.Tensor,          # [num_locations] final dist
    'pointer_distribution': torch.Tensor, # [num_locations] ptr dist
    'generation_probs': torch.Tensor,     # [num_locations] gen dist
    
    # Prediction info
    'target': int,                        # Ground truth location
    'prediction': int,                    # Predicted location
}
```

---

## 7. Output Specification

### 7.1 Directory Structure

```
results/
├── diy/
│   ├── aggregate_pointer_attention.png
│   ├── aggregate_pointer_attention.pdf
│   ├── attention_statistics.csv
│   ├── attention_statistics.tex
│   ├── experiment_metadata.json
│   ├── gate_analysis.png
│   ├── gate_analysis.pdf
│   ├── position_attention.csv
│   ├── position_bias_analysis.png
│   ├── position_bias_analysis.pdf
│   ├── sample_01_attention.png
│   ├── sample_01_attention.pdf
│   ├── ... (samples 02-10)
│   ├── samples_overview.png
│   ├── samples_overview.pdf
│   ├── selected_samples.csv
│   ├── selected_samples.tex
│   ├── self_attention_aggregate.png
│   └── self_attention_aggregate.pdf
├── geolife/
│   └── ... (same structure)
├── cross_dataset_attention_patterns.png
├── cross_dataset_attention_patterns.pdf
├── cross_dataset_comparison.csv
├── cross_dataset_comparison.tex
├── cross_dataset_gate_comparison.png
├── cross_dataset_gate_comparison.pdf
├── key_findings.csv
└── key_findings.tex
```

### 7.2 File Formats

| Extension | Format | Purpose |
|-----------|--------|---------|
| .png | PNG Image | Web/presentation use (150 DPI) |
| .pdf | PDF Vector | Publication use (300 DPI) |
| .csv | CSV Table | Data analysis, spreadsheets |
| .tex | LaTeX Table | Direct inclusion in papers |
| .json | JSON | Metadata, programmatic access |

### 7.3 Metadata JSON Schema

```json
{
  "dataset": "diy|geolife",
  "dataset_name": "DIY|Geolife",
  "seed": 42,
  "num_samples": 12368,
  "accuracy": 0.5658,
  "gate_mean": 0.7872,
  "gate_std": 0.1366,
  "pointer_entropy_mean": 2.3358,
  "timestamp": "2026-01-02T10:52:11.815842",
  "model_config": {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.2
  },
  "num_layers": 2
}
```

---

## Usage Examples

### Running Single Dataset Experiment

```bash
# DIY dataset
python run_attention_experiment.py --dataset diy --seed 42

# Geolife dataset  
python run_attention_experiment.py --dataset geolife --seed 42
```

### Running Cross-Dataset Comparison

```bash
# Must run individual experiments first
python cross_dataset_comparison.py
```

### Programmatic Usage

```python
from attention_extractor import AttentionExtractor, compute_attention_statistics

# Load model
model = load_model(checkpoint_path)
model.eval()

# Create extractor
extractor = AttentionExtractor(model, device)

# Extract single sample
attention = extractor.extract_attention(x, x_dict)
print(f"Gate value: {attention['gate_values'].item():.4f}")
print(f"Pointer entropy: {compute_entropy(attention['pointer_probs']):.4f}")
```

---

*Code Documentation - Version 1.0*
*Last Updated: January 2026*

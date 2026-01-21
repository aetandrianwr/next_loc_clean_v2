# Code Walkthrough: Line-by-Line Analysis

This document provides a detailed walkthrough of each script in the gap_performance_diy_geolife_v2 framework.

---

## 1. analyze_mobility_patterns.py

### 1.1 Script Header and Imports (Lines 1-35)

```python
"""
Comprehensive Mobility Pattern Analysis for Gap Performance Study.

This script analyzes why the pointer mechanism has different impact on GeoLife (46.7%)
vs DIY (8.3%) datasets. The hypothesis is that GeoLife users exhibit more repetitive
mobility patterns compared to DIY users.
"""
```

**Purpose**: The docstring clearly states the research question and hypothesis.

**Key imports**:
- `numpy`: Array operations and statistical computations
- `pandas`: DataFrame creation for results tables
- `Counter` from collections: Efficient counting of location frequencies
- `scipy.stats`: Statistical tests (chi-square, Mann-Whitney U)
- `matplotlib`: Visualization

### 1.2 Matplotlib Configuration (Lines 37-97)

The script configures matplotlib for publication-quality figures:

```python
plt.rcParams.update({
    'font.family': 'serif',           # Academic style
    'axes.grid': False,               # No grid lines
    'axes.spines.top': True,          # Box around plot
    'xtick.direction': 'in',          # Ticks point inward
})
```

**Why these settings?**
- Serif fonts (Times New Roman) are standard in academic publications
- Inside ticks with all four spines creates a classic "box" style
- No grid keeps figures clean and focused on data

### 1.3 The MobilityPatternAnalyzer Class (Lines 133-843)

```python
class MobilityPatternAnalyzer:
    """Analyzes mobility patterns to explain pointer mechanism performance gap."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
```

**Design pattern**: The class encapsulates all analysis logic, creating output directories on initialization.

### 1.4 Target-in-History Analysis (Lines 152-200)

```python
def analyze_target_in_history(self, data: list, name: str) -> dict:
    target_in_history = []
    target_position_from_end = []
    target_frequency_in_history = []
    
    for sample in data:
        x = sample['X']  # Input sequence
        y = sample['Y']  # Target
        
        is_in_history = y in x  # Python 'in' operator
        target_in_history.append(is_in_history)
```

**Algorithm complexity**: O(n*m) where n=samples, m=sequence length
**Space complexity**: O(n) for storing results

**Key insight**: The `y in x` check is O(m) for each sample, but m is small (~20-25).

### 1.5 Entropy Calculation (Lines 264-270)

```python
def calculate_entropy(counts):
    """Calculate Shannon entropy from counts."""
    total = sum(counts)
    if total == 0:
        return 0
    probs = np.array([c / total for c in counts if c > 0])
    return -np.sum(probs * np.log2(probs))
```

**Mathematical basis**: Shannon entropy H = -Σ p(x) log₂ p(x)

**Numerical stability**: We filter out zero counts to avoid log(0).

### 1.6 Statistical Tests (Lines 432-483)

```python
def run_statistical_tests(self, diy_data: list, geolife_data: list) -> dict:
    # Chi-square test for categorical data
    contingency_table = [[diy_in, diy_not_in], [geolife_in, geolife_not_in]]
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Mann-Whitney U for continuous data
    u_stat, p_mannwhitney = stats.mannwhitneyu(
        diy_unique_ratios, geolife_unique_ratios, alternative='two-sided'
    )
    
    # Effect size calculation
    pooled_std = np.sqrt((np.std(diy)**2 + np.std(geolife)**2) / 2)
    cohens_d = (np.mean(diy) - np.mean(geolife)) / pooled_std
```

**Why these tests?**
- Chi-square: Tests independence of categorical variables (target in history: yes/no)
- Mann-Whitney U: Non-parametric test for comparing distributions
- Cohen's d: Measures effect size regardless of sample size

---

## 2. analyze_model_pointer.py

### 2.1 Custom Dataset Class (Lines 141-184)

```python
class NextLocationDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self._compute_statistics()
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        return_dict = {
            'user': torch.tensor(sample['user_X'][0], dtype=torch.long),
            'weekday': torch.tensor(sample['weekday_X'], dtype=torch.long),
            'time': torch.tensor(sample['start_min_X'] // 15, dtype=torch.long),
            'duration': torch.tensor(sample['dur_X'] // 30, dtype=torch.long),
            'diff': torch.tensor(sample['diff'], dtype=torch.long),
        }
        
        x = torch.tensor(sample['X'], dtype=torch.long)
        y = torch.tensor(sample['Y'], dtype=torch.long)
        
        return x, y, return_dict, sample  # Also return raw sample
```

**Design decision**: We return the raw sample alongside processed tensors for analysis purposes.

**Temporal encoding**:
- Time: Divided by 15 → 96 intervals per day (15-minute resolution)
- Duration: Divided by 30 → Duration buckets (30-minute resolution)

### 2.2 Extended Model with Analysis (Lines 217-295)

```python
class PointerGeneratorTransformerWithAnalysis(PointerGeneratorTransformer):
    def forward_with_analysis(self, x, x_dict):
        # ... standard forward pass ...
        
        # Extract intermediate values
        analysis = {
            'gate': gate.squeeze(-1).detach().cpu().numpy(),
            'ptr_probs': ptr_probs.detach().cpu().numpy(),
            'ptr_dist': ptr_dist.detach().cpu().numpy(),
            'gen_probs': gen_probs.detach().cpu().numpy(),
            'final_probs': final_probs.detach().cpu().numpy(),
        }
        
        return log_probs, analysis
```

**Key insight**: By extending the base model and overriding forward, we can extract internal states without modifying the original code.

### 2.3 Model Loading (Lines 298-328)

```python
def load_model(config_path, checkpoint_path, dataset_info, device):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Infer dimensions from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pos_bias_shape = checkpoint['model_state_dict']['position_bias'].shape[0]
    num_locations = checkpoint['model_state_dict']['loc_emb.weight'].shape[0]
    
    # Create and load model
    model = PointerGeneratorTransformerWithAnalysis(
        num_locations=num_locations,
        num_users=num_users,
        max_seq_len=pos_bias_shape,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
```

**Robustness**: We infer model dimensions from the checkpoint rather than relying on config files, ensuring compatibility.

### 2.4 Batch Analysis Loop (Lines 340-480)

```python
def analyze_model_behavior(self, model, dataloader, name):
    model.eval()
    with torch.no_grad():
        for x, y, x_dict, raw_samples in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            
            log_probs, analysis = model.forward_with_analysis(x, x_dict)
            
            predictions = log_probs.argmax(dim=-1).cpu().numpy()
            targets = y.cpu().numpy()
            
            for i in range(len(targets)):
                # Extract per-sample metrics
                gate_val = analysis['gate'][i]
                target = targets[i]
                pred = predictions[i]
                
                is_correct = pred == target
                # ... more analysis ...
```

**Memory management**: Using `torch.no_grad()` prevents gradient computation, saving memory.

**Device handling**: All tensors are moved to the correct device before forward pass.

---

## 3. analyze_recency_patterns.py

### 3.1 Position Calculation (Lines 147-218)

```python
def analyze_target_recency(self, data, name):
    for sample in data:
        x = sample['X']
        y = sample['Y']
        seq_len = len(x)
        
        positions = np.where(x == y)[0]  # All indices where target appears
        
        if len(positions) > 0:
            pos_from_end = seq_len - positions  # Convert to from-end
            most_recent_pos = min(pos_from_end)  # Smallest = most recent
            target_positions.append(most_recent_pos)
```

**Position encoding**:
```
Sequence:    [A, B, C, D, E]
Index:        0  1  2  3  4
Pos from end: 5  4  3  2  1
             ↑              ↑
          Oldest      Most recent
```

### 3.2 Return Pattern Detection (Lines 220-257)

```python
def analyze_return_patterns(self, data, name):
    for sample in data:
        x = sample['X']
        y = sample['Y']
        
        # A→B→A pattern: target equals location 2 steps back
        if len(x) >= 2:
            return_to_prev_prev.append(y == x[-2])
        
        # Return to any recent: target in last 5 locations
        recent_locs = x[-5:] if len(x) >= 5 else x
        return_to_any_recent.append(y in recent_locs)
```

**Pattern types detected**:
1. Bounce-back (A→B→A): User went somewhere and returned
2. Recent return: Target is anywhere in last 5 positions

### 3.3 Predictability Score (Lines 259-315)

```python
def analyze_location_predictability(self, data, name):
    for sample in data:
        x = sample['X']
        y = sample['Y']
        
        positions = np.where(x == y)[0]
        
        if len(positions) > 0:
            most_recent_pos = seq_len - positions[-1]
            
            # Recency score: higher for more recent
            recency = 1 / most_recent_pos
            
            # Frequency score: proportion of visits
            frequency = len(positions) / seq_len
            
            # Combined predictability
            predictability = recency * frequency
```

**Score interpretation**:
- Maximum predictability: target at position 1, appearing many times
- Minimum predictability: target far back, appearing once

---

## 4. Visualization Code Patterns

### 4.1 Classic Scientific Style

```python
def setup_classic_axes(ax):
    """Configure axes to match classic scientific publication style."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, bottom=True, left=True, right=True)
```

**All figures use this consistent style for professional appearance.**

### 4.2 Hatching for Bar Charts

```python
bars = ax.bar(categories, values, color='white', 
              edgecolor=colors, linewidth=1.5)
for bar, hatch, color in zip(bars, hatches, colors):
    bar.set_hatch(hatch)  # '///' for DIY, '...' for GeoLife
    bar.set_edgecolor(color)
```

**Why hatching?** Distinguishes categories without relying on color alone (accessibility).

### 4.3 Dual Format Saving

```python
fig.savefig(output_dir / 'figure.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'figure.pdf', bbox_inches='tight')
plt.close(fig)  # Release memory
```

**PNG**: Raster format for quick viewing
**PDF**: Vector format for publication/scaling

---

*Code Walkthrough Version: 1.0*

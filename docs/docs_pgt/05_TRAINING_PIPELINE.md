# Training Pipeline Documentation

This document covers the complete training pipeline for the Pointer Generator Transformer, including data loading, training loop, optimization, and experiment management.

---

## 1. Training Script Overview

### 1.1 File Location

```
src/training/train_pgt.py
```

### 1.2 Usage

```bash
# Train on GeoLife dataset
python src/training/train_pgt.py --config config/models/config_pgt_geolife.yaml

# Train on DIY dataset  
python src/training/train_pgt.py --config config/models/config_pgt_diy.yaml
```

### 1.3 Script Structure

```
train_pgt.py
├── Utility Functions
│   ├── set_seed()           - Reproducibility
│   ├── load_config()        - Configuration loading
│   └── EasyDict             - Attribute-style dict access
├── Data Loading
│   ├── NextLocationDataset  - Dataset class
│   ├── collate_fn()         - Batch collation
│   └── get_dataloaders()    - DataLoader creation
├── Trainer Class
│   └── TrainerV45           - Training orchestration
├── Experiment Management
│   ├── init_experiment_dir() - Directory setup
│   └── save_results()        - Result saving
└── Main Entry Point
    └── main()               - Script entry
```

---

## 2. Configuration Loading

### 2.1 YAML Configuration

```yaml
# config/models/config_pgt_geolife.yaml
seed: 42

data:
  data_dir: data/geolife_eps20/processed
  dataset_prefix: geolife_eps20_prev7
  dataset: geolife
  experiment_root: experiments
  num_workers: 0

model:
  d_model: 64
  nhead: 4
  num_layers: 2
  dim_feedforward: 128
  dropout: 0.15

training:
  batch_size: 128
  num_epochs: 50
  learning_rate: 0.00065
  weight_decay: 0.015
  label_smoothing: 0.03
  grad_clip: 0.8
  patience: 5
  min_epochs: 8
  warmup_epochs: 5
  use_amp: true
  min_lr: 0.000001
```

### 2.2 Configuration Loading Function

```python
def load_config(path: str) -> Dict:
    """
    Load configuration from YAML file and flatten it.
    
    Args:
        path: Path to YAML config file
    
    Returns:
        Flattened configuration dictionary
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Keep original structure for model and training
    config = {
        'model': cfg.get('model', {}),
        'training': cfg.get('training', {}),
        'data': cfg.get('data', {}),
    }
    
    # Add seed at top level
    config['seed'] = cfg.get('seed', 42)
    
    return config
```

---

## 3. Data Pipeline

### 3.1 NextLocationDataset Class

```python
class NextLocationDataset(Dataset):
    """
    Dataset for next location prediction.
    
    Loads preprocessed data from pickle files and provides:
    - Location sequence (X)
    - Target location (Y)
    - Temporal features: user, time, weekday, duration, diff
    """
    
    def __init__(self, data_path: str, build_user_history: bool = True):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.num_samples = len(self.data)
        self._compute_statistics()
        
        if build_user_history:
            self._build_user_history()
```

### 3.2 Data Sample Format

Each sample in the pickle file is a dictionary:

```python
{
    'X': np.array([2, 5, 3, 7, 5]),      # Location sequence
    'Y': 8,                               # Target location
    'user_X': np.array([1, 1, 1, 1, 1]), # User ID (repeated)
    'weekday_X': np.array([0, 0, 1, 2, 3]), # Weekday for each visit
    'start_min_X': np.array([540, 720, 480, 540, 660]), # Start time (minutes)
    'dur_X': np.array([120, 45, 480, 60, 30]),  # Duration (minutes)
    'diff': np.array([5, 4, 3, 2, 1]),   # Days ago
}
```

### 3.3 Data Preprocessing in __getitem__

```python
def __getitem__(self, idx):
    sample = self.data[idx]
    
    return_dict = {
        'user': torch.tensor(sample['user_X'][0], dtype=torch.long),
        'weekday': torch.tensor(sample['weekday_X'], dtype=torch.long),
        'time': torch.tensor(sample['start_min_X'] // 15, dtype=torch.long),  # 15-min buckets
        'duration': torch.tensor(sample['dur_X'] // 30, dtype=torch.long),    # 30-min buckets
        'diff': torch.tensor(sample['diff'], dtype=torch.long),
    }
    
    x = torch.tensor(sample['X'], dtype=torch.long)
    y = torch.tensor(sample['Y'], dtype=torch.long)
    
    return x, y, return_dict
```

### 3.4 Collate Function

Handles variable-length sequences through padding:

```python
def collate_fn(batch):
    """
    Collate function to handle variable length sequences.
    
    Returns:
        x_batch: Padded sequence tensor (seq_len, batch_size)
        y_batch: Target tensor (batch_size,)
        x_dict: Dictionary with additional features
    """
    x_batch, y_batch = [], []
    x_dict_batch = {'len': []}
    
    for x, y, return_dict in batch:
        x_batch.append(x)
        y_batch.append(y)
        x_dict_batch['len'].append(len(x))
        # ... collect other features
    
    # Pad sequences (padding_value=0)
    x_batch = pad_sequence(x_batch, batch_first=False, padding_value=0)
    y_batch = torch.stack(y_batch)
    
    # Convert lists to tensors
    x_dict_batch['user'] = torch.stack(x_dict_batch['user'])
    x_dict_batch['len'] = torch.tensor(x_dict_batch['len'], dtype=torch.long)
    
    # Pad variable length features
    for key in ['weekday', 'time', 'duration', 'diff']:
        x_dict_batch[key] = pad_sequence(x_dict_batch[key], batch_first=False, padding_value=0)
    
    return x_batch, y_batch, x_dict_batch
```

### 3.5 DataLoader Creation

```python
def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    # Load datasets
    train_ds = NextLocationDataset(train_path, build_user_history=True)
    val_ds = NextLocationDataset(val_path, build_user_history=False)
    test_ds = NextLocationDataset(test_path, build_user_history=False)
    
    # Share user history from training set
    val_ds.user_location_history = train_ds.user_location_history
    test_ds.user_location_history = train_ds.user_location_history
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    # ... similar for val and test
    
    return train_loader, val_loader, test_loader, info
```

---

## 4. Trainer Class (TrainerV45)

### 4.1 Initialization

```python
class TrainerV45:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device,
        experiment_dir: str,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        
        # Create directories
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

### 4.2 Loss Function

```python
# Cross-entropy with label smoothing
self.criterion = nn.CrossEntropyLoss(
    ignore_index=0,           # Ignore padding
    label_smoothing=0.03,     # Prevent overconfidence
)
```

**Label Smoothing**:
- Instead of hard targets [0, 0, 1, 0], use soft targets [0.01, 0.01, 0.97, 0.01]
- Prevents overconfident predictions
- Improves generalization

### 4.3 Optimizer Configuration

```python
self.optimizer = optim.AdamW(
    model.parameters(),
    lr=train_cfg.get('learning_rate', 3e-4),
    weight_decay=train_cfg.get('weight_decay', 0.015),
    betas=(0.9, 0.98),
    eps=1e-9,
)
```

**AdamW vs Adam**:
- AdamW properly decouples weight decay from gradient updates
- Better for Transformer training
- Standard choice for modern deep learning

### 4.4 Learning Rate Schedule

**Warmup + Cosine Decay**:

```python
def _get_lr(self, epoch: int) -> float:
    """Get learning rate with warmup and cosine decay."""
    if epoch < self.warmup_epochs:
        # Linear warmup
        return self.base_lr * (epoch + 1) / self.warmup_epochs
    
    # Cosine decay
    progress = (epoch - self.warmup_epochs) / max(1, self.num_epochs - self.warmup_epochs)
    return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
```

**Visualization of LR Schedule**:

```
LR
│
│    ╱‾‾‾‾╲
│   ╱      ╲
│  ╱        ╲
│ ╱          ╲____
│╱                ‾‾‾‾‾‾‾‾
└────────────────────────────── Epoch
  Warmup   Cosine Decay    Min LR
```

**Why Warmup?**
- Gradients are noisy at start
- Allows model to stabilize before aggressive updates
- Standard for Transformer training

**Why Cosine Decay?**
- Smooth decrease (no sharp drops)
- Continues learning at later epochs
- Better than step decay for fine-tuning

---

## 5. Training Loop

### 5.1 Single Epoch Training

```python
def train_epoch(self) -> float:
    """Train for one epoch."""
    self.model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
    
    for x, y, x_dict in pbar:
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        if self.scaler:  # Mixed precision
            with torch.cuda.amp.autocast():
                logits = self.model(x, x_dict)
                loss = self.criterion(logits, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(x, x_dict)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'avg': f"{total_loss/num_batches:.4f}"})
    
    return total_loss / num_batches
```

### 5.2 Mixed Precision Training (AMP)

```python
self.use_amp = train_cfg.get('use_amp', True)
self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
```

**Benefits**:
- ~2x faster training on modern GPUs
- Lower memory usage
- Minimal accuracy impact

**How it works**:
1. Forward pass in float16 (faster computation)
2. Loss scaling to prevent underflow
3. Backward pass with scaled gradients
4. Unscale gradients for optimizer
5. Optimizer step in float32 (for stability)

### 5.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
```

**Why?**
- Prevents gradient explosion
- Stabilizes training
- Especially important for Transformers

**Typical value**: 0.8

---

## 6. Evaluation

### 6.1 Evaluation Function

```python
@torch.no_grad()
def evaluate(self, loader: DataLoader, split: str = "val") -> Dict:
    """Evaluate on a dataset split."""
    self.model.eval()
    
    all_results = []
    all_true_y = []
    all_pred_y = []
    total_loss = 0.0
    num_batches = 0
    
    for x, y, x_dict in tqdm(loader, desc=f"Eval {split}"):
        x = x.to(self.device)
        y = y.to(self.device)
        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        
        if self.scaler:
            with torch.cuda.amp.autocast():
                logits = self.model(x, x_dict)
                loss = self.criterion(logits, y)
        else:
            logits = self.model(x, x_dict)
            loss = self.criterion(logits, y)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Calculate metrics
        results, true_y, pred_y = calculate_correct_total_prediction(logits, y)
        all_results.append(results)
        all_true_y.append(true_y)
        all_pred_y.append(pred_y)
    
    # Aggregate metrics
    total_results = np.sum(np.stack(all_results), axis=0)
    metrics = get_performance_dict({
        "correct@1": total_results[0],
        "correct@3": total_results[1],
        "correct@5": total_results[2],
        "correct@10": total_results[3],
        "rr": total_results[4],
        "ndcg": total_results[5],
        "total": total_results[6],
        "f1": f1_score(...)
    })
    metrics['loss'] = total_loss / num_batches
    
    return metrics
```

### 6.2 Metrics Computed

| Metric | Description |
|--------|-------------|
| `acc@1` | Top-1 accuracy (%) |
| `acc@5` | Top-5 accuracy (%) |
| `acc@10` | Top-10 accuracy (%) |
| `mrr` | Mean Reciprocal Rank (%) |
| `ndcg` | Normalized DCG@10 (%) |
| `f1` | Weighted F1 score (%) |
| `loss` | Average cross-entropy loss |

---

## 7. Full Training Loop

### 7.1 Main Training Function

```python
def train(self) -> Dict:
    """Full training loop."""
    self.logger.info(f"Training for {self.num_epochs} epochs")
    self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
    
    for epoch in range(self.num_epochs):
        self.current_epoch = epoch + 1
        
        # Set learning rate
        lr = self._get_lr(epoch)
        self._set_lr(lr)
        
        # Train one epoch
        train_loss = self.train_epoch()
        
        # Validate
        val_metrics = self.evaluate(self.val_loader, "val")
        
        self.logger.info(
            f"Epoch {self.current_epoch}/{self.num_epochs} | "
            f"LR: {lr:.2e} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"Acc@1: {val_metrics['acc@1']:.2f}%"
        )
        
        # Early stopping check
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.patience_counter = 0
            self._save_checkpoint("best.pt")
            self.logger.info(f"  ✓ New best (Acc@1: {val_metrics['acc@1']:.2f}%)")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience and self.current_epoch >= self.min_epochs:
                self.logger.info(f"Early stopping at epoch {self.current_epoch}")
                break
    
    # Final evaluation on best model
    self._load_checkpoint("best.pt")
    val_metrics = self.evaluate(self.val_loader, "val")
    test_metrics = self.evaluate(self.test_loader, "test")
    
    return val_metrics, test_metrics
```

### 7.2 Early Stopping

```python
# Parameters
self.patience = train_cfg.get('patience', 25)     # Epochs without improvement
self.min_epochs = train_cfg.get('min_epochs', 8)  # Minimum epochs before stopping

# Logic
if val_metrics['loss'] < self.best_val_loss:
    self.best_val_loss = val_metrics['loss']
    self.patience_counter = 0
    self._save_checkpoint("best.pt")
else:
    self.patience_counter += 1
    if self.patience_counter >= self.patience and self.current_epoch >= self.min_epochs:
        # Stop training
        break
```

**Why Early Stopping?**
- Prevents overfitting
- Saves training time
- Automatically finds optimal training duration

---

## 8. Checkpointing

### 8.1 Save Checkpoint

```python
def _save_checkpoint(self, filename: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': self.current_epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'best_val_loss': self.best_val_loss,
        'patience_counter': self.patience_counter,
        'config': self.config,
    }
    if self.scaler:
        checkpoint['scaler_state_dict'] = self.scaler.state_dict()
    
    torch.save(checkpoint, self.checkpoint_dir / filename)
```

### 8.2 Load Checkpoint

```python
def _load_checkpoint(self, filename: str):
    """Load model checkpoint."""
    checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if self.scaler and 'scaler_state_dict' in checkpoint:
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    # ... restore other state
```

---

## 9. Experiment Management

### 9.1 Directory Structure

```python
def init_experiment_dir(config: Dict, dataset_name: str, model_name: str = "pgt") -> str:
    """
    Create experiment directory with dataset name, model name, and timestamp.
    
    Format: experiments/{dataset_name}_{model_name}_{yyyyMMdd_hhmmss}/
    """
    experiment_root = config['data'].get('experiment_root', 'experiments')
    
    # Get current time in GMT+7
    gmt7 = timezone(timedelta(hours=7))
    now = datetime.now(gmt7)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{dataset_name}_{model_name}_{timestamp}"
    experiment_dir = os.path.join(experiment_root, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    return experiment_dir
```

### 9.2 Output Structure

```
experiments/geolife_pointer_v45_20260102_143000/
├── checkpoints/
│   └── best.pt              # Best model weights
├── config.yaml              # Configuration used
├── config_original.yaml     # Original config file
├── training.log             # Detailed training log
├── val_results.json         # Validation metrics
└── test_results.json        # Test metrics
```

### 9.3 Result Saving

```python
def save_results(experiment_dir: str, config: Dict, val_perf: Dict, test_perf: Dict, config_path: str):
    # Save configuration
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Copy original config
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(experiment_dir, "config_original.yaml"))
    
    # Save validation results
    val_results_path = os.path.join(experiment_dir, "val_results.json")
    with open(val_results_path, "w") as f:
        json.dump(val_results, f, indent=2)
    
    # Save test results
    test_results_path = os.path.join(experiment_dir, "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)
```

---

## 10. Reproducibility

### 10.1 Random Seed Setting

```python
def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 10.2 Deterministic Operations

```python
torch.backends.cudnn.deterministic = True   # Deterministic algorithms
torch.backends.cudnn.benchmark = False      # Disable auto-tuning
```

**Trade-off**: Slightly slower training for exact reproducibility

---

## 11. Logging

### 11.1 Setup

```python
def _setup_logging(self):
    """Setup logging."""
    log_file = self.experiment_dir / 'training.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    self.logger = logging.getLogger(__name__)
```

### 11.2 Log Output Example

```
2026-01-02 14:30:00 - INFO - ============================================================
2026-01-02 14:30:00 - INFO - POINTER GENERATOR TRANSFORMER - Training Started
2026-01-02 14:30:00 - INFO - ============================================================
2026-01-02 14:30:00 - INFO - Model config: {'d_model': 64, 'nhead': 4, ...}
2026-01-02 14:30:00 - INFO - Training config: {'batch_size': 128, 'num_epochs': 50, ...}
2026-01-02 14:30:00 - INFO - ============================================================
2026-01-02 14:30:05 - INFO - Training for 50 epochs
2026-01-02 14:30:05 - INFO - Model parameters: 180,000
2026-01-02 14:30:20 - INFO - Epoch 1/50 | LR: 1.30e-04 | Train: 5.2345 | Val: 4.8765 | Acc@1: 25.67%
2026-01-02 14:30:20 - INFO -   ✓ New best (Acc@1: 25.67%)
...
2026-01-02 14:45:00 - INFO - Early stopping at epoch 35
2026-01-02 14:45:10 - INFO - ============================================================
2026-01-02 14:45:10 - INFO - FINAL TEST RESULTS
2026-01-02 14:45:10 - INFO -   Acc@1:  53.97%
2026-01-02 14:45:10 - INFO -   Acc@5:  81.10%
2026-01-02 14:45:10 - INFO -   Acc@10: 84.38%
2026-01-02 14:45:10 - INFO -   MRR:    65.81%
2026-01-02 14:45:10 - INFO -   NDCG:   70.21%
2026-01-02 14:45:10 - INFO - ============================================================
```

---

## 12. Training Tips

### 12.1 Hyperparameter Recommendations

| Parameter | GeoLife | DIY | Guidance |
|-----------|---------|-----|----------|
| Learning Rate | 6.5e-4 | 7e-4 | Higher for smaller models |
| Batch Size | 128 | 128 | Larger for smoother gradients |
| Warmup Epochs | 5 | 5 | ~10% of total epochs |
| Patience | 5 | 5 | Short patience for quick experiments |
| Gradient Clip | 0.8 | 0.8 | Prevent gradient explosion |
| Label Smoothing | 0.03 | 0.03 | Slight regularization |

### 12.2 Common Issues

| Issue | Solution |
|-------|----------|
| Training too slow | Enable AMP (`use_amp: true`) |
| Out of memory | Reduce batch_size or d_model |
| Poor convergence | Increase learning rate |
| Overfitting | Increase dropout or reduce model size |
| NaN loss | Reduce learning rate, check data |

### 12.3 Monitoring Training

Watch for:
- **Train loss decreasing**: Model is learning
- **Val loss decreasing**: Model is generalizing
- **Val loss increasing while train decreases**: Overfitting
- **Both losses flat**: Learning rate too low or model converged

---

*Next: [06_EVALUATION_METRICS.md](06_EVALUATION_METRICS.md) - Evaluation Metrics Documentation*

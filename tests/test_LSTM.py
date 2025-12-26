#!/usr/bin/env python
"""
Test script for LSTM model and training pipeline.

This script validates:
1. Model can be instantiated correctly
2. Forward pass works with correct shapes
3. Data loading works correctly
4. Metrics calculation is correct
5. Parameter counts are within budget
6. Training script components work

Usage:
    cd /data/next_loc_clean_v2
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
    python tests/test_LSTM.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline.LSTM import LSTMModel, AllEmbeddingLSTM, TemporalEmbedding, FullyConnected
from src.evaluation.metrics import calculate_metrics, calculate_correct_total_prediction


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self, dataset='geolife'):
        if dataset == 'geolife':
            self.base_emb_size = 32
            self.lstm_hidden_size = 128
            self.lstm_num_layers = 2
            self.lstm_dropout = 0.2
            self.total_user_num = 46
            self.total_loc_num = 1187
        else:  # diy
            self.base_emb_size = 96
            self.lstm_hidden_size = 192
            self.lstm_num_layers = 2
            self.lstm_dropout = 0.2
            self.total_user_num = 693
            self.total_loc_num = 7038
        
        self.if_embed_user = True
        self.if_embed_time = True
        self.if_embed_duration = True
        self.if_embed_poi = False
        self.fc_dropout = 0.2
        self.poi_original_size = 16


def test_temporal_embedding():
    """Test TemporalEmbedding module."""
    print("Testing TemporalEmbedding...")
    
    d_input = 32
    te = TemporalEmbedding(d_input, emb_info="all")
    
    # Test forward pass
    seq_len, batch_size = 10, 4
    time = torch.randint(0, 96, (seq_len, batch_size))  # 96 = 24*4 time slots
    weekday = torch.randint(0, 7, (seq_len, batch_size))
    
    out = te(time, weekday)
    assert out.shape == (seq_len, batch_size, d_input), f"Expected shape {(seq_len, batch_size, d_input)}, got {out.shape}"
    print("  ✓ TemporalEmbedding output shape correct")
    print("  ✓ TemporalEmbedding tests passed\n")


def test_all_embedding_lstm():
    """Test AllEmbeddingLSTM module."""
    print("Testing AllEmbeddingLSTM...")
    
    config = MockConfig('geolife')
    total_loc_num = 100
    
    emb = AllEmbeddingLSTM(config.base_emb_size, config, total_loc_num)
    
    # Test forward pass
    seq_len, batch_size = 10, 4
    src = torch.randint(1, total_loc_num, (seq_len, batch_size))
    context_dict = {
        'time': torch.randint(0, 96, (seq_len, batch_size)),
        'weekday': torch.randint(0, 7, (seq_len, batch_size)),
        'duration': torch.randint(0, 96, (seq_len, batch_size)),
    }
    
    out = emb(src, context_dict)
    assert out.shape == (seq_len, batch_size, config.base_emb_size)
    print("  ✓ AllEmbeddingLSTM output shape correct")
    print("  ✓ AllEmbeddingLSTM tests passed\n")


def test_lstm_model_geolife():
    """Test LSTM model with GeoLife configuration."""
    print("Testing LSTM model (GeoLife config)...")
    
    config = MockConfig('geolife')
    total_loc_num = config.total_loc_num
    
    model = LSTMModel(config, total_loc_num)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    
    # Check parameter budget
    assert total_params < 500000, f"GeoLife model exceeds 500K parameters: {total_params:,}"
    print("  ✓ Parameter count within budget (<500K)")
    
    # Test forward pass
    seq_len, batch_size = 10, 4
    device = torch.device('cpu')
    
    src = torch.randint(1, total_loc_num, (seq_len, batch_size))
    context_dict = {
        'len': torch.tensor([8, 10, 6, 9]),  # Variable lengths
        'user': torch.randint(0, config.total_user_num, (batch_size,)),
        'time': torch.randint(0, 96, (seq_len, batch_size)),
        'weekday': torch.randint(0, 7, (seq_len, batch_size)),
        'duration': torch.randint(0, 96, (seq_len, batch_size)),
        'diff': torch.randint(0, 7, (seq_len, batch_size)),
    }
    
    model.eval()
    with torch.no_grad():
        logits = model(src, context_dict, device)
    
    assert logits.shape == (batch_size, total_loc_num), f"Expected shape {(batch_size, total_loc_num)}, got {logits.shape}"
    print("  ✓ LSTM forward pass shape correct")
    print("  ✓ LSTM model (GeoLife) tests passed\n")


def test_lstm_model_diy():
    """Test LSTM model with DIY configuration."""
    print("Testing LSTM model (DIY config)...")
    
    config = MockConfig('diy')
    total_loc_num = config.total_loc_num
    
    model = LSTMModel(config, total_loc_num)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    
    # Check parameter budget
    assert total_params < 3000000, f"DIY model exceeds 3M parameters: {total_params:,}"
    print("  ✓ Parameter count within budget (<3M)")
    
    # Test forward pass
    seq_len, batch_size = 10, 4
    device = torch.device('cpu')
    
    src = torch.randint(1, total_loc_num, (seq_len, batch_size))
    context_dict = {
        'len': torch.tensor([8, 10, 6, 9]),
        'user': torch.randint(0, config.total_user_num, (batch_size,)),
        'time': torch.randint(0, 96, (seq_len, batch_size)),
        'weekday': torch.randint(0, 7, (seq_len, batch_size)),
        'duration': torch.randint(0, 96, (seq_len, batch_size)),
        'diff': torch.randint(0, 7, (seq_len, batch_size)),
    }
    
    model.eval()
    with torch.no_grad():
        logits = model(src, context_dict, device)
    
    assert logits.shape == (batch_size, total_loc_num)
    print("  ✓ LSTM forward pass shape correct")
    print("  ✓ LSTM model (DIY) tests passed\n")


def test_metrics():
    """Test evaluation metrics."""
    print("Testing evaluation metrics...")
    
    batch_size = 32
    num_classes = 100
    
    # Create random predictions and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test calculate_correct_total_prediction
    result_arr, true_y, top1 = calculate_correct_total_prediction(logits, targets)
    
    assert len(result_arr) == 7, "Result array should have 7 elements"
    assert result_arr[-1] == batch_size, "Total should equal batch size"
    print("  ✓ calculate_correct_total_prediction works")
    
    # Test calculate_metrics
    metrics = calculate_metrics(logits, targets)
    
    required_keys = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1', 'total']
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"
    
    # Sanity checks
    assert 0 <= metrics['acc@1'] <= 100, "Acc@1 should be percentage"
    assert 0 <= metrics['acc@5'] <= 100, "Acc@5 should be percentage"
    assert metrics['acc@1'] <= metrics['acc@5'] <= metrics['acc@10'], "Higher k should give better accuracy"
    print("  ✓ calculate_metrics returns correct format")
    print("  ✓ Metrics sanity checks passed")
    
    print("  ✓ Metrics tests passed\n")


def test_data_loading():
    """Test data loading from preprocessed files."""
    print("Testing data loading...")
    
    import pickle
    
    # Test GeoLife data
    geolife_path = "/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_train.pk"
    if os.path.exists(geolife_path):
        data = pickle.load(open(geolife_path, 'rb'))
        
        assert len(data) > 0, "Data should not be empty"
        sample = data[0]
        
        required_keys = ['X', 'Y', 'user_X', 'weekday_X', 'start_min_X', 'dur_X', 'diff']
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
        
        print(f"  ✓ GeoLife data loaded: {len(data)} samples")
    else:
        print("  ⚠ GeoLife data not found, skipping")
    
    # Test DIY data
    diy_path = "/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_train.pk"
    if os.path.exists(diy_path):
        data = pickle.load(open(diy_path, 'rb'))
        print(f"  ✓ DIY data loaded: {len(data)} samples")
    else:
        print("  ⚠ DIY data not found, skipping")
    
    print("  ✓ Data loading tests passed\n")


def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    import yaml
    
    geolife_config_path = "/data/next_loc_clean_v2/config/models/config_LSTM_geolife.yaml"
    if os.path.exists(geolife_config_path):
        with open(geolife_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'data' in config, "Config should have 'data' section"
        assert 'model' in config, "Config should have 'model' section"
        assert 'embedding' in config, "Config should have 'embedding' section"
        
        # Check LSTM specific parameters
        assert 'lstm_hidden_size' in config['model'], "Config should have lstm_hidden_size"
        assert 'lstm_num_layers' in config['model'], "Config should have lstm_num_layers"
        
        print(f"  ✓ GeoLife LSTM config loaded successfully")
    else:
        print("  ⚠ GeoLife LSTM config not found")
    
    diy_config_path = "/data/next_loc_clean_v2/config/models/config_LSTM_diy.yaml"
    if os.path.exists(diy_config_path):
        with open(diy_config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ DIY LSTM config loaded successfully")
    else:
        print("  ⚠ DIY LSTM config not found")
    
    print("  ✓ Config loading tests passed\n")


def test_variable_sequence_length():
    """Test model with variable sequence lengths."""
    print("Testing variable sequence lengths...")
    
    config = MockConfig('geolife')
    total_loc_num = 100
    model = LSTMModel(config, total_loc_num)
    device = torch.device('cpu')
    model.eval()
    
    # Test with different sequence lengths
    for seq_len in [5, 10, 20, 50]:
        batch_size = 4
        src = torch.randint(1, total_loc_num, (seq_len, batch_size))
        # Variable lengths within batch
        lengths = [min(seq_len, l) for l in [3, seq_len, seq_len-2, seq_len-1]]
        context_dict = {
            'len': torch.tensor(lengths),
            'user': torch.randint(0, config.total_user_num, (batch_size,)),
            'time': torch.randint(0, 96, (seq_len, batch_size)),
            'weekday': torch.randint(0, 7, (seq_len, batch_size)),
            'duration': torch.randint(0, 96, (seq_len, batch_size)),
            'diff': torch.randint(0, 7, (seq_len, batch_size)),
        }
        
        with torch.no_grad():
            logits = model(src, context_dict, device)
        
        assert logits.shape == (batch_size, total_loc_num)
        print(f"  ✓ Sequence length {seq_len} works")
    
    print("  ✓ Variable sequence length tests passed\n")


def test_packed_sequence():
    """Test that packed sequences work correctly with LSTM."""
    print("Testing packed sequence handling...")
    
    config = MockConfig('geolife')
    total_loc_num = 100
    model = LSTMModel(config, total_loc_num)
    device = torch.device('cpu')
    model.eval()
    
    # Test with variable lengths in a batch
    seq_len, batch_size = 15, 4
    src = torch.randint(1, total_loc_num, (seq_len, batch_size))
    
    # Different lengths for each sample in batch
    lengths = [3, 15, 7, 10]
    context_dict = {
        'len': torch.tensor(lengths),
        'user': torch.randint(0, config.total_user_num, (batch_size,)),
        'time': torch.randint(0, 96, (seq_len, batch_size)),
        'weekday': torch.randint(0, 7, (seq_len, batch_size)),
        'duration': torch.randint(0, 96, (seq_len, batch_size)),
        'diff': torch.randint(0, 7, (seq_len, batch_size)),
    }
    
    with torch.no_grad():
        logits = model(src, context_dict, device)
    
    assert logits.shape == (batch_size, total_loc_num)
    assert not torch.isnan(logits).any(), "Output should not contain NaN"
    print("  ✓ Packed sequence handling works")
    print("  ✓ Packed sequence tests passed\n")


def test_gpu_if_available():
    """Test model on GPU if available."""
    print("Testing GPU support...")
    
    if torch.cuda.is_available():
        config = MockConfig('geolife')
        total_loc_num = 100
        device = torch.device('cuda')
        
        model = LSTMModel(config, total_loc_num).to(device)
        
        seq_len, batch_size = 10, 4
        src = torch.randint(1, total_loc_num, (seq_len, batch_size)).to(device)
        context_dict = {
            'len': torch.tensor([seq_len] * batch_size).to(device),
            'user': torch.randint(0, config.total_user_num, (batch_size,)).to(device),
            'time': torch.randint(0, 96, (seq_len, batch_size)).to(device),
            'weekday': torch.randint(0, 7, (seq_len, batch_size)).to(device),
            'duration': torch.randint(0, 96, (seq_len, batch_size)).to(device),
            'diff': torch.randint(0, 7, (seq_len, batch_size)).to(device),
        }
        
        model.eval()
        with torch.no_grad():
            logits = model(src, context_dict, device)
        
        assert logits.device.type == 'cuda'
        print("  ✓ GPU forward pass works")
    else:
        print("  ⚠ CUDA not available, skipping GPU test")
    
    print("  ✓ GPU tests passed\n")


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print("Testing gradient flow...")
    
    config = MockConfig('geolife')
    total_loc_num = 100
    model = LSTMModel(config, total_loc_num)
    device = torch.device('cpu')
    model.train()
    
    seq_len, batch_size = 10, 4
    src = torch.randint(1, total_loc_num, (seq_len, batch_size))
    targets = torch.randint(0, total_loc_num, (batch_size,))
    
    context_dict = {
        'len': torch.tensor([seq_len] * batch_size),
        'user': torch.randint(0, config.total_user_num, (batch_size,)),
        'time': torch.randint(0, 96, (seq_len, batch_size)),
        'weekday': torch.randint(0, 7, (seq_len, batch_size)),
        'duration': torch.randint(0, 96, (seq_len, batch_size)),
        'diff': torch.randint(0, 7, (seq_len, batch_size)),
    }
    
    logits = model(src, context_dict, device)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "Model should have non-zero gradients"
    print("  ✓ Gradients flow through model")
    print("  ✓ Gradient flow tests passed\n")


def test_experiment_output_exists():
    """Test that experiment outputs were created."""
    print("Testing experiment outputs...")
    
    import glob
    
    # Find LSTM experiment folders
    geolife_experiments = glob.glob("/data/next_loc_clean_v2/experiments/geolife_LSTM_*")
    diy_experiments = glob.glob("/data/next_loc_clean_v2/experiments/diy_LSTM_*")
    
    if geolife_experiments:
        exp_dir = geolife_experiments[-1]  # Latest one
        
        # Check required files
        required_files = ['checkpoints/checkpoint.pt', 'config.yaml', 
                         'training.log', 'test_results.json', 'val_results.json']
        
        for f in required_files:
            path = os.path.join(exp_dir, f)
            assert os.path.exists(path), f"Missing: {path}"
        
        print(f"  ✓ GeoLife experiment exists: {exp_dir}")
        
        # Check test results format
        import json
        with open(os.path.join(exp_dir, 'test_results.json'), 'r') as f:
            results = json.load(f)
        
        for key in ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']:
            assert key in results, f"Missing key in results: {key}"
        
        print("  ✓ Test results format correct")
    else:
        print("  ⚠ No GeoLife LSTM experiment found")
    
    if diy_experiments:
        print(f"  ✓ DIY experiment exists: {diy_experiments[-1]}")
    else:
        print("  ⚠ No DIY LSTM experiment found")
    
    print("  ✓ Experiment output tests passed\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("LSTM Model Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_temporal_embedding,
        test_all_embedding_lstm,
        test_lstm_model_geolife,
        test_lstm_model_diy,
        test_metrics,
        test_data_loading,
        test_config_loading,
        test_variable_sequence_length,
        test_packed_sequence,
        test_gradient_flow,
        test_gpu_if_available,
        test_experiment_output_exists,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {str(e)}\n")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

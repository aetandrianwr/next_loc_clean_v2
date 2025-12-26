#!/usr/bin/env python
"""
Test script for MHSA model and training pipeline.

This script validates:
1. Model can be instantiated correctly
2. Forward pass works with correct shapes
3. Data loading works correctly
4. Metrics calculation is correct
5. Training script components work

Usage:
    cd /data/next_loc_clean_v2
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
    python tests/test_MHSA.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline.MHSA import MHSA, PositionalEncoding, TemporalEmbedding, AllEmbedding
from src.evaluation.metrics import calculate_metrics, calculate_correct_total_prediction


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.base_emb_size = 32
        self.if_embed_user = True
        self.if_embed_time = True
        self.if_embed_duration = True
        self.if_embed_poi = False
        self.total_user_num = 10
        self.nhead = 4
        self.dim_feedforward = 64
        self.num_encoder_layers = 2
        self.fc_dropout = 0.1
        self.poi_original_size = 16


def test_positional_encoding():
    """Test PositionalEncoding module."""
    print("Testing PositionalEncoding...")
    
    emb_size = 32
    dropout = 0.1
    pe = PositionalEncoding(emb_size, dropout)
    
    # Test forward pass
    seq_len, batch_size = 10, 4
    x = torch.randn(seq_len, batch_size, emb_size)
    out = pe(x)
    
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    print("  ✓ PositionalEncoding output shape correct")
    
    # Test that positional encoding is added
    pe.eval()  # Disable dropout
    x_zeros = torch.zeros(seq_len, batch_size, emb_size)
    out_zeros = pe(x_zeros)
    assert not torch.allclose(out_zeros, x_zeros), "Positional encoding should modify input"
    print("  ✓ PositionalEncoding adds positional information")
    
    print("  ✓ PositionalEncoding tests passed\n")


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


def test_mhsa_model():
    """Test MHSA model."""
    print("Testing MHSA model...")
    
    config = MockConfig()
    total_loc_num = 100
    
    model = MHSA(config, total_loc_num)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    seq_len, batch_size = 10, 4
    device = torch.device('cpu')
    
    src = torch.randint(1, total_loc_num, (seq_len, batch_size))
    context_dict = {
        'len': torch.tensor([seq_len] * batch_size),
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
    print("  ✓ MHSA forward pass shape correct")
    
    # Test attention maps
    attention_maps = model.get_attention_maps(src, context_dict, device)
    assert len(attention_maps) == config.num_encoder_layers, f"Expected {config.num_encoder_layers} attention maps"
    print("  ✓ MHSA attention maps extraction works")
    
    print("  ✓ MHSA model tests passed\n")


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
        print(f"  ✓ Sample keys: {list(sample.keys())}")
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
    
    geolife_config_path = "/data/next_loc_clean_v2/config/models/config_MHSA_geolife.yaml"
    if os.path.exists(geolife_config_path):
        with open(geolife_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'data' in config, "Config should have 'data' section"
        assert 'model' in config, "Config should have 'model' section"
        assert 'embedding' in config, "Config should have 'embedding' section"
        
        print(f"  ✓ GeoLife config loaded successfully")
    else:
        print("  ⚠ GeoLife config not found")
    
    diy_config_path = "/data/next_loc_clean_v2/config/models/config_MHSA_diy.yaml"
    if os.path.exists(diy_config_path):
        with open(diy_config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ DIY config loaded successfully")
    else:
        print("  ⚠ DIY config not found")
    
    print("  ✓ Config loading tests passed\n")


def test_variable_sequence_length():
    """Test model with variable sequence lengths."""
    print("Testing variable sequence lengths...")
    
    config = MockConfig()
    total_loc_num = 100
    model = MHSA(config, total_loc_num)
    device = torch.device('cpu')
    model.eval()
    
    # Test with different sequence lengths
    for seq_len in [5, 10, 20, 50]:
        batch_size = 4
        src = torch.randint(1, total_loc_num, (seq_len, batch_size))
        context_dict = {
            'len': torch.tensor([seq_len] * batch_size),
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


def test_gpu_if_available():
    """Test model on GPU if available."""
    print("Testing GPU support...")
    
    if torch.cuda.is_available():
        config = MockConfig()
        total_loc_num = 100
        device = torch.device('cuda')
        
        model = MHSA(config, total_loc_num).to(device)
        
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


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MHSA Model Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        test_positional_encoding,
        test_temporal_embedding,
        test_mhsa_model,
        test_metrics,
        test_data_loading,
        test_config_loading,
        test_variable_sequence_length,
        test_gpu_if_available,
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

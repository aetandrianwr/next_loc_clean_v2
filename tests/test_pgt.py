#!/usr/bin/env python
"""
Test script for Pointer Generator Transformer (PGT) model and training pipeline.

This script validates:
1. Model can be instantiated correctly
2. Forward pass works with correct shapes
3. Data loading works correctly
4. Metrics calculation is correct
5. Training script components work
6. Configuration loading works

Usage:
    cd /data/next_loc_clean_v2
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
    python tests/test_pgt.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.proposed.pgt import PointerGeneratorTransformer
from src.evaluation.metrics import calculate_metrics, calculate_correct_total_prediction


def test_model_instantiation():
    """Test that PointerGeneratorTransformer can be instantiated with various configurations."""
    print("Testing model instantiation...")

    # Test default configuration
    model = PointerGeneratorTransformer(
        num_locations=100,
        num_users=10,
    )
    print(f"  Default config: {model.count_parameters():,} parameters")

    # Test GeoLife configuration
    model_geolife = PointerGeneratorTransformer(
        num_locations=1187,
        num_users=46,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.15,
    )
    print(f"  GeoLife config: {model_geolife.count_parameters():,} parameters")

    # Test DIY configuration
    model_diy = PointerGeneratorTransformer(
        num_locations=7038,
        num_users=693,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.15,
    )
    print(f"  DIY config: {model_diy.count_parameters():,} parameters")

    print("  Model instantiation tests passed\n")


def test_forward_pass():
    """Test forward pass with correct shapes."""
    print("Testing forward pass...")

    num_locations = 100
    num_users = 10
    seq_len = 15
    batch_size = 4

    model = PointerGeneratorTransformer(
        num_locations=num_locations,
        num_users=num_users,
        d_model=64,
        nhead=4,
        num_layers=2,
    )
    model.eval()

    # Create input data (seq_len, batch_size) format as expected by training script
    x = torch.randint(1, num_locations, (seq_len, batch_size))
    x_dict = {
        "user": torch.randint(0, num_users, (batch_size,)),
        "time": torch.randint(0, 96, (seq_len, batch_size)),
        "weekday": torch.randint(0, 7, (seq_len, batch_size)),
        "duration": torch.randint(0, 48, (seq_len, batch_size)),
        "diff": torch.randint(0, 8, (seq_len, batch_size)),
        "len": torch.tensor([seq_len] * batch_size),
    }

    with torch.no_grad():
        output = model(x, x_dict)

    expected_shape = (batch_size, num_locations)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )
    print(f"  Output shape: {output.shape}")

    # Check output is log probabilities
    assert (output <= 0).all(), "Output should be log probabilities (all <= 0)"
    print("  Output is log probabilities")

    # Check output sums to ~1 after exp
    probs = torch.exp(output)
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), (
        "Probabilities should sum to 1"
    )
    print("  Probabilities sum to 1")

    print("  Forward pass tests passed\n")


def test_variable_sequence_lengths():
    """Test model with variable sequence lengths."""
    print("Testing variable sequence lengths...")

    model = PointerGeneratorTransformer(
        num_locations=100,
        num_users=10,
        d_model=64,
        nhead=4,
        num_layers=2,
    )
    model.eval()

    for seq_len in [5, 10, 20, 50, 100]:
        batch_size = 4
        x = torch.randint(1, 100, (seq_len, batch_size))
        x_dict = {
            "user": torch.randint(0, 10, (batch_size,)),
            "time": torch.randint(0, 96, (seq_len, batch_size)),
            "weekday": torch.randint(0, 7, (seq_len, batch_size)),
            "duration": torch.randint(0, 48, (seq_len, batch_size)),
            "diff": torch.randint(0, 8, (seq_len, batch_size)),
            "len": torch.tensor([seq_len] * batch_size),
        }

        with torch.no_grad():
            output = model(x, x_dict)

        assert output.shape == (batch_size, 100)
        print(f"  Sequence length {seq_len} works")

    print("  Variable sequence length tests passed\n")


def test_padding_handling():
    """Test that model correctly handles padded sequences."""
    print("Testing padding handling...")

    model = PointerGeneratorTransformer(
        num_locations=100,
        num_users=10,
        d_model=64,
        nhead=4,
        num_layers=2,
    )
    model.eval()

    # Create batched input with different lengths
    max_len = 20
    batch_size = 4
    lengths = [10, 15, 8, 20]

    # Create padded sequence
    x = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, length in enumerate(lengths):
        x[:length, i] = torch.randint(1, 100, (length,))

    x_dict = {
        "user": torch.randint(0, 10, (batch_size,)),
        "time": torch.randint(0, 96, (max_len, batch_size)),
        "weekday": torch.randint(0, 7, (max_len, batch_size)),
        "duration": torch.randint(0, 48, (max_len, batch_size)),
        "diff": torch.randint(0, 8, (max_len, batch_size)),
        "len": torch.tensor(lengths),
    }

    with torch.no_grad():
        output = model(x, x_dict)

    assert output.shape == (batch_size, 100)
    print("  Padded sequences handled correctly")

    print("  Padding handling tests passed\n")


def test_pointer_generation_gate():
    """Test that pointer-generation gate produces valid outputs."""
    print("Testing pointer-generation gate...")

    model = PointerGeneratorTransformer(
        num_locations=100,
        num_users=10,
        d_model=64,
        nhead=4,
        num_layers=2,
    )
    model.eval()

    # Run forward pass multiple times and check gate values
    for _ in range(5):
        x = torch.randint(1, 100, (15, 4))
        x_dict = {
            "user": torch.randint(0, 10, (4,)),
            "time": torch.randint(0, 96, (15, 4)),
            "weekday": torch.randint(0, 7, (15, 4)),
            "duration": torch.randint(0, 48, (15, 4)),
            "diff": torch.randint(0, 8, (15, 4)),
            "len": torch.tensor([15, 15, 15, 15]),
        }

        with torch.no_grad():
            output = model(x, x_dict)

        # Output should be valid probabilities
        probs = torch.exp(output)
        assert (probs >= 0).all(), "Probabilities should be non-negative"
        assert (probs <= 1).all(), "Probabilities should be <= 1"

    print("  Pointer-generation gate produces valid outputs")
    print("  Pointer-generation gate tests passed\n")


def test_metrics_integration():
    """Test that model outputs work with evaluation metrics."""
    print("Testing metrics integration...")

    model = PointerGeneratorTransformer(
        num_locations=100,
        num_users=10,
        d_model=64,
        nhead=4,
        num_layers=2,
    )
    model.eval()

    batch_size = 32
    x = torch.randint(1, 100, (15, batch_size))
    targets = torch.randint(1, 100, (batch_size,))
    x_dict = {
        "user": torch.randint(0, 10, (batch_size,)),
        "time": torch.randint(0, 96, (15, batch_size)),
        "weekday": torch.randint(0, 7, (15, batch_size)),
        "duration": torch.randint(0, 48, (15, batch_size)),
        "diff": torch.randint(0, 8, (15, batch_size)),
        "len": torch.tensor([15] * batch_size),
    }

    with torch.no_grad():
        output = model(x, x_dict)

    # Test with calculate_correct_total_prediction
    results, true_y, pred_y = calculate_correct_total_prediction(output, targets)

    assert len(results) == 7, "Should have 7 result values"
    assert results[-1] == batch_size, "Total should equal batch size"
    print("  calculate_correct_total_prediction works")

    # Test with calculate_metrics
    metrics = calculate_metrics(output, targets)

    required_keys = ["acc@1", "acc@5", "acc@10", "mrr", "ndcg", "f1"]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"

    print("  calculate_metrics works")
    print("  Metrics integration tests passed\n")


def test_data_loading():
    """Test data loading from preprocessed files."""
    print("Testing data loading...")

    import pickle

    # Test GeoLife data
    geolife_path = "/data/next_loc_clean_v2/data/geolife_eps20/processed/geolife_eps20_prev7_train.pk"
    if os.path.exists(geolife_path):
        data = pickle.load(open(geolife_path, "rb"))

        assert len(data) > 0, "Data should not be empty"
        sample = data[0]

        required_keys = [
            "X",
            "Y",
            "user_X",
            "weekday_X",
            "start_min_X",
            "dur_X",
            "diff",
        ]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"

        print(f"  GeoLife data loaded: {len(data)} samples")
    else:
        print("  GeoLife data not found, skipping")

    # Test DIY data
    diy_path = (
        "/data/next_loc_clean_v2/data/diy_eps50/processed/diy_eps50_prev7_train.pk"
    )
    if os.path.exists(diy_path):
        data = pickle.load(open(diy_path, "rb"))
        print(f"  DIY data loaded: {len(data)} samples")
    else:
        print("  DIY data not found, skipping")

    print("  Data loading tests passed\n")


def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")

    import yaml

    configs = [
        "/data/next_loc_clean_v2/config/models/config_pgt_geolife.yaml",
        "/data/next_loc_clean_v2/config/models/config_pgt_diy.yaml",
    ]

    for config_path in configs:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            assert "data" in config, f"Config {config_path} should have 'data' section"
            assert "model" in config, (
                f"Config {config_path} should have 'model' section"
            )
            assert "training" in config, (
                f"Config {config_path} should have 'training' section"
            )

            # Verify model config has required fields
            model_cfg = config["model"]
            required_model_keys = [
                "d_model",
                "nhead",
                "num_layers",
                "dim_feedforward",
                "dropout",
            ]
            for key in required_model_keys:
                assert key in model_cfg, f"Model config missing key: {key}"

            print(f"  {os.path.basename(config_path)} loaded and validated")
        else:
            print(f"  {os.path.basename(config_path)} not found")

    print("  Config loading tests passed\n")


def test_gpu_if_available():
    """Test model on GPU if available."""
    print("Testing GPU support...")

    if torch.cuda.is_available():
        device = torch.device("cuda")

        model = PointerGeneratorTransformer(
            num_locations=100,
            num_users=10,
            d_model=64,
            nhead=4,
            num_layers=2,
        ).to(device)
        model.eval()

        x = torch.randint(1, 100, (15, 4)).to(device)
        x_dict = {
            "user": torch.randint(0, 10, (4,)).to(device),
            "time": torch.randint(0, 96, (15, 4)).to(device),
            "weekday": torch.randint(0, 7, (15, 4)).to(device),
            "duration": torch.randint(0, 48, (15, 4)).to(device),
            "diff": torch.randint(0, 8, (15, 4)).to(device),
            "len": torch.tensor([15, 15, 15, 15]).to(device),
        }

        with torch.no_grad():
            output = model(x, x_dict)

        assert output.device.type == "cuda"
        print("  GPU forward pass works")
    else:
        print("  CUDA not available, skipping GPU test")

    print("  GPU tests passed\n")


def test_amp_support():
    """Test automatic mixed precision support."""
    print("Testing AMP support...")

    if torch.cuda.is_available():
        device = torch.device("cuda")

        model = PointerGeneratorTransformer(
            num_locations=100,
            num_users=10,
            d_model=64,
            nhead=4,
            num_layers=2,
        ).to(device)
        model.eval()

        x = torch.randint(1, 100, (15, 4)).to(device)
        x_dict = {
            "user": torch.randint(0, 10, (4,)).to(device),
            "time": torch.randint(0, 96, (15, 4)).to(device),
            "weekday": torch.randint(0, 7, (15, 4)).to(device),
            "duration": torch.randint(0, 48, (15, 4)).to(device),
            "diff": torch.randint(0, 8, (15, 4)).to(device),
            "len": torch.tensor([15, 15, 15, 15]).to(device),
        }

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(x, x_dict)

        assert output is not None
        print("  AMP forward pass works")
    else:
        print("  CUDA not available, skipping AMP test")

    print("  AMP tests passed\n")


def test_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("Testing gradient flow...")

    model = PointerGeneratorTransformer(
        num_locations=100,
        num_users=10,
        d_model=64,
        nhead=4,
        num_layers=2,
    )
    model.train()

    x = torch.randint(1, 100, (15, 4))
    targets = torch.randint(1, 100, (4,))
    x_dict = {
        "user": torch.randint(0, 10, (4,)),
        "time": torch.randint(0, 96, (15, 4)),
        "weekday": torch.randint(0, 7, (15, 4)),
        "duration": torch.randint(0, 48, (15, 4)),
        "diff": torch.randint(0, 8, (15, 4)),
        "len": torch.tensor([15, 15, 15, 15]),
    }

    output = model(x, x_dict)
    loss = torch.nn.CrossEntropyLoss()(output, targets)
    loss.backward()

    # Check that gradients exist and are not all zeros
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.abs().sum() > 0:
                has_grad = True
                break

    assert has_grad, "Gradients should flow through the model"
    print("  Gradients flow correctly")
    print("  Gradient flow tests passed\n")


def test_experiment_output():
    """Test that experiment outputs are saved correctly."""
    print("Testing experiment outputs...")

    import json

    # Check for recent experiment directories (both old and new naming)
    experiments_dir = "/data/next_loc_clean_v2/experiments"
    pgt_dirs = [
        d for d in os.listdir(experiments_dir) if "pgt" in d or "pointer_v45" in d
    ]

    if pgt_dirs:
        for exp_dir in pgt_dirs:
            exp_path = os.path.join(experiments_dir, exp_dir)

            # Check required files
            required_files = [
                "config.yaml",
                "training.log",
                "test_results.json",
                "val_results.json",
            ]
            for file in required_files:
                file_path = os.path.join(exp_path, file)
                assert os.path.exists(file_path), f"Missing file: {file}"

            # Check checkpoint exists
            checkpoint_path = os.path.join(exp_path, "checkpoints", "best.pt")
            assert os.path.exists(checkpoint_path), (
                f"Missing checkpoint: {checkpoint_path}"
            )

            # Verify test results format
            with open(os.path.join(exp_path, "test_results.json"), "r") as f:
                test_results = json.load(f)

            required_keys = ["acc@1", "acc@5", "acc@10", "mrr", "ndcg", "total"]
            for key in required_keys:
                assert key in test_results, f"Missing key in test results: {key}"

            print(f"  {exp_dir} validated (Acc@1: {test_results['acc@1']:.2f}%)")
    else:
        print("  No PGT experiment directories found")

    print("  Experiment output tests passed\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Pointer Generator Transformer (PGT) Model Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_model_instantiation,
        test_forward_pass,
        test_variable_sequence_lengths,
        test_padding_handling,
        test_pointer_generation_gate,
        test_metrics_integration,
        test_data_loading,
        test_config_loading,
        test_gpu_if_available,
        test_amp_support,
        test_gradient_flow,
        test_experiment_output,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  {test.__name__} FAILED: {str(e)}\n")
            import traceback

            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
Test script for 1st-Order Markov Chain Model.

This script tests the Markov1stModel implementation and its integration
with the evaluation metrics module.

Usage:
    cd /data/next_loc_clean_v2
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
    python tests/test_markov1st.py

Tests:
1. Model initialization and fitting
2. Prediction functionality
3. Logits generation for metrics compatibility
4. Model save/load functionality
5. Integration with metrics.py
6. End-to-end evaluation on real data
"""

import os
import sys
import tempfile
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline.markov1st import Markov1stModel
from src.evaluation.metrics import calculate_metrics


def test_model_initialization():
    """Test model initialization."""
    print("Test 1: Model initialization...")
    
    model = Markov1stModel(num_locations=100, random_seed=42)
    
    assert model.num_locations == 100
    assert model.random_seed == 42
    assert model.total_parameters == 0  # Before fitting
    
    print("  ✓ Model initialization passed")


def test_model_fitting():
    """Test model fitting with synthetic data."""
    print("\nTest 2: Model fitting...")
    
    # Create synthetic training data
    train_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
        {'X': np.array([1, 2, 4]), 'user_X': np.array([1, 1, 1]), 'Y': 3},
        {'X': np.array([2, 3, 4]), 'user_X': np.array([1, 1, 1]), 'Y': 5},
        {'X': np.array([1, 2, 3]), 'user_X': np.array([2, 2, 2]), 'Y': 5},
        {'X': np.array([3, 4, 5]), 'user_X': np.array([2, 2, 2]), 'Y': 6},
    ]
    
    model = Markov1stModel(num_locations=10, random_seed=42)
    model.fit(train_data)
    
    # Check that transitions were learned
    assert len(model.transition_counts) == 2  # Two users
    assert model.total_parameters > 0
    
    print(f"  Total parameters: {model.total_parameters}")
    print("  ✓ Model fitting passed")


def test_prediction():
    """Test prediction functionality."""
    print("\nTest 3: Prediction...")
    
    # Create training data with clear patterns
    train_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},  # Repeated for strong signal
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 5},
    ]
    
    model = Markov1stModel(num_locations=10, random_seed=42)
    model.fit(train_data)
    
    # Test single prediction
    pred = model.predict_single(current_loc=3, user_id=1, top_k=5)
    
    # Location 4 should be predicted first (appears twice after 3)
    assert pred[0] == 4, f"Expected 4 as top prediction, got {pred[0]}"
    assert len(pred) == 5
    
    print(f"  Predictions: {pred}")
    print("  ✓ Prediction passed")


def test_batch_prediction():
    """Test batch prediction on multiple samples."""
    print("\nTest 4: Batch prediction...")
    
    train_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
        {'X': np.array([2, 3, 4]), 'user_X': np.array([1, 1, 1]), 'Y': 5},
    ]
    
    test_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
        {'X': np.array([2, 3, 4]), 'user_X': np.array([1, 1, 1]), 'Y': 5},
    ]
    
    model = Markov1stModel(num_locations=10, random_seed=42)
    model.fit(train_data)
    
    predictions, targets = model.predict(test_data, top_k=5)
    
    assert len(predictions) == 2
    assert len(targets) == 2
    assert targets[0] == 4
    assert targets[1] == 5
    
    print(f"  Predictions: {[p.tolist() for p in predictions]}")
    print(f"  Targets: {targets}")
    print("  ✓ Batch prediction passed")


def test_logits_generation():
    """Test logits generation for metrics compatibility."""
    print("\nTest 5: Logits generation...")
    
    train_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 5},
    ]
    
    test_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
    ]
    
    model = Markov1stModel(num_locations=10, random_seed=42)
    model.fit(train_data)
    
    logits, targets = model.predict_as_logits(test_data)
    
    assert logits.shape == (1, 10)  # (num_samples, num_locations)
    assert targets.shape == (1,)
    assert isinstance(logits, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    
    # Location 4 should have highest logit (most transitions)
    assert torch.argmax(logits[0]).item() == 4
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Top prediction: {torch.argmax(logits[0]).item()}")
    print("  ✓ Logits generation passed")


def test_metrics_integration():
    """Test integration with metrics.py."""
    print("\nTest 6: Metrics integration...")
    
    train_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
        {'X': np.array([2, 3, 4]), 'user_X': np.array([1, 1, 1]), 'Y': 5},
        {'X': np.array([3, 4, 5]), 'user_X': np.array([1, 1, 1]), 'Y': 6},
    ]
    
    test_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
        {'X': np.array([2, 3, 4]), 'user_X': np.array([1, 1, 1]), 'Y': 5},
    ]
    
    model = Markov1stModel(num_locations=10, random_seed=42)
    model.fit(train_data)
    
    logits, targets = model.predict_as_logits(test_data)
    metrics = calculate_metrics(logits, targets)
    
    # Check all expected metrics are present
    expected_keys = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1', 'total']
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
    
    print(f"  Acc@1: {metrics['acc@1']:.2f}%")
    print(f"  MRR: {metrics['mrr']:.2f}%")
    print("  ✓ Metrics integration passed")


def test_save_load():
    """Test model save and load functionality."""
    print("\nTest 7: Save/Load functionality...")
    
    train_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
        {'X': np.array([2, 3, 4]), 'user_X': np.array([1, 1, 1]), 'Y': 5},
    ]
    
    model = Markov1stModel(num_locations=10, random_seed=42)
    model.fit(train_data)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    try:
        model.save(temp_path)
        
        # Load model
        loaded_model = Markov1stModel.load(temp_path)
        
        # Verify loaded model
        assert loaded_model.num_locations == model.num_locations
        assert loaded_model.random_seed == model.random_seed
        assert loaded_model.total_parameters == model.total_parameters
        
        # Test prediction with loaded model
        test_data = [{'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4}]
        
        orig_logits, _ = model.predict_as_logits(test_data)
        loaded_logits, _ = loaded_model.predict_as_logits(test_data)
        
        assert torch.allclose(orig_logits, loaded_logits), "Loaded model produces different predictions"
        
        print("  ✓ Save/Load functionality passed")
    finally:
        os.unlink(temp_path)


def test_real_data():
    """Test on real data if available."""
    print("\nTest 8: Real data integration...")
    
    import pickle
    
    geolife_train = 'data/geolife_eps20/processed/geolife_eps20_prev7_train.pk'
    geolife_test = 'data/geolife_eps20/processed/geolife_eps20_prev7_test.pk'
    
    if not os.path.exists(geolife_train):
        print("  ⚠ Skipping real data test (data not found)")
        return
    
    # Load data
    with open(geolife_train, 'rb') as f:
        train_data = pickle.load(f)
    with open(geolife_test, 'rb') as f:
        test_data = pickle.load(f)
    
    # Use first 1000 samples for quick test
    train_subset = train_data[:1000]
    test_subset = test_data[:100]
    
    model = Markov1stModel(num_locations=1187, random_seed=42)
    model.fit(train_subset)
    
    logits, targets = model.predict_as_logits(test_subset)
    metrics = calculate_metrics(logits, targets)
    
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Test samples: {len(test_subset)}")
    print(f"  Acc@1: {metrics['acc@1']:.2f}%")
    print(f"  MRR: {metrics['mrr']:.2f}%")
    print("  ✓ Real data integration passed")


def test_fallback_behavior():
    """Test fallback behavior when transitions are not found."""
    print("\nTest 9: Fallback behavior...")
    
    train_data = [
        {'X': np.array([1, 2, 3]), 'user_X': np.array([1, 1, 1]), 'Y': 4},
    ]
    
    model = Markov1stModel(num_locations=10, random_seed=42)
    model.fit(train_data)
    
    # Test with unknown user
    pred_unknown_user = model.predict_single(current_loc=3, user_id=999, top_k=5)
    assert len(pred_unknown_user) == 5
    
    # Test with unknown location
    pred_unknown_loc = model.predict_single(current_loc=999, user_id=1, top_k=5)
    assert len(pred_unknown_loc) == 5
    
    print(f"  Unknown user prediction: {pred_unknown_user}")
    print(f"  Unknown location prediction: {pred_unknown_loc}")
    print("  ✓ Fallback behavior passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running 1st-Order Markov Model Tests")
    print("=" * 60)
    
    test_model_initialization()
    test_model_fitting()
    test_prediction()
    test_batch_prediction()
    test_logits_generation()
    test_metrics_integration()
    test_save_load()
    test_real_data()
    test_fallback_behavior()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

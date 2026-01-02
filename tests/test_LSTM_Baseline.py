"""
Test script for LSTM Baseline Model.

This script verifies that the LSTM Baseline implementation works correctly
by testing model instantiation, forward pass, and basic functionality.

Usage:
    python tests/test_LSTM_Baseline.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import unittest


class EasyDict(dict):
    """Dictionary with attribute-style access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class TestLSTMBaseline(unittest.TestCase):
    """Test cases for LSTM Baseline model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EasyDict({
            'base_emb_size': 32,
            'hidden_size': 64,
            'num_layers': 2,
            'lstm_dropout': 0.2,
            'fc_dropout': 0.2,
            'if_embed_user': True,
            'if_embed_time': True,
            'if_embed_duration': True,
            'total_user_num': 46,
        })
        self.total_loc_num = 1187
        self.batch_size = 4
        self.seq_len = 10
        self.device = torch.device('cpu')

    def test_model_import(self):
        """Test that model can be imported."""
        from src.models.baseline.LSTM_Baseline import LSTMBaseline
        self.assertTrue(callable(LSTMBaseline))

    def test_model_instantiation(self):
        """Test model can be instantiated."""
        from src.models.baseline.LSTM_Baseline import LSTMBaseline
        
        model = LSTMBaseline(config=self.config, total_loc_num=self.total_loc_num)
        self.assertIsNotNone(model)
        self.assertEqual(model.d_input, 32)
        self.assertEqual(model.hidden_size, 64)
        self.assertEqual(model.num_layers, 2)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        from src.models.baseline.LSTM_Baseline import LSTMBaseline
        
        model = LSTMBaseline(config=self.config, total_loc_num=self.total_loc_num)
        model.eval()

        # Create dummy input
        x = torch.randint(1, self.total_loc_num, (self.seq_len, self.batch_size))
        x_dict = {
            'len': torch.tensor([self.seq_len] * self.batch_size),
            'user': torch.randint(1, 46, (self.batch_size,)),
            'time': torch.randint(0, 96, (self.seq_len, self.batch_size)),
            'weekday': torch.randint(0, 7, (self.seq_len, self.batch_size)),
            'duration': torch.randint(0, 96, (self.seq_len, self.batch_size)),
        }

        with torch.no_grad():
            logits = model(x, x_dict, self.device)

        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, self.total_loc_num))

    def test_variable_length_sequences(self):
        """Test model handles variable length sequences."""
        from src.models.baseline.LSTM_Baseline import LSTMBaseline
        
        model = LSTMBaseline(config=self.config, total_loc_num=self.total_loc_num)
        model.eval()

        # Create input with variable lengths
        x = torch.randint(1, self.total_loc_num, (self.seq_len, self.batch_size))
        lengths = torch.tensor([10, 8, 6, 4])  # Variable lengths
        
        x_dict = {
            'len': lengths,
            'user': torch.randint(1, 46, (self.batch_size,)),
            'time': torch.randint(0, 96, (self.seq_len, self.batch_size)),
            'weekday': torch.randint(0, 7, (self.seq_len, self.batch_size)),
            'duration': torch.randint(0, 96, (self.seq_len, self.batch_size)),
        }

        with torch.no_grad():
            logits = model(x, x_dict, self.device)

        self.assertEqual(logits.shape, (self.batch_size, self.total_loc_num))

    def test_parameter_count(self):
        """Test model parameter count method."""
        from src.models.baseline.LSTM_Baseline import LSTMBaseline
        
        model = LSTMBaseline(config=self.config, total_loc_num=self.total_loc_num)
        param_count = model.count_parameters()
        
        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)
        
        # Verify against manual count
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(param_count, manual_count)

    def test_no_user_embedding(self):
        """Test model works without user embedding."""
        config_no_user = EasyDict({
            'base_emb_size': 32,
            'hidden_size': 64,
            'num_layers': 2,
            'lstm_dropout': 0.2,
            'fc_dropout': 0.2,
            'if_embed_user': False,  # Disabled
            'if_embed_time': True,
            'if_embed_duration': True,
            'total_user_num': 46,
        })
        
        from src.models.baseline.LSTM_Baseline import LSTMBaseline
        
        model = LSTMBaseline(config=config_no_user, total_loc_num=self.total_loc_num)
        model.eval()

        x = torch.randint(1, self.total_loc_num, (self.seq_len, self.batch_size))
        x_dict = {
            'len': torch.tensor([self.seq_len] * self.batch_size),
            'user': torch.randint(1, 46, (self.batch_size,)),
            'time': torch.randint(0, 96, (self.seq_len, self.batch_size)),
            'weekday': torch.randint(0, 7, (self.seq_len, self.batch_size)),
            'duration': torch.randint(0, 96, (self.seq_len, self.batch_size)),
        }

        with torch.no_grad():
            logits = model(x, x_dict, self.device)

        self.assertEqual(logits.shape, (self.batch_size, self.total_loc_num))

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        from src.models.baseline.LSTM_Baseline import LSTMBaseline
        
        model = LSTMBaseline(config=self.config, total_loc_num=self.total_loc_num)
        model.train()

        x = torch.randint(1, self.total_loc_num, (self.seq_len, self.batch_size))
        y = torch.randint(1, self.total_loc_num, (self.batch_size,))
        x_dict = {
            'len': torch.tensor([self.seq_len] * self.batch_size),
            'user': torch.randint(1, 46, (self.batch_size,)),
            'time': torch.randint(0, 96, (self.seq_len, self.batch_size)),
            'weekday': torch.randint(0, 7, (self.seq_len, self.batch_size)),
            'duration': torch.randint(0, 96, (self.seq_len, self.batch_size)),
        }

        logits = model(x, x_dict, self.device)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")


class TestTrainingScript(unittest.TestCase):
    """Test cases for training script components."""

    def test_config_loading(self):
        """Test configuration loading."""
        import yaml
        
        geolife_config_path = 'config/models/config_LSTM_Baseline_geolife.yaml'
        diy_config_path = 'config/models/config_LSTM_Baseline_diy.yaml'

        # Test Geolife config
        with open(geolife_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertEqual(config['seed'], 42)
        self.assertEqual(config['data']['dataset'], 'geolife')
        self.assertEqual(config['training']['batch_size'], 32)
        self.assertEqual(config['embedding']['base_emb_size'], 32)

        # Test DIY config
        with open(diy_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertEqual(config['seed'], 42)
        self.assertEqual(config['data']['dataset'], 'diy')
        self.assertEqual(config['training']['batch_size'], 256)
        self.assertEqual(config['embedding']['base_emb_size'], 96)

    def test_metrics_import(self):
        """Test metrics module can be imported."""
        from src.evaluation.metrics import calculate_metrics, calculate_correct_total_prediction
        self.assertTrue(callable(calculate_metrics))
        self.assertTrue(callable(calculate_correct_total_prediction))


class TestMetricsIntegration(unittest.TestCase):
    """Test integration with evaluation metrics."""

    def test_metrics_with_model_output(self):
        """Test metrics calculation with model output."""
        from src.evaluation.metrics import calculate_metrics
        
        batch_size = 32
        num_locations = 100
        
        # Simulate model output
        logits = torch.randn(batch_size, num_locations)
        targets = torch.randint(0, num_locations, (batch_size,))
        
        metrics = calculate_metrics(logits, targets)
        
        # Check all expected metrics are present
        expected_keys = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1', 'total']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['acc@1'], 0)
        self.assertLessEqual(metrics['acc@1'], 100)
        self.assertEqual(metrics['total'], batch_size)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

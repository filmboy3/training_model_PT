"""
Unit tests for HDT model architecture.
"""

import unittest
import torch
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.hdt_architecture import HybridDisentanglingTransformer
from src.models.losses import MultiTaskLoss

class TestModelArchitecture(unittest.TestCase):
    """Test HDT model architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = HybridDisentanglingTransformer(
            input_channels=6,
            d_model=256,
            nhead=8,
            num_transformer_layers=6,
            num_cnn_layers=4,
            num_joints=17,
            num_exercise_types=10,
            num_form_classes=5,
            dropout=0.1
        )
        self.batch_size = 4
        self.seq_length = 200
        self.input_channels = 6
    
    def test_model_forward_pass(self):
        """Test model forward pass with correct shapes."""
        # Create dummy input
        imu_data = torch.randn(self.batch_size, self.seq_length, self.input_channels)
        
        # Forward pass
        outputs = self.model(imu_data)
        
        # Check output structure
        expected_keys = ['pose_mean', 'pose_log_var', 'exercise_logits', 'form_logits', 'uncertainty']
        for key in expected_keys:
            self.assertIn(key, outputs)
        
        # Check output shapes
        self.assertEqual(outputs['pose_mean'].shape, (self.batch_size, self.seq_length, 17, 3))
        self.assertEqual(outputs['pose_log_var'].shape, (self.batch_size, self.seq_length, 17, 3))
        self.assertEqual(outputs['exercise_logits'].shape, (self.batch_size, 10))
        self.assertEqual(outputs['form_logits'].shape, (self.batch_size, 5))
        self.assertEqual(outputs['uncertainty'].shape, (self.batch_size, 1))
    
    def test_loss_computation(self):
        """Test multi-task loss computation."""
        loss_fn = MultiTaskLoss()
        
        # Create dummy data
        imu_data = torch.randn(self.batch_size, self.seq_length, self.input_channels)
        target_poses = torch.randn(self.batch_size, self.seq_length, 51)
        exercise_labels = torch.randint(0, 10, (self.batch_size,))
        form_labels = torch.randint(0, 5, (self.batch_size,))
        
        # Forward pass
        outputs = self.model(imu_data)
        
        # Compute loss
        loss_dict = loss_fn(
            outputs['pose_mean'], outputs['pose_log_var'],
            target_poses, exercise_labels, form_labels,
            outputs['exercise_logits'], outputs['form_logits']
        )
        
        # Check loss structure
        self.assertIn('total_loss', loss_dict)
        self.assertIn('pose_loss', loss_dict)
        self.assertIn('exercise_loss', loss_dict)
        self.assertIn('form_loss', loss_dict)
        
        # Check loss values are finite
        for key, value in loss_dict.items():
            self.assertFalse(torch.isnan(value))
            self.assertFalse(torch.isinf(value))

if __name__ == '__main__':
    unittest.main()

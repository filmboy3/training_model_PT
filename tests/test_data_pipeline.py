"""
Unit tests for data pipeline components.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import h5py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.physics_engine import PhysicsDataEngine
from simple_dataset_loader import ProductionDataset

class TestDataPipeline(unittest.TestCase):
    """Test data generation and loading pipeline."""
    
    def test_physics_engine_initialization(self):
        """Test physics engine loads correctly."""
        engine = PhysicsDataEngine()
        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.data)
    
    def test_squat_sequence_generation(self):
        """Test single squat sequence generation."""
        engine = PhysicsDataEngine()
        sequence = engine.generate_squat_sequence(duration=1.0)
        
        # Check structure
        self.assertIn('poses', sequence)
        self.assertIn('imu_data', sequence)
        self.assertIn('exercise_label', sequence)
        self.assertIn('form_label', sequence)
        
        # Check shapes
        poses = sequence['poses']
        imu_data = sequence['imu_data']
        
        self.assertEqual(poses.shape[1], 51)  # 51D pose vector
        self.assertEqual(imu_data.shape[1], 6)  # 6D IMU vector
        self.assertEqual(poses.shape[0], imu_data.shape[0])  # Same sequence length
    
    def test_dataset_loader(self):
        """Test production dataset loader."""
        dataset_path = "data/production_squats_10k.h5"
        
        if Path(dataset_path).exists():
            dataset = ProductionDataset(dataset_path, max_seq_length=200)
            
            # Check dataset properties
            self.assertGreater(len(dataset), 0)
            
            # Check sample
            sample = dataset[0]
            self.assertIn('imu_data', sample)
            self.assertIn('poses', sample)
            self.assertIn('exercise_label', sample)
            self.assertIn('form_label', sample)
            
            # Check tensor shapes
            self.assertEqual(sample['imu_data'].shape, (200, 6))
            self.assertEqual(sample['poses'].shape, (200, 51))

if __name__ == '__main__':
    unittest.main()

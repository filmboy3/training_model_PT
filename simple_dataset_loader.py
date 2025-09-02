"""
Simple dataset loader for production MuJoCo synthetic data.
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ProductionDataset(Dataset):
    """Simple dataset loader for MuJoCo generated data."""
    
    def __init__(self, data_path: str, max_seq_length: int = 200):
        self.data_path = Path(data_path)
        self.max_seq_length = max_seq_length
        
        # Load metadata
        with h5py.File(self.data_path, 'r') as f:
            self.num_sequences = f['metadata'].attrs['num_sequences']
            self.exercise_labels = f['exercise_labels'][:]
            self.form_labels = f['form_labels'][:]
        
        logger.info(f"Loaded dataset: {self.num_sequences} sequences")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            # Load sequence data
            poses = torch.tensor(f[f'poses/sequence_{idx}'][:], dtype=torch.float32)
            imu_data = torch.tensor(f[f'imu_data/sequence_{idx}'][:], dtype=torch.float32)
            
            # Get labels
            exercise_label = torch.tensor([self.exercise_labels[idx]], dtype=torch.long)
            form_label = torch.tensor([self.form_labels[idx]], dtype=torch.long)
            
            # Get actual sequence length
            actual_seq_length = poses.shape[0]
            
            # Pad or truncate to max_seq_length
            if actual_seq_length < self.max_seq_length:
                # Pad sequences
                pad_length = self.max_seq_length - actual_seq_length
                poses = torch.cat([poses, torch.zeros(pad_length, poses.shape[1])], dim=0)
                imu_data = torch.cat([imu_data, torch.zeros(pad_length, imu_data.shape[1])], dim=0)
            elif actual_seq_length > self.max_seq_length:
                # Truncate sequences
                poses = poses[:self.max_seq_length]
                imu_data = imu_data[:self.max_seq_length]
                actual_seq_length = self.max_seq_length
            
            return {
                'imu_data': imu_data,
                'poses': poses,
                'exercise_label': exercise_label,
                'form_label': form_label,
                'seq_length': torch.tensor([actual_seq_length], dtype=torch.long)
            }

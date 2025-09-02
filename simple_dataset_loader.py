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
            self.num_sequences = f.attrs['metadata']['num_sequences']
            self.exercise_labels = f['exercise_labels'][:]
            self.form_labels = f['form_labels'][:]
        
        logger.info(f"Loaded dataset: {self.num_sequences} sequences")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            # Load pose and IMU data
            poses = f['poses'][f'sequence_{idx}'][:]
            imu_data = f['imu_data'][f'sequence_{idx}'][:]
            
            # Pad or truncate to fixed length
            seq_len = min(len(poses), self.max_seq_length)
            
            # Pad sequences if needed
            if len(poses) < self.max_seq_length:
                poses_padded = np.zeros((self.max_seq_length, poses.shape[1]))
                poses_padded[:len(poses)] = poses
                
                imu_padded = np.zeros((self.max_seq_length, imu_data.shape[1]))
                imu_padded[:len(imu_data)] = imu_data
            else:
                poses_padded = poses[:self.max_seq_length]
                imu_padded = imu_data[:self.max_seq_length]
            
            return {
                'imu_data': torch.FloatTensor(imu_padded),
                'poses': torch.FloatTensor(poses_padded),
                'exercise_label': torch.LongTensor([self.exercise_labels[idx]]),
                'form_label': torch.LongTensor([self.form_labels[idx]]),
                'seq_length': torch.LongTensor([seq_len])
            }

"""
Dataset implementation for QuantumLeap Pose Engine.
Handles loading and preprocessing of synthetic physics-based data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import random

from ..data import PhysicsDataEngine, DomainRandomizer, IMUSimulator
from .config import DataConfig

logger = logging.getLogger(__name__)


class QuantumLeapDataset(Dataset):
    """
    Dataset for QuantumLeap Pose Engine training.
    Loads synthetic physics-based motion data with IMU simulation.
    """
    
    def __init__(self,
                 data_path: str,
                 config: DataConfig,
                 split: str = 'train',
                 transform: Optional[callable] = None):
        """
        Args:
            data_path: Path to HDF5 dataset file
            config: Data configuration
            split: 'train', 'val', or 'test'
            transform: Optional data transformation function
        """
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.transform = transform
        
        # Load dataset
        self._load_dataset()
        
        # Setup augmentation
        if split == 'train':
            self.domain_randomizer = DomainRandomizer()
            self.imu_simulator = IMUSimulator()
        else:
            self.domain_randomizer = None
            self.imu_simulator = None
    
    def _load_dataset(self):
        """Load dataset from HDF5 file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # Load metadata
            self.metadata = dict(f.attrs)
            
            # Load sequences
            self.joint_positions = f['joint_positions'][:]
            self.joint_rotations = f['joint_rotations'][:]
            self.imu_data = f['imu_data'][:]
            
            # Load labels if available
            if 'exercise_labels' in f:
                self.exercise_labels = f['exercise_labels'][:]
            else:
                # Default to squat (0) for MVM
                self.exercise_labels = np.zeros(len(self.joint_positions), dtype=np.int64)
            
            if 'form_labels' in f:
                self.form_labels = f['form_labels'][:]
            else:
                # Default to good form (2) for synthetic data
                self.form_labels = np.full(len(self.joint_positions), 2, dtype=np.int64)
        
        # Split dataset
        total_sequences = len(self.joint_positions)
        indices = list(range(total_sequences))
        random.shuffle(indices)
        
        train_end = int(self.config.train_split * total_sequences)
        val_end = train_end + int(self.config.val_split * total_sequences)
        
        if self.split == 'train':
            self.indices = indices[:train_end]
        elif self.split == 'val':
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]
        
        logger.info(f"Loaded {len(self.indices)} sequences for {self.split} split")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        actual_idx = self.indices[idx]
        
        # Load base data
        joint_positions = self.joint_positions[actual_idx]  # [seq_len, joints, 3]
        joint_rotations = self.joint_rotations[actual_idx]  # [seq_len, joints, 4]
        imu_data = self.imu_data[actual_idx]  # [seq_len, 6]
        exercise_label = self.exercise_labels[actual_idx]
        form_label = self.form_labels[actual_idx]
        
        # Apply augmentation for training (disabled for now to avoid tensor issues)
        # TODO: Fix domain randomization tensor compatibility
        pass
        
        # Convert to tensors
        sample = {
            'imu_data': torch.from_numpy(imu_data).float(),
            'joint_positions': torch.from_numpy(joint_positions).float(),
            'joint_rotations': torch.from_numpy(joint_rotations).float(),
            'exercise_label': torch.tensor(exercise_label, dtype=torch.long),
            'form_label': torch.tensor(form_label, dtype=torch.long),
            'sequence_length': torch.tensor(len(imu_data), dtype=torch.long)
        }
        
        # Apply custom transform if provided
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _apply_temporal_jitter(self, 
                              joint_positions: np.ndarray,
                              joint_rotations: np.ndarray,
                              imu_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply temporal jittering for augmentation."""
        seq_len = len(joint_positions)
        
        # Random time shift (±5% of sequence length)
        max_shift = int(0.05 * seq_len)
        shift = random.randint(-max_shift, max_shift)
        
        if shift > 0:
            # Pad beginning, truncate end
            joint_positions = np.concatenate([
                np.repeat(joint_positions[:1], shift, axis=0),
                joint_positions[:-shift]
            ])
            joint_rotations = np.concatenate([
                np.repeat(joint_rotations[:1], shift, axis=0),
                joint_rotations[:-shift]
            ])
            imu_data = np.concatenate([
                np.repeat(imu_data[:1], shift, axis=0),
                imu_data[:-shift]
            ])
        elif shift < 0:
            # Truncate beginning, pad end
            shift = abs(shift)
            joint_positions = np.concatenate([
                joint_positions[shift:],
                np.repeat(joint_positions[-1:], shift, axis=0)
            ])
            joint_rotations = np.concatenate([
                joint_rotations[shift:],
                np.repeat(joint_rotations[-1:], shift, axis=0)
            ])
            imu_data = np.concatenate([
                imu_data[shift:],
                np.repeat(imu_data[-1:], shift, axis=0)
            ])
        
        return joint_positions, joint_rotations, imu_data
    
    def _apply_amplitude_scaling(self, imu_data: np.ndarray) -> np.ndarray:
        """Apply amplitude scaling to IMU data."""
        # Random scaling factor (0.8 to 1.2)
        scale_factor = random.uniform(0.8, 1.2)
        return imu_data * scale_factor
    
    def _apply_rotation_augmentation(self, 
                                   joint_positions: np.ndarray,
                                   imu_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random rotation around vertical axis."""
        # Random rotation angle (±30 degrees)
        angle = random.uniform(-np.pi/6, np.pi/6)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix around Y-axis
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        # Apply rotation to joint positions
        joint_positions_rotated = np.einsum('ij,tkj->tki', rotation_matrix, joint_positions)
        
        # Apply rotation to IMU data (accelerometer and gyroscope)
        accel = imu_data[:, :3]
        gyro = imu_data[:, 3:]
        
        accel_rotated = np.einsum('ij,tj->ti', rotation_matrix, accel)
        gyro_rotated = np.einsum('ij,tj->ti', rotation_matrix, gyro)
        
        imu_data_rotated = np.concatenate([accel_rotated, gyro_rotated], axis=1)
        
        return joint_positions_rotated, imu_data_rotated


class QuantumLeapDataModule:
    """
    Data module for QuantumLeap training.
    Handles dataset creation, data loading, and preprocessing.
    """
    
    def __init__(self, config: DataConfig, data_dir: str = "./data"):
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset paths
        self.dataset_path = self.data_dir / "synthetic_squats.h5"
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """Generate or load dataset."""
        if not self.dataset_path.exists():
            logger.info("Generating synthetic dataset...")
            # Use simple physics engine to avoid MuJoCo dependency issues
            from src.data.simple_physics_engine import SimplePhysicsEngine
            physics_engine = SimplePhysicsEngine()
            physics_engine.generate_dataset(
                num_sequences=self.config.num_sequences,
                sequence_length=self.config.sequence_length,
                output_path=str(self.dataset_path)
            )
            
            logger.info(f"Dataset saved to {self.dataset_path}")
        else:
            logger.info(f"Using existing dataset at {self.dataset_path}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        if stage == 'fit' or stage is None:
            self.train_dataset = QuantumLeapDataset(
                self.dataset_path, self.config, split='train'
            )
            self.val_dataset = QuantumLeapDataset(
                self.dataset_path, self.config, split='val'
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = QuantumLeapDataset(
                self.dataset_path, self.config, split='test'
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False
        )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    Pads sequences to the same length within a batch.
    """
    # Find maximum sequence length in batch
    max_seq_len = max(sample['sequence_length'].item() for sample in batch)
    
    # Initialize batch tensors
    batch_size = len(batch)
    imu_dim = batch[0]['imu_data'].shape[-1]
    num_joints = batch[0]['joint_positions'].shape[-2]
    
    # Padded tensors
    imu_data = torch.zeros(batch_size, max_seq_len, imu_dim)
    joint_positions = torch.zeros(batch_size, max_seq_len, num_joints, 3)
    joint_rotations = torch.zeros(batch_size, max_seq_len, num_joints, 4)
    
    # Masks and labels
    masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    exercise_labels = torch.zeros(batch_size, dtype=torch.long)
    form_labels = torch.zeros(batch_size, dtype=torch.long)
    sequence_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill batch
    for i, sample in enumerate(batch):
        seq_len = sample['sequence_length'].item()
        
        imu_data[i, :seq_len] = sample['imu_data']
        joint_positions[i, :seq_len] = sample['joint_positions']
        joint_rotations[i, :seq_len] = sample['joint_rotations']
        
        masks[i, :seq_len] = True
        exercise_labels[i] = sample['exercise_label']
        form_labels[i] = sample['form_label']
        sequence_lengths[i] = seq_len
    
    return {
        'imu_data': imu_data,
        'joint_positions': joint_positions,
        'joint_rotations': joint_rotations,
        'masks': masks,
        'exercise_labels': exercise_labels,
        'form_labels': form_labels,
        'sequence_lengths': sequence_lengths
    }


if __name__ == "__main__":
    # Test dataset creation and loading
    from .config import DataConfig
    
    config = DataConfig()
    config.num_sequences = 100  # Small test dataset
    config.batch_size = 4
    
    # Create data module
    data_module = QuantumLeapDataModule(config)
    data_module.prepare_data()
    data_module.setup('fit')
    
    # Test data loading
    train_loader = data_module.train_dataloader()
    
    print("Dataset test:")
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Val dataset size: {len(data_module.val_dataset)}")
    
    # Test batch loading
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"IMU data shape: {batch['imu_data'].shape}")
    print(f"Joint positions shape: {batch['joint_positions'].shape}")
    print(f"Exercise labels: {batch['exercise_labels']}")
    print(f"Form labels: {batch['form_labels']}")
    print(f"Sequence lengths: {batch['sequence_lengths']}")

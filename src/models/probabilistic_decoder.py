"""
Probabilistic pose decoder for QuantumLeap Pose Engine.
Outputs pose estimates with uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class ProbabilisticPoseDecoder(nn.Module):
    """
    Probabilistic decoder that outputs pose mean and variance.
    Used within the HDT architecture's MultiTaskDecoder.
    """
    
    def __init__(self,
                 input_dim: int = 256,
                 num_joints: int = 17,
                 hidden_dims: list = [512, 256, 128]):
        super().__init__()
        
        self.num_joints = num_joints
        self.input_dim = input_dim
        
        # Build decoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Output layers for mean and log variance
        self.mean_head = nn.Linear(prev_dim, num_joints * 3)
        self.log_var_head = nn.Linear(prev_dim, num_joints * 3)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through probabilistic decoder.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (pose_mean, pose_log_var) each [batch_size, seq_len, num_joints, 3]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply decoder
        features = self.decoder(x)
        
        # Get mean and log variance
        pose_mean = self.mean_head(features)
        pose_log_var = self.log_var_head(features)
        
        # Reshape to joint format
        pose_mean = pose_mean.view(batch_size, seq_len, self.num_joints, 3)
        pose_log_var = pose_log_var.view(batch_size, seq_len, self.num_joints, 3)
        
        return pose_mean, pose_log_var
    
    def sample(self, 
               pose_mean: torch.Tensor, 
               pose_log_var: torch.Tensor,
               num_samples: int = 1) -> torch.Tensor:
        """
        Sample poses from the predicted distribution.
        
        Args:
            pose_mean: Mean poses [batch_size, seq_len, num_joints, 3]
            pose_log_var: Log variance [batch_size, seq_len, num_joints, 3]
            num_samples: Number of samples to draw
            
        Returns:
            Sampled poses [num_samples, batch_size, seq_len, num_joints, 3]
        """
        pose_std = torch.exp(0.5 * pose_log_var)
        
        samples = []
        for _ in range(num_samples):
            noise = torch.randn_like(pose_mean)
            sample = pose_mean + pose_std * noise
            samples.append(sample)
        
        return torch.stack(samples, dim=0)


if __name__ == "__main__":
    # Test probabilistic decoder
    decoder = ProbabilisticPoseDecoder(
        input_dim=256,
        num_joints=17
    )
    
    # Test forward pass
    batch_size, seq_len, input_dim = 4, 100, 256
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    pose_mean, pose_log_var = decoder(test_input)
    
    print("Probabilistic Decoder Test:")
    print(f"Input shape: {test_input.shape}")
    print(f"Pose mean shape: {pose_mean.shape}")
    print(f"Pose log_var shape: {pose_log_var.shape}")
    
    # Test sampling
    samples = decoder.sample(pose_mean, pose_log_var, num_samples=5)
    print(f"Samples shape: {samples.shape}")
    
    print("✓ Probabilistic decoder test passed!")

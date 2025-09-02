"""
Hybrid Disentangling Transformer (HDT) - Core architecture for QuantumLeap Pose Engine.
This novel architecture separates "what" (exercise type) from "how" (form/style) using
a CNN frontend + Transformer encoder + multi-task probabilistic decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class IMUConvEncoder(nn.Module):
    """
    1D CNN frontend for processing IMU data.
    Extracts local temporal features from raw sensor data.
    """
    
    def __init__(self, 
                 input_channels: int = 6,  # accel_xyz + gyro_xyz
                 hidden_dim: int = 256,
                 num_layers: int = 4):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Progressive 1D convolutions with residual connections
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = hidden_dim // (2 ** (num_layers - i - 1))
            
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=7, 
                         padding=3, stride=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                
                nn.Conv1d(out_channels, out_channels, kernel_size=5,
                         padding=2, stride=2),  # Downsample
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Final projection to transformer dimension
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: IMU data [batch_size, seq_len, input_channels]
            
        Returns:
            Encoded features [batch_size, seq_len//downsampling_factor, hidden_dim]
        """
        # Transpose for conv1d: [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Transpose back: [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # Project to transformer dimension
        x = self.projection(x)
        
        return x


class DisentanglingTransformerEncoder(nn.Module):
    """
    Transformer encoder with disentangling attention mechanism.
    Separates content (what) from style (how) representations.
    """
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Disentangling projections
        self.content_projection = nn.Linear(d_model, d_model // 2)
        self.style_projection = nn.Linear(d_model, d_model // 2)
        
        # Cross-attention for disentangling
        self.content_attention = nn.MultiheadAttention(
            embed_dim=d_model // 2,
            num_heads=nhead // 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.style_attention = nn.MultiheadAttention(
            embed_dim=d_model // 2,
            num_heads=nhead // 2,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.content_norm = nn.LayerNorm(d_model // 2)
        self.style_norm = nn.LayerNorm(d_model // 2)
        
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch_size, seq_len, d_model]
            mask: Attention mask [seq_len, seq_len]
            
        Returns:
            Tuple of (unified_features, content_features, style_features)
        """
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder.pe[:seq_len, :].transpose(0, 1)
        
        # Apply transformer encoder
        unified_features = self.transformer_encoder(x, mask=mask)
        
        # Disentangle into content and style
        content_raw = self.content_projection(unified_features)
        style_raw = self.style_projection(unified_features)
        
        # Apply cross-attention for disentangling
        # Content attends to style (what is independent of how)
        content_features, _ = self.content_attention(
            content_raw, style_raw, style_raw
        )
        content_features = self.content_norm(content_features + content_raw)
        
        # Style attends to content (how is conditioned on what)
        style_features, _ = self.style_attention(
            style_raw, content_raw, content_raw
        )
        style_features = self.style_norm(style_features + style_raw)
        
        return unified_features, content_features, style_features


class MultiTaskDecoder(nn.Module):
    """
    Multi-task decoder with probabilistic outputs.
    Predicts pose, exercise type, and form quality simultaneously.
    """
    
    def __init__(self,
                 unified_dim: int = 256,
                 content_dim: int = 128,
                 style_dim: int = 128,
                 num_joints: int = 17,
                 num_exercise_types: int = 10,
                 num_form_classes: int = 5):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_exercise_types = num_exercise_types
        self.num_form_classes = num_form_classes
        
        # Pose decoder (probabilistic)
        self.pose_decoder = nn.Sequential(
            nn.Linear(unified_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_joints * 3 * 2)  # mean + log_var for xyz
        )
        
        # Exercise type classifier (uses content features)
        self.exercise_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(content_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_exercise_types)
        )
        
        # Form quality regressor (uses style features)
        self.form_regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(style_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_form_classes)
        )
        
        # Uncertainty estimation for pose
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(unified_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, 
                unified_features: torch.Tensor,
                content_features: torch.Tensor,
                style_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            unified_features: [batch_size, seq_len, unified_dim]
            content_features: [batch_size, seq_len, content_dim]
            style_features: [batch_size, seq_len, style_dim]
            
        Returns:
            Dictionary containing all predictions
        """
        batch_size, seq_len, _ = unified_features.shape
        
        # Pose prediction (probabilistic)
        pose_params = self.pose_decoder(unified_features)
        pose_params = pose_params.view(batch_size, seq_len, self.num_joints, 3, 2)
        
        pose_mean = pose_params[..., 0]  # [batch, seq_len, joints, 3]
        pose_log_var = pose_params[..., 1]  # [batch, seq_len, joints, 3]
        
        # Exercise type classification (sequence-level)
        exercise_logits = self.exercise_classifier(
            content_features.transpose(1, 2)  # [batch, dim, seq_len]
        )
        
        # Form quality prediction (sequence-level)
        form_logits = self.form_regressor(
            style_features.transpose(1, 2)  # [batch, dim, seq_len]
        )
        
        # Global uncertainty estimation
        uncertainty = self.uncertainty_estimator(unified_features).mean(dim=1)  # [batch, 1]
        
        return {
            'pose_mean': pose_mean,
            'pose_log_var': pose_log_var,
            'exercise_logits': exercise_logits,
            'form_logits': form_logits,
            'uncertainty': uncertainty
        }


class HybridDisentanglingTransformer(nn.Module):
    """
    Complete HDT architecture for QuantumLeap Pose Engine.
    
    Architecture:
    1. CNN Frontend: Processes raw IMU data
    2. Disentangling Transformer: Separates content from style
    3. Multi-task Decoder: Probabilistic pose + exercise + form predictions
    """
    
    def __init__(self,
                 input_channels: int = 6,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_transformer_layers: int = 6,
                 num_cnn_layers: int = 4,
                 num_joints: int = 17,
                 num_exercise_types: int = 10,
                 num_form_classes: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_channels = input_channels
        self.d_model = d_model
        self.num_joints = num_joints
        
        # CNN Frontend
        self.cnn_encoder = IMUConvEncoder(
            input_channels=input_channels,
            hidden_dim=d_model,
            num_layers=num_cnn_layers
        )
        
        # Disentangling Transformer Encoder
        self.transformer_encoder = DisentanglingTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        # Multi-task Decoder
        self.decoder = MultiTaskDecoder(
            unified_dim=d_model,
            content_dim=d_model // 2,
            style_dim=d_model // 2,
            num_joints=num_joints,
            num_exercise_types=num_exercise_types,
            num_form_classes=num_form_classes
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, 
                imu_data: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete HDT architecture.
        
        Args:
            imu_data: Raw IMU data [batch_size, seq_len, input_channels]
            mask: Optional attention mask
            
        Returns:
            Dictionary containing all model predictions
        """
        # CNN encoding
        cnn_features = self.cnn_encoder(imu_data)
        
        # Transformer encoding with disentangling
        unified_features, content_features, style_features = self.transformer_encoder(
            cnn_features, mask=mask
        )
        
        # Multi-task decoding
        predictions = self.decoder(unified_features, content_features, style_features)
        
        # Add intermediate features for analysis
        predictions.update({
            'cnn_features': cnn_features,
            'unified_features': unified_features,
            'content_features': content_features,
            'style_features': style_features
        })
        
        return predictions
    
    def get_pose_distribution(self, predictions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pose mean and variance from model predictions.
        
        Args:
            predictions: Model output dictionary
            
        Returns:
            Tuple of (pose_mean, pose_variance)
        """
        pose_mean = predictions['pose_mean']
        pose_log_var = predictions['pose_log_var']
        pose_var = torch.exp(pose_log_var)
        
        return pose_mean, pose_var
    
    def sample_poses(self, 
                    predictions: Dict[str, torch.Tensor],
                    num_samples: int = 1) -> torch.Tensor:
        """
        Sample poses from the predicted distributions.
        
        Args:
            predictions: Model output dictionary
            num_samples: Number of samples to draw
            
        Returns:
            Sampled poses [num_samples, batch_size, seq_len, num_joints, 3]
        """
        pose_mean, pose_var = self.get_pose_distribution(predictions)
        pose_std = torch.sqrt(pose_var)
        
        # Sample from Gaussian distribution
        samples = []
        for _ in range(num_samples):
            noise = torch.randn_like(pose_mean)
            sample = pose_mean + pose_std * noise
            samples.append(sample)
        
        return torch.stack(samples, dim=0)


def create_hdt_model(config: Dict) -> HybridDisentanglingTransformer:
    """
    Factory function to create HDT model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized HDT model
    """
    return HybridDisentanglingTransformer(
        input_channels=config.get('input_channels', 6),
        d_model=config.get('d_model', 256),
        nhead=config.get('nhead', 8),
        num_transformer_layers=config.get('num_transformer_layers', 6),
        num_cnn_layers=config.get('num_cnn_layers', 4),
        num_joints=config.get('num_joints', 17),
        num_exercise_types=config.get('num_exercise_types', 10),
        num_form_classes=config.get('num_form_classes', 5),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == "__main__":
    # Test the HDT architecture
    config = {
        'input_channels': 6,
        'd_model': 256,
        'nhead': 8,
        'num_transformer_layers': 6,
        'num_joints': 17,
        'num_exercise_types': 3,  # squat, pushup, overhead_press
        'num_form_classes': 5     # excellent, good, fair, poor, dangerous
    }
    
    model = create_hdt_model(config)
    
    # Test forward pass
    batch_size, seq_len, input_channels = 4, 100, 6
    test_input = torch.randn(batch_size, seq_len, input_channels)
    
    with torch.no_grad():
        predictions = model(test_input)
    
    print("HDT Architecture Test:")
    print(f"Input shape: {test_input.shape}")
    print(f"Pose mean shape: {predictions['pose_mean'].shape}")
    print(f"Pose log_var shape: {predictions['pose_log_var'].shape}")
    print(f"Exercise logits shape: {predictions['exercise_logits'].shape}")
    print(f"Form logits shape: {predictions['form_logits'].shape}")
    print(f"Uncertainty shape: {predictions['uncertainty'].shape}")
    
    # Test pose sampling
    pose_samples = model.sample_poses(predictions, num_samples=5)
    print(f"Pose samples shape: {pose_samples.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

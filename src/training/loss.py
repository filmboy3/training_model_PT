"""
Loss functions for QuantumLeap Pose Engine.
Implements GaussianNLLLoss for probabilistic pose estimation and multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss for probabilistic pose estimation.
    This is the core loss for our probabilistic approach.
    """
    
    def __init__(self, 
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 full_cov: bool = False):
        """
        Args:
            eps: Small value to avoid numerical instability
            reduction: 'mean', 'sum', or 'none'
            full_cov: Whether to use full covariance matrix (not implemented)
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.full_cov = full_cov
        
        if full_cov:
            raise NotImplementedError("Full covariance not implemented yet")
    
    def forward(self, 
                pred_mean: torch.Tensor,
                pred_log_var: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.
        
        Args:
            pred_mean: Predicted means [batch, seq_len, joints, 3]
            pred_log_var: Predicted log variances [batch, seq_len, joints, 3]
            target: Ground truth poses [batch, seq_len, joints, 3]
            mask: Optional mask for valid timesteps [batch, seq_len]
            
        Returns:
            NLL loss scalar
        """
        # Ensure numerical stability
        pred_var = torch.exp(pred_log_var) + self.eps
        
        # Compute squared error
        squared_error = (pred_mean - target) ** 2
        
        # Compute NLL: 0.5 * (log(2π) + log(var) + (x-μ)²/var)
        nll = 0.5 * (
            np.log(2 * np.pi) + 
            pred_log_var + 
            squared_error / pred_var
        )
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match pose dimensions
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [batch, seq_len, 1, 1]
            mask_expanded = mask_expanded.expand_as(nll)
            nll = nll * mask_expanded
            
            # Adjust reduction for masked elements
            if self.reduction == 'mean':
                return nll.sum() / mask_expanded.sum()
            elif self.reduction == 'sum':
                return nll.sum()
        
        # Standard reduction
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted loss that adapts based on model confidence.
    Higher uncertainty regions get lower weight.
    """
    
    def __init__(self, base_loss_fn: nn.Module, uncertainty_weight: float = 1.0):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, 
                pred_mean: torch.Tensor,
                pred_log_var: torch.Tensor,
                target: torch.Tensor,
                uncertainty: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute uncertainty-weighted loss.
        
        Args:
            pred_mean: Predicted means
            pred_log_var: Predicted log variances
            target: Ground truth
            uncertainty: Global uncertainty estimate [batch, 1]
            mask: Optional mask
            
        Returns:
            Weighted loss
        """
        # Compute base loss
        base_loss = self.base_loss_fn(pred_mean, pred_log_var, target, mask)
        
        # Weight by inverse uncertainty (higher uncertainty = lower weight)
        uncertainty_weight = 1.0 / (1.0 + self.uncertainty_weight * uncertainty.mean())
        
        return base_loss * uncertainty_weight


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining pose estimation, exercise classification, and form assessment.
    Uses learnable task weighting for optimal balance.
    """
    
    def __init__(self,
                 pose_weight: float = 1.0,
                 exercise_weight: float = 0.5,
                 form_weight: float = 0.3,
                 uncertainty_weight: float = 0.1,
                 learnable_weights: bool = True):
        super().__init__()
        
        # Loss functions
        self.pose_loss_fn = GaussianNLLLoss(reduction='mean')
        self.exercise_loss_fn = nn.CrossEntropyLoss()
        self.form_loss_fn = nn.CrossEntropyLoss()
        self.uncertainty_loss_fn = nn.MSELoss()
        
        # Task weights
        if learnable_weights:
            # Learnable task weights (log-parameterized for stability)
            self.log_pose_weight = nn.Parameter(torch.log(torch.tensor(pose_weight)))
            self.log_exercise_weight = nn.Parameter(torch.log(torch.tensor(exercise_weight)))
            self.log_form_weight = nn.Parameter(torch.log(torch.tensor(form_weight)))
            self.log_uncertainty_weight = nn.Parameter(torch.log(torch.tensor(uncertainty_weight)))
        else:
            # Fixed weights
            self.register_buffer('log_pose_weight', torch.log(torch.tensor(pose_weight)))
            self.register_buffer('log_exercise_weight', torch.log(torch.tensor(exercise_weight)))
            self.register_buffer('log_form_weight', torch.log(torch.tensor(form_weight)))
            self.register_buffer('log_uncertainty_weight', torch.log(torch.tensor(uncertainty_weight)))
        
        self.learnable_weights = learnable_weights
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth targets dictionary
            mask: Optional sequence mask
            
        Returns:
            Dictionary of losses and total loss
        """
        losses = {}
        
        # Pose estimation loss (primary task)
        if 'pose_mean' in predictions and 'poses' in targets:
            pose_loss = self.pose_loss_fn(
                predictions['pose_mean'],
                predictions['pose_log_var'],
                targets['poses'],
                mask
            )
            losses['pose_loss'] = pose_loss
        
        # Exercise classification loss
        if 'exercise_logits' in predictions and 'exercise_labels' in targets:
            exercise_loss = self.exercise_loss_fn(
                predictions['exercise_logits'],
                targets['exercise_labels']
            )
            losses['exercise_loss'] = exercise_loss
        
        # Form quality assessment loss
        if 'form_logits' in predictions and 'form_labels' in targets:
            form_loss = self.form_loss_fn(
                predictions['form_logits'],
                targets['form_labels']
            )
            losses['form_loss'] = form_loss
        
        # Uncertainty calibration loss
        if 'uncertainty' in predictions and 'uncertainty_target' in targets:
            uncertainty_loss = self.uncertainty_loss_fn(
                predictions['uncertainty'],
                targets['uncertainty_target']
            )
            losses['uncertainty_loss'] = uncertainty_loss
        
        # Compute weighted total loss
        total_loss = 0.0
        
        if 'pose_loss' in losses:
            weight = torch.exp(self.log_pose_weight)
            total_loss += weight * losses['pose_loss']
            losses['pose_weight'] = weight
        
        if 'exercise_loss' in losses:
            weight = torch.exp(self.log_exercise_weight)
            total_loss += weight * losses['exercise_loss']
            losses['exercise_weight'] = weight
        
        if 'form_loss' in losses:
            weight = torch.exp(self.log_form_weight)
            total_loss += weight * losses['form_loss']
            losses['form_weight'] = weight
        
        if 'uncertainty_loss' in losses:
            weight = torch.exp(self.log_uncertainty_weight)
            total_loss += weight * losses['uncertainty_loss']
            losses['uncertainty_weight'] = weight
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        return {
            'pose_weight': torch.exp(self.log_pose_weight).item(),
            'exercise_weight': torch.exp(self.log_exercise_weight).item(),
            'form_weight': torch.exp(self.log_form_weight).item(),
            'uncertainty_weight': torch.exp(self.log_uncertainty_weight).item()
        }


class CalibrationLoss(nn.Module):
    """
    Expected Calibration Error (ECE) for uncertainty calibration.
    Measures how well predicted confidence matches actual accuracy.
    """
    
    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = n_bins
    
    def forward(self,
                pred_mean: torch.Tensor,
                pred_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute Expected Calibration Error.
        
        Args:
            pred_mean: Predicted means [batch, seq_len, joints, 3]
            pred_var: Predicted variances [batch, seq_len, joints, 3]
            target: Ground truth [batch, seq_len, joints, 3]
            
        Returns:
            ECE loss
        """
        # Compute prediction confidence (inverse of uncertainty)
        pred_std = torch.sqrt(pred_var)
        confidence = 1.0 / (1.0 + pred_std.mean(dim=-1))  # [batch, seq_len, joints]
        
        # Compute accuracy (inverse of error)
        error = torch.norm(pred_mean - target, dim=-1)  # [batch, seq_len, joints]
        accuracy = 1.0 / (1.0 + error)
        
        # Flatten for binning
        confidence_flat = confidence.flatten()
        accuracy_flat = accuracy.flatten()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=confidence.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidence_flat > bin_lower) & (confidence_flat <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracy_flat[in_bin].mean()
                avg_confidence_in_bin = confidence_flat[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class PhysicsConstraintLoss(nn.Module):
    """
    Physics-based constraint loss to ensure biomechanically plausible poses.
    Enforces joint angle limits and kinematic consistency.
    """
    
    def __init__(self, 
                 joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
                 bone_length_constraints: bool = True):
        super().__init__()
        
        # Default joint limits (in radians)
        self.joint_limits = joint_limits or {
            'knee': (-2.8, 0.0),      # Knee can't hyperextend
            'elbow': (-2.6, 0.0),     # Elbow can't hyperextend
            'hip': (-1.6, 1.6),       # Hip flexion/extension
            'shoulder': (-3.1, 3.1),  # Shoulder rotation
        }
        
        self.bone_length_constraints = bone_length_constraints
    
    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute physics constraint violations.
        
        Args:
            poses: Predicted poses [batch, seq_len, joints, 3]
            
        Returns:
            Physics constraint loss
        """
        batch_size, seq_len, num_joints, _ = poses.shape
        
        total_loss = 0.0
        
        # Bone length consistency loss
        if self.bone_length_constraints:
            bone_length_loss = self._compute_bone_length_loss(poses)
            total_loss += bone_length_loss
        
        # Joint angle limit loss
        joint_angle_loss = self._compute_joint_angle_loss(poses)
        total_loss += joint_angle_loss
        
        return total_loss
    
    def _compute_bone_length_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """Enforce consistent bone lengths across time."""
        # Define bone connections (simplified skeleton)
        bone_connections = [
            (0, 1),   # torso to head
            (0, 2),   # torso to left shoulder
            (0, 3),   # torso to right shoulder
            (2, 4),   # left shoulder to left elbow
            (3, 5),   # right shoulder to right elbow
            (0, 6),   # torso to left hip
            (0, 7),   # torso to right hip
            (6, 8),   # left hip to left knee
            (7, 9),   # right hip to right knee
            (8, 10),  # left knee to left ankle
            (9, 11),  # right knee to right ankle
        ]
        
        bone_length_loss = 0.0
        
        for joint1, joint2 in bone_connections:
            if joint1 < poses.size(2) and joint2 < poses.size(2):
                # Compute bone lengths across time
                bone_vectors = poses[:, :, joint2] - poses[:, :, joint1]
                bone_lengths = torch.norm(bone_vectors, dim=-1)  # [batch, seq_len]
                
                # Penalize variance in bone length (should be constant)
                length_variance = torch.var(bone_lengths, dim=1)  # [batch]
                bone_length_loss += length_variance.mean()
        
        return bone_length_loss
    
    def _compute_joint_angle_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """Enforce joint angle limits."""
        # This is a simplified implementation
        # In practice, would need proper joint angle computation from 3D positions
        
        # For now, just penalize extreme joint positions
        joint_position_loss = 0.0
        
        # Penalize joints that are too far from anatomical positions
        # This is a placeholder - real implementation would compute actual joint angles
        extreme_positions = torch.clamp(torch.abs(poses) - 2.0, min=0.0)
        joint_position_loss = extreme_positions.mean()
        
        return joint_position_loss


if __name__ == "__main__":
    # Test loss functions
    batch_size, seq_len, num_joints = 4, 100, 17
    
    # Test GaussianNLLLoss
    gaussian_loss = GaussianNLLLoss()
    
    pred_mean = torch.randn(batch_size, seq_len, num_joints, 3)
    pred_log_var = torch.randn(batch_size, seq_len, num_joints, 3) * 0.1
    target = torch.randn(batch_size, seq_len, num_joints, 3)
    mask = torch.ones(batch_size, seq_len)
    
    nll_loss = gaussian_loss(pred_mean, pred_log_var, target, mask)
    print(f"Gaussian NLL Loss: {nll_loss.item():.4f}")
    
    # Test MultiTaskLoss
    multi_task_loss = MultiTaskLoss()
    
    predictions = {
        'pose_mean': pred_mean,
        'pose_log_var': pred_log_var,
        'exercise_logits': torch.randn(batch_size, 3),
        'form_logits': torch.randn(batch_size, 5),
        'uncertainty': torch.rand(batch_size, 1)
    }
    
    targets = {
        'poses': target,
        'exercise_labels': torch.randint(0, 3, (batch_size,)),
        'form_labels': torch.randint(0, 5, (batch_size,)),
        'uncertainty_target': torch.rand(batch_size, 1)
    }
    
    losses = multi_task_loss(predictions, targets, mask)
    print(f"Multi-task losses: {losses}")
    print(f"Task weights: {multi_task_loss.get_task_weights()}")
    
    # Test CalibrationLoss
    cal_loss = CalibrationLoss()
    pred_var = torch.exp(pred_log_var)
    ece = cal_loss(pred_mean, pred_var, target)
    print(f"Expected Calibration Error: {ece.item():.4f}")
    
    # Test PhysicsConstraintLoss
    physics_loss = PhysicsConstraintLoss()
    physics_constraint = physics_loss(pred_mean)
    print(f"Physics Constraint Loss: {physics_constraint.item():.4f}")

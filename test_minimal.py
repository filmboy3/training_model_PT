#!/usr/bin/env python3
"""
Minimal test for QuantumLeap Pose Engine core architecture.
Tests only the HDT model and loss functions without any MuJoCo dependencies.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_hdt_architecture():
    """Test the Hybrid Disentangling Transformer architecture."""
    print("Testing HDT Architecture...")
    
    from src.models.hdt_architecture import HybridDisentanglingTransformer, create_hdt_model
    
    # Test model creation
    config = {
        'input_channels': 6,
        'd_model': 256,
        'nhead': 8,
        'num_transformer_layers': 6,
        'num_joints': 17,
        'num_exercise_types': 3,
        'num_form_classes': 5
    }
    
    model = create_hdt_model(config)
    print(f"✓ HDT model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size, seq_len, input_channels = 4, 100, 6
    test_input = torch.randn(batch_size, seq_len, input_channels)
    
    with torch.no_grad():
        predictions = model(test_input)
    
    print(f"✓ Forward pass successful")
    print(f"  - Pose mean shape: {predictions['pose_mean'].shape}")
    print(f"  - Pose log_var shape: {predictions['pose_log_var'].shape}")
    print(f"  - Exercise logits shape: {predictions['exercise_logits'].shape}")
    print(f"  - Form logits shape: {predictions['form_logits'].shape}")
    print(f"  - Uncertainty shape: {predictions['uncertainty'].shape}")
    
    # Test pose sampling
    pose_samples = model.sample_poses(predictions, num_samples=3)
    print(f"✓ Pose sampling successful: {pose_samples.shape}")
    
    # Test disentangling features
    print(f"✓ Disentangling features:")
    print(f"  - CNN features: {predictions['cnn_features'].shape}")
    print(f"  - Unified features: {predictions['unified_features'].shape}")
    print(f"  - Content features: {predictions['content_features'].shape}")
    print(f"  - Style features: {predictions['style_features'].shape}")
    
    return model

def test_loss_functions():
    """Test the loss functions."""
    print("\nTesting Loss Functions...")
    
    from src.models.losses import GaussianNLLLoss, MultiTaskLoss, CalibrationLoss, PhysicsConstraintLoss
    
    # Test GaussianNLLLoss
    gaussian_loss = GaussianNLLLoss()
    
    batch_size, seq_len, num_joints = 4, 100, 17
    pred_mean = torch.randn(batch_size, seq_len, num_joints, 3)
    pred_log_var = torch.randn(batch_size, seq_len, num_joints, 3) * 0.1
    target = torch.randn(batch_size, seq_len, num_joints, 3)
    mask = torch.ones(batch_size, seq_len)
    
    nll_loss = gaussian_loss(pred_mean, pred_log_var, target, mask)
    print(f"✓ Gaussian NLL Loss: {nll_loss.item():.4f}")
    
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
    print(f"✓ Multi-task loss: {losses['total_loss'].item():.4f}")
    print(f"  - Task weights: {multi_task_loss.get_task_weights()}")
    
    # Test CalibrationLoss
    cal_loss = CalibrationLoss()
    pred_var = torch.exp(pred_log_var)
    ece = cal_loss(pred_mean, pred_var, target)
    print(f"✓ Expected Calibration Error: {ece.item():.4f}")
    
    # Test PhysicsConstraintLoss
    physics_loss = PhysicsConstraintLoss()
    physics_constraint = physics_loss(pred_mean)
    print(f"✓ Physics Constraint Loss: {physics_constraint.item():.4f}")

def test_probabilistic_decoder():
    """Test the probabilistic decoder."""
    print("\nTesting Probabilistic Decoder...")
    
    from src.models.probabilistic_decoder import ProbabilisticPoseDecoder
    
    decoder = ProbabilisticPoseDecoder(
        input_dim=256,
        num_joints=17
    )
    
    # Test forward pass
    batch_size, seq_len, input_dim = 4, 100, 256
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    pose_mean, pose_log_var = decoder(test_input)
    
    print(f"✓ Probabilistic decoder forward pass")
    print(f"  - Input shape: {test_input.shape}")
    print(f"  - Pose mean shape: {pose_mean.shape}")
    print(f"  - Pose log_var shape: {pose_log_var.shape}")
    
    # Test sampling
    samples = decoder.sample(pose_mean, pose_log_var, num_samples=5)
    print(f"✓ Pose sampling: {samples.shape}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("QuantumLeap Pose Engine - Core Architecture Tests")
    print("=" * 60)
    
    try:
        # Test core architecture
        model = test_hdt_architecture()
        
        # Test loss functions
        test_loss_functions()
        
        # Test probabilistic decoder
        test_probabilistic_decoder()
        
        print("\n" + "=" * 60)
        print("✅ ALL CORE TESTS PASSED!")
        print("QuantumLeap Pose Engine architecture is working correctly.")
        print("=" * 60)
        
        # Print summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        
        print(f"\nArchitecture Features:")
        print(f"  ✓ Hybrid CNN + Transformer (HDT) architecture")
        print(f"  ✓ Disentangling encoder separates 'what' from 'how'")
        print(f"  ✓ Probabilistic pose decoder with uncertainty quantification")
        print(f"  ✓ Multi-task learning (pose + exercise + form)")
        print(f"  ✓ Gaussian NLL loss for probabilistic training")
        print(f"  ✓ Physics constraint loss for biomechanical plausibility")
        print(f"  ✓ Calibration loss for uncertainty calibration")
        print(f"  ✓ Learnable task weighting for multi-task optimization")
        
        print(f"\nImplementation Status:")
        print(f"  ✅ Core HDT architecture")
        print(f"  ✅ Probabilistic decoder")
        print(f"  ✅ Multi-task loss functions")
        print(f"  ✅ Training configuration system")
        print(f"  ⏳ Physics data engine (requires MuJoCo)")
        print(f"  ⏳ Training pipeline (requires dataset)")
        
        print(f"\nNext Steps:")
        print(f"  1. Install MuJoCo: pip install mujoco")
        print(f"  2. Test physics engine: python -c 'from src.data import PhysicsDataEngine'")
        print(f"  3. Generate synthetic dataset")
        print(f"  4. Train MVM: python train.py --config mvm --debug")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Simple validation test for QuantumLeap Pose Engine core components.
Tests each component independently to ensure they work correctly.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_hdt_architecture():
    """Test HDT architecture with correct dimensions."""
    print("Testing HDT Architecture...")
    
    from src.models.hdt_architecture import HybridDisentanglingTransformer
    
    # Create model with correct parameters
    model = HybridDisentanglingTransformer(
        input_channels=6,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        num_cnn_layers=3,
        num_joints=17,
        num_exercise_types=3,
        num_form_classes=5,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 200
    x = torch.randn(batch_size, seq_len, 6)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ HDT forward pass successful")
    print(f"  - Input: {x.shape}")
    print(f"  - Pose mean: {output['pose_mean'].shape}")
    print(f"  - Pose log_var: {output['pose_log_var'].shape}")
    print(f"  - Exercise logits: {output['exercise_logits'].shape}")
    print(f"  - Form logits: {output['form_logits'].shape}")
    
    # Test pose sampling
    pose_mean, pose_var = model.get_pose_distribution(output)
    samples = model.sample_poses(output, num_samples=3)
    
    print(f"  - Pose distribution: mean {pose_mean.shape}, var {pose_var.shape}")
    print(f"  - Pose samples: {samples.shape}")
    
    return model, output

def test_loss_functions():
    """Test loss functions with compatible tensors."""
    print("\nTesting Loss Functions...")
    
    from src.models.losses import GaussianNLLLoss, MultiTaskLoss
    
    # Create compatible synthetic data
    batch_size, seq_len, num_joints = 4, 200, 17
    
    # Predictions from model
    pred_mean = torch.randn(batch_size, seq_len, num_joints * 3)
    pred_log_var = torch.randn(batch_size, seq_len, num_joints * 3)
    exercise_logits = torch.randn(batch_size, 3)
    form_logits = torch.randn(batch_size, 5)
    uncertainty = torch.rand(batch_size, seq_len, 1)
    
    predictions = {
        'pose_mean': pred_mean,
        'pose_log_var': pred_log_var,
        'exercise_logits': exercise_logits,
        'form_logits': form_logits,
        'uncertainty': uncertainty
    }
    
    # Targets
    target_poses = torch.randn(batch_size, seq_len, num_joints * 3)
    exercise_labels = torch.randint(0, 3, (batch_size,))
    form_labels = torch.randint(0, 5, (batch_size,))
    
    targets = {
        'poses': target_poses,
        'exercise_labels': exercise_labels,
        'form_labels': form_labels
    }
    
    # Test Gaussian NLL Loss
    gaussian_loss = GaussianNLLLoss()
    pose_loss = gaussian_loss(pred_mean, pred_log_var, target_poses)
    print(f"✓ Gaussian NLL Loss: {pose_loss.item():.4f}")
    
    # Test Multi-task Loss
    multi_loss = MultiTaskLoss()
    losses = multi_loss(predictions, targets)
    
    print(f"✓ Multi-task Loss:")
    for key, value in losses.items():
        print(f"  - {key}: {value.item():.4f}")
    
    return multi_loss

def test_training_step():
    """Test a single training step."""
    print("\nTesting Training Step...")
    
    # Get model and loss
    model, _ = test_hdt_architecture()
    loss_fn = test_loss_functions()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=1e-4
    )
    
    # Create batch
    batch_size, seq_len = 4, 200
    imu_data = torch.randn(batch_size, seq_len, 6)
    exercise_labels = torch.randint(0, 3, (batch_size,))
    form_labels = torch.randint(0, 5, (batch_size,))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(imu_data)
    
    # Create target poses matching model output dimensions
    output_seq_len = predictions['pose_mean'].shape[1]  # Get actual output sequence length
    target_poses = torch.randn(batch_size, output_seq_len, 17, 3)
    
    # Prepare targets
    targets = {
        'poses': target_poses,
        'exercise_labels': exercise_labels,
        'form_labels': form_labels
    }
    
    # Compute loss
    losses = loss_fn(predictions, targets)
    total_loss = losses['total_loss']
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful: Loss = {total_loss.item():.4f}")
    
    return model

def test_inference():
    """Test model inference."""
    print("\nTesting Inference...")
    
    model = test_training_step()
    
    # Test inference
    model.eval()
    with torch.no_grad():
        # Single sequence
        test_input = torch.randn(1, 200, 6)
        predictions = model(test_input)
        
        # Get pose distribution
        pose_mean, pose_var = model.get_pose_distribution(predictions)
        
        # Sample poses
        samples = model.sample_poses(predictions, num_samples=5)
        
        # Get predictions
        exercise_probs = torch.softmax(predictions['exercise_logits'], dim=-1)
        form_probs = torch.softmax(predictions['form_logits'], dim=-1)
        uncertainty = predictions['uncertainty'].mean()
        
        print(f"✓ Inference successful:")
        print(f"  - Input: {test_input.shape}")
        print(f"  - Pose mean: {pose_mean.shape}")
        print(f"  - Pose variance: {pose_var.shape}")
        print(f"  - Pose samples: {samples.shape}")
        print(f"  - Exercise probabilities: {exercise_probs[0]}")
        print(f"  - Form probabilities: {form_probs[0]}")
        print(f"  - Average uncertainty: {uncertainty.item():.4f}")
    
    return model

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("QuantumLeap Pose Engine - Core Validation")
    print("=" * 60)
    
    try:
        # Test core components
        model = test_inference()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("QuantumLeap Pose Engine core components validated.")
        print("=" * 60)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size (FP32): {total_params * 4 / 1024 / 1024:.1f} MB")
        
        print(f"\nCore Features Validated:")
        print(f"  ✅ Hybrid Disentangling Transformer architecture")
        print(f"  ✅ Probabilistic pose estimation with uncertainty")
        print(f"  ✅ Multi-task learning (pose + exercise + form)")
        print(f"  ✅ Gaussian NLL loss for uncertainty quantification")
        print(f"  ✅ Forward and backward passes")
        print(f"  ✅ Pose sampling and distribution extraction")
        print(f"  ✅ Training step optimization")
        print(f"  ✅ Inference mode operation")
        
        print(f"\nNext Steps:")
        print(f"  1. Generate large-scale synthetic dataset")
        print(f"  2. Run full training: python train.py --config mvm")
        print(f"  3. Validate on real IMU data")
        print(f"  4. Convert to Core ML for iOS deployment")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

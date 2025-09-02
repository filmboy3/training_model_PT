#!/usr/bin/env python3
"""
Test script for QuantumLeap Pose Engine core components.
Tests the HDT architecture and training pipeline without requiring MuJoCo.
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
    
    return model

def test_loss_functions():
    """Test the loss functions."""
    print("\nTesting Loss Functions...")
    
    from src.models.losses import GaussianNLLLoss, MultiTaskLoss
    
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

def test_training_config():
    """Test training configuration."""
    print("\nTesting Training Configuration...")
    
    from src.training.config import get_mvm_config, TrainingConfig
    
    # Test MVM config
    config = get_mvm_config()
    print(f"✓ MVM config created")
    print(f"  - Device: {config.system.device}")
    print(f"  - Model d_model: {config.model.d_model}")
    print(f"  - Batch size: {config.data.batch_size}")
    print(f"  - Max epochs: {config.optimization.max_epochs}")
    
    # Test serialization
    config_dict = config.to_dict()
    loaded_config = TrainingConfig.from_dict(config_dict)
    print(f"✓ Config serialization works")

def test_domain_randomization():
    """Test domain randomization without physics engine."""
    print("\nTesting Domain Randomization...")
    
    from src.data.domain_randomization import DomainRandomizer
    
    randomizer = DomainRandomizer()
    
    # Test sensor noise
    clean_imu = torch.randn(100, 6)  # 100 timesteps, 6 channels
    noisy_imu = randomizer.apply_sensor_noise(clean_imu)
    
    print(f"✓ Sensor noise applied")
    print(f"  - Input shape: {clean_imu.shape}")
    print(f"  - Output shape: {noisy_imu.shape}")
    print(f"  - Noise level: {(noisy_imu - clean_imu).std().item():.4f}")

def test_imu_simulator():
    """Test IMU simulator without physics engine."""
    print("\nTesting IMU Simulator...")
    
    from src.data.imu_simulator import IMUSimulator
    
    simulator = IMUSimulator()
    
    # Create dummy motion data
    seq_len = 100
    positions = torch.randn(seq_len, 3)  # Random 3D positions
    rotations = torch.randn(seq_len, 4)  # Random quaternions
    
    # Simulate IMU data
    imu_data = simulator.simulate_from_motion(positions, rotations)
    
    print(f"✓ IMU simulation successful")
    print(f"  - Input positions: {positions.shape}")
    print(f"  - Output IMU data: {imu_data.shape}")
    print(f"  - Accel range: [{imu_data[:, :3].min().item():.2f}, {imu_data[:, :3].max().item():.2f}]")
    print(f"  - Gyro range: [{imu_data[:, 3:].min().item():.2f}, {imu_data[:, 3:].max().item():.2f}]")

def main():
    """Run all tests."""
    print("=" * 60)
    print("QuantumLeap Pose Engine - Core Component Tests")
    print("=" * 60)
    
    try:
        # Test core architecture
        model = test_hdt_architecture()
        
        # Test loss functions
        test_loss_functions()
        
        # Test configuration
        test_training_config()
        
        # Test domain randomization
        test_domain_randomization()
        
        # Test IMU simulator
        test_imu_simulator()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("QuantumLeap Pose Engine core components are working correctly.")
        print("=" * 60)
        
        # Print summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        
        print(f"\nNext Steps:")
        print(f"  1. Install MuJoCo: pip install mujoco")
        print(f"  2. Generate synthetic dataset: python -c 'from src.data import PhysicsDataEngine; PhysicsDataEngine().generate_dataset(1000, 200, \"data/test.h5\")'")
        print(f"  3. Train MVM: python train.py --config mvm --debug")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

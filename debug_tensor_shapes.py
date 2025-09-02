"""
Debug tensor shapes throughout the forward pass to identify the mismatch.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch.utils.data import DataLoader, random_split
from src.models.hdt_architecture import HybridDisentanglingTransformer
from src.models.losses import MultiTaskLoss
from simple_dataset_loader import ProductionDataset

def debug_tensor_shapes():
    print("🔍 Debugging tensor shapes...")
    
    # Load small dataset
    dataset = ProductionDataset("data/production_squats_10k.h5", max_seq_length=200)
    subset_dataset, _ = random_split(dataset, [10, len(dataset) - 10])
    loader = DataLoader(subset_dataset, batch_size=2, shuffle=False)
    
    # Create model
    model = HybridDisentanglingTransformer(
        input_channels=6,
        d_model=256,
        nhead=8,
        num_transformer_layers=6,
        num_cnn_layers=4,
        num_joints=17,
        num_exercise_types=10,
        num_form_classes=5,
        dropout=0.1
    )
    
    # Create loss
    loss_fn = MultiTaskLoss()
    
    # Get one batch
    batch = next(iter(loader))
    
    print("\n📊 Input shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    # Prepare inputs
    imu_data = batch['imu_data']
    poses = batch['poses']
    exercise_labels = batch['exercise_label'].squeeze()
    form_labels = batch['form_label'].squeeze()
    
    print(f"\n🔧 Processed input shapes:")
    print(f"  imu_data: {imu_data.shape}")
    print(f"  poses: {poses.shape}")
    print(f"  exercise_labels: {exercise_labels.shape}")
    print(f"  form_labels: {form_labels.shape}")
    
    # Forward pass
    print(f"\n🚀 Model forward pass...")
    outputs = model(imu_data)
    
    print(f"\n📤 Model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Reshape poses for loss computation
    poses_reshaped = poses.view(poses.shape[0], poses.shape[1], 17, 3)
    print(f"\n🔄 Reshaped poses: {poses_reshaped.shape}")
    
    # Prepare for loss
    predictions = {
        'pose_mean': outputs['pose_mean'],
        'pose_log_var': outputs['pose_log_var'],
        'exercise_logits': outputs['exercise_logits'],
        'form_logits': outputs['form_logits'],
        'uncertainty': outputs['uncertainty']
    }
    
    targets = {
        'poses': poses_reshaped,
        'exercise_labels': exercise_labels,
        'form_labels': form_labels
    }
    
    print(f"\n🎯 Loss function inputs:")
    print("Predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    print("Targets:")
    for key, value in targets.items():
        print(f"  {key}: {value.shape}")
    
    # Try loss computation
    try:
        print(f"\n🧮 Computing loss...")
        loss_dict = loss_fn(predictions, targets)
        print(f"✅ Loss computation successful!")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        
        # Debug GaussianNLLLoss specifically
        print(f"\n🔍 Debugging GaussianNLLLoss...")
        try:
            pose_loss = loss_fn.pose_loss_fn(
                predictions['pose_mean'],
                predictions['pose_log_var'],
                targets['poses']
            )
            print(f"✅ Pose loss: {pose_loss.item():.4f}")
        except Exception as pose_e:
            print(f"❌ Pose loss failed: {pose_e}")
            print(f"  pose_mean: {predictions['pose_mean'].shape}")
            print(f"  pose_log_var: {predictions['pose_log_var'].shape}")
            print(f"  target poses: {targets['poses'].shape}")

if __name__ == "__main__":
    debug_tensor_shapes()

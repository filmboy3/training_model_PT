"""
Smoke test for QuantumLeap Pose Engine production training pipeline.
Executes 1-epoch dry run to validate end-to-end integrity before full training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np
from pathlib import Path
import logging
import time

from src.models.hdt_architecture import HybridDisentanglingTransformer
from src.models.losses import MultiTaskLoss
from simple_dataset_loader import ProductionDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def smoke_test():
    """Execute smoke test - 1 epoch dry run."""
    logger.info("🧪 Starting QuantumLeap Smoke Test...")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Initialize wandb for smoke test
    wandb.init(
        project="quantumleap-pose-engine",
        name="smoke-test-1-epoch",
        tags=["smoke-test", "validation", "1-epoch"],
        notes="Smoke test to validate end-to-end pipeline integrity"
    )
    
    try:
        # 1. Load dataset
        logger.info("📊 Loading production dataset...")
        dataset = ProductionDataset("data/production_squats_10k.h5", max_seq_length=200)
        
        # Small subset for smoke test (100 sequences)
        subset_size = 100
        subset_dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
        
        # Train/val split
        train_size = int(0.8 * subset_size)
        val_size = subset_size - train_size
        train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        logger.info(f"Dataset loaded: {train_size} train, {val_size} val samples")
        
        # 2. Create model
        logger.info("🏗️ Creating HDT model...")
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
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created: {total_params:,} parameters")
        
        # 3. Create loss and optimizer
        loss_fn = MultiTaskLoss(
            pose_weight=1.0,
            exercise_weight=0.5,
            form_weight=0.3,
            uncertainty_weight=0.1
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # 4. Training step test
        logger.info("🚀 Testing training step...")
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            imu_data = batch['imu_data'].to(device)
            poses = batch['poses'].to(device).view(batch['poses'].shape[0], batch['poses'].shape[1], 17, 3)
            exercise_labels = batch['exercise_label'].squeeze().to(device)
            form_labels = batch['form_label'].squeeze().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(imu_data)
            
            # Compute loss
            predictions = {
                'pose_mean': outputs['pose_mean'],
                'pose_log_var': outputs['pose_log_var'],
                'exercise_logits': outputs['exercise_logits'],
                'form_logits': outputs['form_logits'],
                'uncertainty': outputs['uncertainty']
            }
            
            targets = {
                'poses': poses,
                'exercise_labels': exercise_labels,
                'form_labels': form_labels
            }
            
            loss_dict = loss_fn(predictions, targets)
            
            total_loss = loss_dict['total_loss']
            
            # Check for NaN/inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.error(f"❌ Invalid loss detected: {total_loss}")
                return False
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(total_loss.item())
            
            # Log first few batches
            if batch_idx < 3:
                logger.info(f"Batch {batch_idx}: Loss = {total_loss.item():.4f}")
            
            # Break after 5 batches for smoke test
            if batch_idx >= 4:
                break
        
        avg_train_loss = np.mean(train_losses)
        logger.info(f"✅ Training step successful: Avg loss = {avg_train_loss:.4f}")
        
        # 5. Validation step test
        logger.info("🔍 Testing validation step...")
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move to device
                imu_data = batch['imu_data'].to(device)
                poses = batch['poses'].to(device).view(batch['poses'].shape[0], batch['poses'].shape[1], 17, 3)
                exercise_labels = batch['exercise_label'].squeeze().to(device)
                form_labels = batch['form_label'].squeeze().to(device)
                
                # Forward pass
                outputs = model(imu_data)
                
                # Compute loss
                predictions = {
                    'pose_mean': outputs['pose_mean'],
                    'pose_log_var': outputs['pose_log_var'],
                    'exercise_logits': outputs['exercise_logits'],
                    'form_logits': outputs['form_logits'],
                    'uncertainty': outputs['uncertainty']
                }
                
                targets = {
                    'poses': poses,
                    'exercise_labels': exercise_labels,
                    'form_labels': form_labels
                }
                
                loss_dict = loss_fn(predictions, targets)
                
                total_loss = loss_dict['total_loss']
                
                # Check for NaN/inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.error(f"❌ Invalid validation loss: {total_loss}")
                    return False
                
                val_losses.append(total_loss.item())
                
                # Break after 2 batches for smoke test
                if batch_idx >= 1:
                    break
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f"✅ Validation step successful: Avg loss = {avg_val_loss:.4f}")
        
        # 6. Log to wandb
        wandb.log({
            "smoke_test/train_loss": avg_train_loss,
            "smoke_test/val_loss": avg_val_loss,
            "smoke_test/model_params": total_params,
            "smoke_test/status": "PASSED"
        })
        
        # 7. Test model outputs
        logger.info("🧠 Testing model output shapes...")
        sample_batch = next(iter(train_loader))
        imu_data = sample_batch['imu_data'].to(device)
        
        with torch.no_grad():
            outputs = model(imu_data)
            
            expected_shapes = {
                'pose_mean': (imu_data.shape[0], imu_data.shape[1], 17, 3),
                'pose_log_var': (imu_data.shape[0], imu_data.shape[1], 17, 3),
                'exercise_logits': (imu_data.shape[0], 10),
                'form_logits': (imu_data.shape[0], 5),
                'uncertainty': (imu_data.shape[0], 1)
            }
            
            for key, expected_shape in expected_shapes.items():
                actual_shape = outputs[key].shape
                if actual_shape != expected_shape:
                    logger.error(f"❌ Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
                    return False
                logger.info(f"✅ {key}: {actual_shape}")
        
        logger.info("🎉 SMOKE TEST PASSED!")
        logger.info("✅ End-to-end pipeline integrity validated")
        logger.info("✅ Data loading successful")
        logger.info("✅ Model forward/backward pass successful")
        logger.info("✅ Loss computation stable (no NaN/inf)")
        logger.info("✅ Wandb logging successful")
        logger.info("✅ Model output shapes correct")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SMOKE TEST FAILED: {e}")
        wandb.log({"smoke_test/status": "FAILED", "smoke_test/error": str(e)})
        return False
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    success = smoke_test()
    
    if success:
        print("\n" + "="*60)
        print("🎉 SMOKE TEST PASSED - READY FOR PRODUCTION TRAINING!")
        print("="*60)
        print("Next steps:")
        print("1. Launch full production training with train_production_model.py")
        print("2. Monitor training progress in Weights & Biases")
        print("3. Validate model performance after training")
        exit(0)
    else:
        print("\n" + "="*60)
        print("❌ SMOKE TEST FAILED - DO NOT PROCEED TO PRODUCTION")
        print("="*60)
        print("Fix the identified issues before launching production training.")
        exit(1)

"""
Simplified production training script for QuantumLeap Pose Engine.
Uses the corrected ProductionDataset and simplified training loop.
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
from tqdm import tqdm

from src.models.hdt_architecture import HybridDisentanglingTransformer
from src.models.losses import MultiTaskLoss
from simple_dataset_loader import ProductionDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        imu_data = batch['imu_data'].to(device)
        poses = batch['poses'].to(device).view(batch['poses'].shape[0], batch['poses'].shape[1], 17, 3)
        exercise_labels = batch['exercise_label'].squeeze().to(device)
        form_labels = batch['form_label'].squeeze().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(imu_data)
        
        # Prepare for loss computation
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
        
        # Compute loss
        loss_dict = loss_fn(predictions, targets)
        total_loss_batch = loss_dict['total_loss']
        
        # Backward pass
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
        
        # Log to wandb every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                'train/batch_loss': total_loss_batch.item(),
                'train/pose_loss': loss_dict['pose_loss'].item(),
                'train/exercise_loss': loss_dict['exercise_loss'].item(),
                'train/form_loss': loss_dict['form_loss'].item(),
                'train/step': epoch * len(train_loader) + batch_idx
            })
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, loss_fn, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move to device
            imu_data = batch['imu_data'].to(device)
            poses = batch['poses'].to(device).view(batch['poses'].shape[0], batch['poses'].shape[1], 17, 3)
            exercise_labels = batch['exercise_label'].squeeze().to(device)
            form_labels = batch['form_label'].squeeze().to(device)
            
            # Forward pass
            outputs = model(imu_data)
            
            # Prepare for loss computation
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
            
            # Compute loss
            loss_dict = loss_fn(predictions, targets)
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
    
    return total_loss / num_batches

def main():
    """Main training function."""
    logger.info("🚀 Starting QuantumLeap Production Training...")
    
    # Configuration
    config = {
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'sequence_length': 200,
        'num_workers': 4,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Initialize wandb
    wandb.init(
        project="quantumleap-pose-engine",
        name="production-training-10k-simplified",
        tags=["production", "10k-dataset", "hdt", "mujoco"],
        notes="Full production training on 10K synthetic squat dataset with fixed HDT architecture",
        config=config
    )
    
    try:
        # Load dataset
        logger.info("📊 Loading production dataset...")
        dataset = ProductionDataset("data/production_squats_10k.h5", max_seq_length=config['sequence_length'])
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(config['train_split'] * total_size)
        val_size = int(config['val_split'] * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers']
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        
        # Create model
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
        logger.info(f"Model parameters: {total_params:,}")
        
        # Create loss and optimizer
        loss_fn = MultiTaskLoss(
            pose_weight=1.0,
            exercise_weight=0.5,
            form_weight=0.3,
            uncertainty_weight=0.1
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['num_epochs']
        )
        
        # Training loop
        logger.info("🎯 Starting training loop...")
        best_val_loss = float('inf')
        
        for epoch in range(config['num_epochs']):
            # Train
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
            
            # Validate
            val_loss = validate_epoch(model, val_loader, loss_fn, device)
            
            # Update scheduler
            scheduler.step()
            
            # Log epoch results
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'val/epoch_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, 'models/best_model.pth')
                logger.info(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, f'models/checkpoint_epoch_{epoch+1}.pth')
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()

"""
Production training script for QuantumLeap Pose Engine.
Trains HDT model on large-scale MuJoCo synthetic dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, Any

from src.models.hdt_architecture import HybridDisentanglingTransformer
from simple_dataset_loader import ProductionDataset
# from src.training.trainer import QuantumLeapTrainer  # Not needed for this simplified version
from src.models.losses import MultiTaskLoss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_training_config() -> Dict[str, Any]:
    """Setup comprehensive training configuration."""
    return {
        # Model architecture
        'model': {
            'input_dim': 51,           # Pose vector dimension
            'imu_dim': 6,              # IMU sensor dimension
            'hidden_dim': 256,         # Transformer hidden dimension
            'num_heads': 8,            # Multi-head attention
            'num_layers': 6,           # Transformer layers
            'num_exercises': 10,       # Exercise classification classes
            'num_form_classes': 5,     # Form quality classes
            'dropout': 0.1,            # Dropout rate
            'max_seq_length': 200      # Maximum sequence length
        },
        
        # Training parameters
        'training': {
            'batch_size': 32,          # Batch size
            'learning_rate': 1e-4,     # Learning rate
            'weight_decay': 1e-5,      # L2 regularization
            'num_epochs': 100,         # Training epochs
            'warmup_epochs': 10,       # Learning rate warmup
            'patience': 15,            # Early stopping patience
            'gradient_clip': 1.0,      # Gradient clipping
            'accumulation_steps': 1    # Gradient accumulation
        },
        
        # Loss weights
        'loss_weights': {
            'pose_weight': 1.0,        # Pose reconstruction loss
            'exercise_weight': 0.5,    # Exercise classification loss
            'form_weight': 0.3,        # Form quality loss
            'uncertainty_weight': 0.1   # Uncertainty regularization
        },
        
        # Data parameters
        'data': {
            'dataset_path': 'data/production_squats_10k.h5',
            'train_split': 0.8,        # 80% training
            'val_split': 0.1,          # 10% validation
            'test_split': 0.1,         # 10% testing
            'num_workers': 4,          # DataLoader workers
            'pin_memory': True,        # Pin memory for GPU
            'shuffle': True            # Shuffle training data
        },
        
        # Experiment tracking
        'wandb': {
            'project': 'quantumleap-pose-engine',
            'name': 'production-training-10k',
            'tags': ['mujoco', 'synthetic', 'production', '10k-dataset'],
            'notes': 'Production training on 10K MuJoCo synthetic squat sequences'
        },
        
        # Checkpointing
        'checkpoint': {
            'save_dir': 'checkpoints/production',
            'save_frequency': 5,       # Save every 5 epochs
            'keep_best': True,         # Keep best model
            'metric': 'val_total_loss' # Metric for best model
        }
    }

def create_model(config: Dict[str, Any]) -> HybridDisentanglingTransformer:
    """Create HDT model with production configuration."""
    model_config = config['model']
    
    model = HybridDisentanglingTransformer(
        input_channels=model_config['imu_dim'],
        d_model=model_config['hidden_dim'],
        nhead=model_config['num_heads'],
        num_transformer_layers=model_config['num_layers'],
        num_cnn_layers=4,
        num_joints=17,
        num_exercise_types=model_config['num_exercises'],
        num_form_classes=model_config['num_form_classes'],
        dropout=model_config['dropout']
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model

def create_datasets(config: Dict[str, Any]):
    """Create train, validation, and test datasets."""
    logger.info("Creating datasets...")
    
    # Dataset path
    dataset_path = Path("data/production_squats_10k.h5")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Load full dataset using ProductionDataset
    full_dataset = ProductionDataset(
        data_path=str(dataset_path),
        max_seq_length=config['model']['sequence_length']
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config['data']['train_split'] * total_size)
    val_size = int(config['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, config: Dict[str, Any]):
    """Create data loaders."""
    data_config = config['data']
    training_config = config['training']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=data_config['shuffle'],
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory']
    )
    
    return train_loader, val_loader, test_loader

def main():
    """Main training function."""
    logger.info("Starting QuantumLeap Pose Engine production training...")
    
    # Setup configuration
    config = setup_training_config()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Wait for dataset generation to complete
    dataset_path = Path(config['data']['dataset_path'])
    if not dataset_path.exists():
        logger.info("Waiting for dataset generation to complete...")
        while not dataset_path.exists():
            time.sleep(10)
        logger.info("Dataset found! Proceeding with training...")
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        tags=config['wandb']['tags'],
        notes=config['wandb']['notes'],
        config=config
    )
    
    try:
        # Create model
        logger.info("Creating HDT model...")
        model = create_model(config)
        model = model.to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset, val_dataset, test_dataset = create_datasets(config)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, config
        )
        
        # Create loss function
        loss_fn = MultiTaskLoss(
            pose_weight=config['loss_weights']['pose_weight'],
            exercise_weight=config['loss_weights']['exercise_weight'],
            form_weight=config['loss_weights']['form_weight'],
            uncertainty_weight=config['loss_weights']['uncertainty_weight']
        )
        
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Create trainer
        trainer = QuantumLeapTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config
        )
        
        # Create checkpoint directory
        checkpoint_dir = Path(config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Start training
        logger.info("Starting training...")
        best_model = trainer.train(
            num_epochs=config['training']['num_epochs'],
            checkpoint_dir=checkpoint_dir,
            patience=config['training']['patience']
        )
        
        # Final evaluation
        logger.info("Running final evaluation...")
        test_metrics = trainer.evaluate(test_loader, split='test')
        
        logger.info("Training completed successfully!")
        logger.info(f"Final test metrics: {test_metrics}")
        
        # Save final model
        final_model_path = checkpoint_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'test_metrics': test_metrics
        }, final_model_path)
        
        logger.info(f"Final model saved to: {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()

"""
Training pipeline for QuantumLeap Pose Engine.
Implements the complete training loop with experiment tracking and model management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import time
import json
from collections import defaultdict
import wandb

from ..models import HybridDisentanglingTransformer, MultiTaskLoss, create_hdt_model
from .config import TrainingConfig
from .dataset import QuantumLeapDataModule, collate_fn

logger = logging.getLogger(__name__)


class QuantumLeapTrainer:
    """
    Complete training pipeline for QuantumLeap Pose Engine.
    Handles model training, validation, checkpointing, and experiment tracking.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.system.device)
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize loss function
        self.loss_fn = self._create_loss_function()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.optimization.use_amp else None
        
        # Initialize data module
        self.data_module = QuantumLeapDataModule(
            config.data, 
            config.system.data_dir
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Initialize experiment tracking
        self._init_experiment_tracking()
        
        # Create output directories
        self.checkpoint_dir = Path(config.system.checkpoint_dir) / config.experiment.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"QuantumLeap Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.system.seed)
        np.random.seed(self.config.system.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.system.seed)
            torch.cuda.manual_seed_all(self.config.system.seed)
        
        if self.config.system.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif self.config.system.benchmark:
            torch.backends.cudnn.benchmark = True
    
    def _create_model(self) -> HybridDisentanglingTransformer:
        """Create and initialize the HDT model."""
        model = create_hdt_model(self.config.model.__dict__)
        model = model.to(self.device)
        
        # Enable channels_last memory format if requested
        if self.config.system.channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        # Compile model if using PyTorch 2.0+
        if self.config.system.compile_model and hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        return model
    
    def _create_loss_function(self) -> MultiTaskLoss:
        """Create the multi-task loss function."""
        return MultiTaskLoss(
            pose_weight=self.config.optimization.pose_weight,
            exercise_weight=self.config.optimization.exercise_weight,
            form_weight=self.config.optimization.form_weight,
            uncertainty_weight=self.config.optimization.uncertainty_weight,
            learnable_weights=self.config.optimization.learnable_task_weights
        ).to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        if self.config.optimization.optimizer.lower() == 'adamw':
            return optim.AdamW(
                list(self.model.parameters()) + list(self.loss_fn.parameters()),
                lr=self.config.optimization.learning_rate,
                weight_decay=self.config.optimization.weight_decay,
                betas=self.config.optimization.betas,
                eps=self.config.optimization.eps
            )
        elif self.config.optimization.optimizer.lower() == 'adam':
            return optim.Adam(
                list(self.model.parameters()) + list(self.loss_fn.parameters()),
                lr=self.config.optimization.learning_rate,
                weight_decay=self.config.optimization.weight_decay,
                betas=self.config.optimization.betas,
                eps=self.config.optimization.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimization.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.optimization.scheduler.lower() == 'cosine_annealing':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.optimization.max_epochs,
                eta_min=self.config.optimization.min_lr
            )
        elif self.config.optimization.scheduler.lower() == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def _init_experiment_tracking(self):
        """Initialize Weights & Biases experiment tracking."""
        if self.config.experiment.use_wandb:
            wandb.init(
                project=self.config.experiment.wandb_project,
                entity=self.config.experiment.wandb_entity,
                name=self.config.experiment.run_name or self.config.experiment.experiment_name,
                tags=self.config.experiment.wandb_tags,
                config=self.config.to_dict()
            )
            wandb.watch(self.model, log='all', log_freq=1000)
    
    def train(self):
        """Main training loop."""
        logger.info("Starting QuantumLeap training...")
        
        # Prepare data
        self.data_module.prepare_data()
        self.data_module.setup('fit')
        
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.optimization.max_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            if epoch % (self.config.experiment.val_interval // len(train_loader)) == 0:
                val_metrics = self._validate_epoch(val_loader)
                
                # Check for improvement
                current_metric = val_metrics[self.config.experiment.monitor_metric]
                if self._is_better_metric(current_metric, self.best_metric):
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.optimization.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.config.experiment.monitor_metric, 0))
                else:
                    self.scheduler.step()
            
            # Save periodic checkpoint
            if epoch % (self.config.experiment.save_interval // len(train_loader)) == 0:
                self._save_checkpoint(is_best=False)
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
        
        logger.info("Training completed!")
        
        # Final evaluation
        self._final_evaluation()
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            with autocast(enabled=self.config.optimization.use_amp):
                predictions = self.model(batch['imu_data'])
                
                # Prepare targets
                targets = {
                    'poses': batch['joint_positions'],
                    'exercise_labels': batch['exercise_label'],
                    'form_labels': batch['form_label']
                }
                
                # Compute loss
                losses = self.loss_fn(predictions, targets)
                total_loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.optimization.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.optimization.max_grad_norm
                )
                self.optimizer.step()
            
            # Update metrics
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key].append(value.item())
            
            self.global_step += 1
            
            # Log training metrics
            if batch_idx % self.config.experiment.log_interval == 0:
                self._log_training_step(losses, batch_idx, len(train_loader))
        
        # Average metrics over epoch
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch['imu_data'])
                
                # Prepare targets
                targets = {
                    'poses': batch['joint_positions'],
                    'exercise_labels': batch['exercise_label'],
                    'form_labels': batch['form_label']
                }
                
                # Compute loss
                losses = self.loss_fn(predictions, targets)
                
                # Update metrics
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        epoch_metrics[key].append(value.item())
                
                # Compute additional validation metrics
                val_metrics = self._compute_validation_metrics(predictions, targets)
                for key, value in val_metrics.items():
                    epoch_metrics[key].append(value)
        
        # Average metrics over epoch
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def _compute_validation_metrics(self, 
                                   predictions: Dict[str, torch.Tensor],
                                   targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute additional validation metrics."""
        metrics = {}
        
        # Pose estimation metrics
        if 'pose_mean' in predictions and 'poses' in targets:
            pose_error = torch.norm(
                predictions['pose_mean'] - targets['poses'], 
                dim=-1
            )  # [batch, seq_len, joints]
            
            # Apply mask
            masked_error = pose_error * masks.unsqueeze(-1)
            valid_elements = masks.unsqueeze(-1).sum()
            
            metrics['pose_mpjpe'] = (masked_error.sum() / valid_elements).item() * 1000  # mm
            metrics['pose_pck'] = (masked_error < 0.05).float().mean().item()  # PCK@50mm
        
        # Exercise classification accuracy
        if 'exercise_logits' in predictions and 'exercise_labels' in targets:
            exercise_pred = torch.argmax(predictions['exercise_logits'], dim=-1)
            exercise_acc = (exercise_pred == targets['exercise_labels']).float().mean()
            metrics['exercise_accuracy'] = exercise_acc.item()
        
        # Form classification accuracy
        if 'form_logits' in predictions and 'form_labels' in targets:
            form_pred = torch.argmax(predictions['form_logits'], dim=-1)
            form_acc = (form_pred == targets['form_labels']).float().mean()
            metrics['form_accuracy'] = form_acc.item()
        
        # Uncertainty calibration
        if 'uncertainty' in predictions:
            avg_uncertainty = predictions['uncertainty'].mean()
            metrics['avg_uncertainty'] = avg_uncertainty.item()
        
        return metrics
    
    def _is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.config.experiment.mode == 'min':
            return current < best - self.config.optimization.min_delta
        else:
            return current > best + self.config.optimization.min_delta
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {self.best_metric:.6f}")
        
        # Keep only top-k checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only top-k."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > self.config.experiment.save_top_k:
            # Sort by modification time and remove oldest
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for checkpoint in checkpoints[:-self.config.experiment.save_top_k]:
                checkpoint.unlink()
    
    def _log_training_step(self, losses: Dict[str, torch.Tensor], batch_idx: int, total_batches: int):
        """Log training step metrics."""
        if self.config.experiment.use_wandb:
            log_dict = {f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v 
                       for k, v in losses.items()}
            log_dict['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
            log_dict['train/epoch'] = self.current_epoch
            log_dict['train/step'] = self.global_step
            
            wandb.log(log_dict, step=self.global_step)
        
        if batch_idx % (self.config.experiment.log_interval * 10) == 0:
            logger.info(
                f"Epoch {self.current_epoch} [{batch_idx}/{total_batches}] "
                f"Loss: {losses['total_loss'].item():.6f}"
            )
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int):
        """Log epoch metrics."""
        if self.config.experiment.use_wandb:
            log_dict = {}
            log_dict.update({f"train_epoch/{k}": v for k, v in train_metrics.items()})
            log_dict.update({f"val_epoch/{k}": v for k, v in val_metrics.items()})
            log_dict['epoch'] = epoch
            
            # Log task weights if learnable
            if self.config.optimization.learnable_task_weights:
                task_weights = self.loss_fn.get_task_weights()
                log_dict.update({f"weights/{k}": v for k, v in task_weights.items()})
            
            wandb.log(log_dict, step=self.global_step)
        
        # Store metrics
        for key, value in train_metrics.items():
            self.train_metrics[key].append(value)
        for key, value in val_metrics.items():
            self.val_metrics[key].append(value)
        
        logger.info(
            f"Epoch {epoch} - Train Loss: {train_metrics.get('total_loss', 0):.6f}, "
            f"Val Loss: {val_metrics.get('total_loss', 0):.6f}, "
            f"Val MPJPE: {val_metrics.get('pose_mpjpe', 0):.2f}mm"
        )
    
    def _final_evaluation(self):
        """Perform final evaluation on test set."""
        logger.info("Performing final evaluation...")
        
        # Load best model
        best_checkpoint = self.checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Setup test data
        self.data_module.setup('test')
        test_loader = self.data_module.test_dataloader()
        
        # Evaluate
        test_metrics = self._validate_epoch(test_loader)
        
        logger.info("Final test metrics:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.6f}")
        
        if self.config.experiment.use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


if __name__ == "__main__":
    # Test trainer setup
    from .config import get_mvm_config
    
    config = get_mvm_config()
    config.data.num_sequences = 100  # Small test dataset
    config.optimization.max_epochs = 2
    config.experiment.use_wandb = False
    
    trainer = QuantumLeapTrainer(config)
    
    print("Trainer initialized successfully!")
    print(f"Model device: {next(trainer.model.parameters()).device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # Test single forward pass
    batch_size, seq_len, input_channels = 4, 100, 6
    test_input = torch.randn(batch_size, seq_len, input_channels).to(trainer.device)
    test_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(trainer.device)
    
    with torch.no_grad():
        predictions = trainer.model(test_input, mask=test_mask)
    
    print("Forward pass test successful!")
    print(f"Predictions keys: {list(predictions.keys())}")

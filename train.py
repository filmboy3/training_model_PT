#!/usr/bin/env python3
"""
Main training script for QuantumLeap Pose Engine.
Run this script to train the HDT model on synthetic physics-based data.

Usage:
    python train.py --config mvm                    # Train MVM (squats only)
    python train.py --config full                   # Train full multi-exercise model
    python train.py --config ablation_no_physics    # Ablation study
    python train.py --resume checkpoints/best_model.pt  # Resume from checkpoint
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.training import QuantumLeapTrainer
from src.training.config import (
    get_mvm_config, 
    get_full_config, 
    get_ablation_config,
    TrainingConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train QuantumLeap Pose Engine")
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='mvm',
        choices=['mvm', 'full', 'ablation_no_physics', 'ablation_no_disentangling', 'ablation_deterministic'],
        help='Training configuration preset'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to custom config JSON file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory for dataset storage'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Directory for outputs and logs'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory for model checkpoints'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='quantumleap-pose-engine',
        help='Weights & Biases project name'
    )
    
    parser.add_argument(
        '--wandb-entity',
        type=str,
        help='Weights & Biases entity/username'
    )
    
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (small dataset, few epochs)'
    )
    
    return parser.parse_args()


def get_config(args) -> TrainingConfig:
    """Get training configuration based on arguments."""
    if args.config_file:
        # Load custom config from file
        config = TrainingConfig.load(args.config_file)
    else:
        # Use preset configuration
        if args.config == 'mvm':
            config = get_mvm_config()
        elif args.config == 'full':
            config = get_full_config()
        elif args.config.startswith('ablation_'):
            ablation_type = args.config.replace('ablation_', '')
            config = get_ablation_config(ablation_type)
        else:
            raise ValueError(f"Unknown config: {args.config}")
    
    # Override with command line arguments
    if args.data_dir:
        config.system.data_dir = args.data_dir
    if args.output_dir:
        config.system.output_dir = args.output_dir
    if args.checkpoint_dir:
        config.system.checkpoint_dir = args.checkpoint_dir
    if args.wandb_project:
        config.experiment.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.experiment.wandb_entity = args.wandb_entity
    if args.no_wandb:
        config.experiment.use_wandb = False
    
    # Debug mode adjustments
    if args.debug:
        config.data.num_sequences = 500
        config.data.batch_size = 8
        config.optimization.max_epochs = 5
        config.experiment.log_interval = 10
        config.experiment.val_interval = 50
        config.experiment.save_interval = 100
        config.experiment.experiment_name += "_debug"
        logger.info("Debug mode enabled - using small dataset and few epochs")
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("QuantumLeap Pose Engine Training")
    logger.info("Physics-First AI for Human Pose Estimation")
    logger.info("=" * 80)
    
    # Get configuration
    config = get_config(args)
    
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Experiment: {config.experiment.experiment_name}")
    logger.info(f"Device: {config.system.device}")
    logger.info(f"Model: HDT with {config.model.d_model}d, {config.model.num_transformer_layers} layers")
    logger.info(f"Dataset: {config.data.num_sequences} sequences")
    logger.info(f"Batch size: {config.data.batch_size}")
    logger.info(f"Max epochs: {config.optimization.max_epochs}")
    
    # Save configuration
    config_save_path = Path(config.system.output_dir) / config.experiment.experiment_name / "config.json"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(config_save_path))
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Initialize trainer
    trainer = QuantumLeapTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from: {args.resume}")
    
    try:
        # Start training
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save current state
        checkpoint_path = Path(config.system.checkpoint_dir) / config.experiment.experiment_name / "interrupted.pt"
        trainer._save_checkpoint()
        logger.info(f"Saved interrupted checkpoint to: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if config.experiment.use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()

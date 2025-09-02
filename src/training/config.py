"""
Training configuration for QuantumLeap Pose Engine.
Centralized configuration management for experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """HDT model architecture configuration."""
    input_channels: int = 6
    d_model: int = 256
    nhead: int = 8
    num_transformer_layers: int = 6
    num_cnn_layers: int = 4
    num_joints: int = 17
    num_exercise_types: int = 3  # squat, pushup, overhead_press for MVM
    num_form_classes: int = 5    # excellent, good, fair, poor, dangerous
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Data generation and loading configuration."""
    # Physics simulation
    num_sequences: int = 10000
    sequence_length: int = 200
    dt: float = 0.01  # 100Hz sampling
    
    # Exercise parameters
    squat_depth_range: tuple = (0.3, 0.8)
    speed_variation: float = 0.3
    duration_range: tuple = (3.0, 8.0)
    
    # Domain randomization - expanded ranges for Phase 2
    physics_randomization: bool = True
    sensor_noise: bool = True
    placement_randomization: bool = True
    
    # Enhanced DR parameters (for future expansion)
    gravity_range: tuple = (9.7, 9.9)  # TODO: Expand to (9.5, 10.0) in Phase 2
    body_mass_range: tuple = (0.9, 1.1)  # TODO: Expand to (0.7, 1.3) in Phase 2
    imu_noise_range: tuple = (0.01, 0.1)  # TODO: Expand to (0.005, 0.2) in Phase 2
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Augmentation
    temporal_jitter: bool = True
    amplitude_scaling: bool = True
    rotation_augmentation: bool = True


@dataclass
class OptimizationConfig:
    """Optimization and training configuration."""
    # Optimizer
    optimizer: str = 'adamw'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate scheduling
    scheduler: str = 'cosine_annealing'
    warmup_epochs: int = 10
    max_epochs: int = 200
    min_lr: float = 1e-6
    
    # Loss weights
    pose_weight: float = 1.0
    exercise_weight: float = 0.5
    form_weight: float = 0.3
    uncertainty_weight: float = 0.1
    physics_constraint_weight: float = 0.05
    learnable_task_weights: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4


@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration."""
    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "quantumleap-pose-engine"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["mvm", "squat", "physics-first"])
    
    # Experiment naming
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    
    # Logging
    log_interval: int = 100
    val_interval: int = 1000
    save_interval: int = 5000
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val_pose_loss"
    mode: str = "min"
    
    # Visualization
    log_pose_samples: bool = True
    log_attention_maps: bool = True
    log_feature_distributions: bool = True
    num_vis_samples: int = 4


@dataclass
class SystemConfig:
    """System and hardware configuration."""
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    
    # Performance
    compile_model: bool = False  # PyTorch 2.0 compilation
    channels_last: bool = False


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Auto-detect device
        if self.system.device == "auto":
            if torch.cuda.is_available():
                self.system.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.system.device = "mps"
            else:
                self.system.device = "cpu"
        
        # Create directories
        os.makedirs(self.system.data_dir, exist_ok=True)
        os.makedirs(self.system.output_dir, exist_ok=True)
        os.makedirs(self.system.checkpoint_dir, exist_ok=True)
        
        # Set experiment name if not provided
        if self.experiment.experiment_name is None:
            self.experiment.experiment_name = f"hdt_d{self.model.d_model}_l{self.model.num_transformer_layers}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'optimization': self.optimization.__dict__,
            'experiment': self.experiment.__dict__,
            'system': self.system.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            optimization=OptimizationConfig(**config_dict.get('optimization', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different experiments
def get_mvm_config() -> TrainingConfig:
    """Get configuration for Minimum Viable Model (MVM) training."""
    config = TrainingConfig()
    
    # MVM-specific settings
    config.model.num_exercise_types = 1  # Only squats
    config.data.num_sequences = 5000     # Smaller dataset for faster iteration
    config.optimization.max_epochs = 100
    config.experiment.wandb_tags = ["mvm", "squat-only", "physics-first"]
    config.experiment.experiment_name = "mvm_squat_baseline"
    
    return config


def get_full_config() -> TrainingConfig:
    """Get configuration for full multi-exercise training."""
    config = TrainingConfig()
    
    # Full model settings
    config.model.num_exercise_types = 10  # Multiple exercises
    config.data.num_sequences = 50000     # Large dataset
    config.optimization.max_epochs = 300
    config.experiment.wandb_tags = ["full-model", "multi-exercise", "physics-first"]
    config.experiment.experiment_name = "full_multi_exercise"
    
    return config


def get_ablation_config(ablation_type: str) -> TrainingConfig:
    """Get configuration for ablation studies."""
    config = TrainingConfig()
    
    if ablation_type == "no_physics":
        config.data.physics_randomization = False
        config.optimization.physics_constraint_weight = 0.0
        config.experiment.wandb_tags = ["ablation", "no-physics"]
        config.experiment.experiment_name = "ablation_no_physics"
    
    elif ablation_type == "no_disentangling":
        # Would need model architecture changes
        config.experiment.wandb_tags = ["ablation", "no-disentangling"]
        config.experiment.experiment_name = "ablation_no_disentangling"
    
    elif ablation_type == "deterministic":
        # Remove probabilistic components
        config.experiment.wandb_tags = ["ablation", "deterministic"]
        config.experiment.experiment_name = "ablation_deterministic"
    
    return config


if __name__ == "__main__":
    # Test configuration creation and serialization
    config = get_mvm_config()
    
    print("MVM Configuration:")
    print(f"Device: {config.system.device}")
    print(f"Model d_model: {config.model.d_model}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.optimization.learning_rate}")
    print(f"Max epochs: {config.optimization.max_epochs}")
    print(f"Experiment name: {config.experiment.experiment_name}")
    
    # Test serialization
    config.save("test_config.json")
    loaded_config = TrainingConfig.load("test_config.json")
    
    print(f"\nSerialization test passed: {config.to_dict() == loaded_config.to_dict()}")
    
    # Clean up
    os.remove("test_config.json")

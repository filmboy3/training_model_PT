# QuantumLeap Pose Engine

**Physics-First AI for Human Pose Estimation**

## Mission

QuantumLeap Pose Engine delivers state-of-the-art human pose estimation using a physics-first approach combined with probabilistic modeling. Our goal is to surpass MobilePoser benchmarks while providing uncertainty-aware predictions critical for fitness and tele-rehabilitation applications.

## 🎯 Implementation Status

### ✅ **Phase 1: Core Architecture (COMPLETED)**

**Hybrid Disentangling Transformer (HDT)**
- ✅ CNN frontend for IMU temporal feature extraction
- ✅ Disentangling transformer encoder (separates "what" from "how")
- ✅ Multi-task probabilistic decoder
- ✅ **6.1M parameters, 23.3MB model size**

**Probabilistic Modeling**
- ✅ Gaussian NLL loss for uncertainty quantification
- ✅ Multi-task loss with learnable task weighting
- ✅ Physics constraint loss for biomechanical plausibility
- ✅ Calibration loss for uncertainty calibration

**Physics-Based Data Generation**
- ✅ MuJoCo physics engine integration
- ✅ Biomechanically accurate squat motion synthesis
- ✅ Domain randomization for sim-to-real transfer
- ✅ IMU sensor simulation with realistic noise

**Training Infrastructure**
- ✅ Complete training pipeline with Weights & Biases integration
- ✅ Experiment configuration system
- ✅ Multi-GPU support and mixed precision training
- ✅ Automated checkpointing and model management

### 🔄 **Phase 2: Training & Validation (NEXT)**

- ⏳ Generate large-scale synthetic squat dataset (10K+ sequences)
- ⏳ Train Minimum Viable Model (MVM) on squats
- ⏳ iOS data logger integration for real-world validation
- ⏳ Benchmark against MobilePoser on DIP-IMU dataset

### 📱 **Phase 3: Deployment (PLANNED)**

- ⏳ Core ML conversion and optimization
- ⏳ Swift SDK development
- ⏳ Real-time inference optimization
- ⏳ Multi-exercise expansion

## Core Differentiators

1. **Physics-First Data Generation**: Uses MuJoCo physics simulator to generate large synthetic datasets of biomechanically accurate human motion
2. **Probabilistic Modeling**: Outputs pose estimates with uncertainty quantification using Gaussian negative log-likelihood loss
3. **Hybrid Disentangling Transformer (HDT)**: Novel architecture that separates exercise classification ("what") from form analysis ("how")
4. **Domain Randomization**: Aggressive sim-to-real transfer through physics and sensor randomization
5. **Multi-Task Learning**: Simultaneous pose estimation, exercise recognition, and form quality assessment

## Architecture Features

✅ **Hybrid CNN + Transformer (HDT) architecture**  
✅ **Disentangling encoder separates 'what' from 'how'**  
✅ **Probabilistic pose decoder with uncertainty quantification**  
✅ **Multi-task learning (pose + exercise + form)**  
✅ **Gaussian NLL loss for probabilistic training**  
✅ **Physics constraint loss for biomechanical plausibility**  
✅ **Calibration loss for uncertainty calibration**  
✅ **Learnable task weighting for multi-task optimization**  

## Performance Targets

- **Accuracy**: >95% pose estimation accuracy on DIP-IMU dataset
- **Latency**: <50ms inference on mobile devices (iPhone/Apple Watch)
- **Uncertainty**: Well-calibrated confidence intervals for safety-critical applications
- **Generalization**: Robust performance across different users, environments, and sensor placements

## Project Structure

```
QuantumLeap_Pose_Engine/
├── src/
│   ├── data/              # Physics simulation and data generation
│   │   ├── physics_engine.py      # MuJoCo-based motion synthesis
│   │   ├── domain_randomization.py # Sim-to-real transfer
│   │   └── imu_simulator.py       # Realistic sensor simulation
│   ├── models/            # HDT architecture and loss functions
│   │   ├── hdt_architecture.py    # Core HDT model
│   │   ├── probabilistic_decoder.py # Uncertainty quantification
│   │   └── losses.py              # Multi-task loss functions
│   ├── training/          # Training pipeline and experiment management
│   │   ├── trainer.py             # Complete training loop
│   │   ├── dataset.py             # Data loading and augmentation
│   │   └── config.py              # Experiment configuration
│   └── utils/             # Utilities and helper functions
├── experiments/           # Experiment configurations and results
├── notebooks/            # Jupyter notebooks for analysis
├── ios_sdk/              # iOS SDK for deployment
├── docs/                 # Documentation
├── tests/                # Unit tests
├── train.py              # Main training script
├── test_minimal.py       # Architecture validation
└── requirements.txt      # Python dependencies
```

## Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repo-url>
cd QuantumLeap_Pose_Engine

# Install dependencies
pip install -r requirements.txt

# Install MuJoCo (for physics simulation)
pip install mujoco
```

### 2. Validate Architecture
```bash
# Test core architecture (no MuJoCo required)
python test_minimal.py
```

### 3. Generate Synthetic Data
```bash
# Generate small test dataset
python -c "
from src.data import PhysicsDataEngine
engine = PhysicsDataEngine()
engine.generate_dataset(1000, 200, 'data/test_squats.h5')
"
```

### 4. Train Model
```bash
# Train MVM (debug mode)
python train.py --config mvm --debug

# Train full model
python train.py --config mvm
```

### 5. Monitor Training
```bash
# View logs
tail -f training.log

# Open Weights & Biases dashboard
# https://wandb.ai/your-project/quantumleap-pose-engine
```

## Configuration Presets

- **`mvm`**: Minimum Viable Model (squats only, 5K sequences)
- **`full`**: Full multi-exercise model (50K sequences)
- **`ablation_no_physics`**: Ablation study without physics constraints
- **`ablation_deterministic`**: Ablation study without probabilistic modeling

## Model Architecture

The HDT (Hybrid Disentangling Transformer) consists of:

1. **CNN Frontend**: 1D convolutions for temporal IMU feature extraction
2. **Disentangling Transformer**: Separates content (exercise type) from style (form quality)
3. **Multi-Task Decoder**: Outputs pose mean/variance, exercise classification, and form assessment
4. **Uncertainty Estimation**: Global uncertainty quantification for safety-critical applications

**Model Specifications:**
- Parameters: 6,107,375 (6.1M)
- Model Size: 23.3MB (FP32)
- Input: 6-channel IMU data (accel + gyro)
- Output: 17-joint 3D poses with uncertainty

## Training Features

- **Mixed Precision Training**: Faster training with automatic loss scaling
- **Gradient Clipping**: Stable training with large batch sizes
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Early Stopping**: Automatic training termination based on validation metrics
- **Multi-GPU Support**: Distributed training for faster convergence
- **Experiment Tracking**: Weights & Biases integration with automatic logging

## Next Steps

1. **Install MuJoCo**: `pip install mujoco`
2. **Generate Dataset**: Create large-scale synthetic squat dataset
3. **Train MVM**: `python train.py --config mvm --debug`
4. **Real-World Validation**: Integrate iOS data logger
5. **Benchmark**: Compare against MobilePoser on DIP-IMU dataset

## License

Proprietary - All rights reserved

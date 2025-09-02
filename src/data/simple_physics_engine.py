"""
Simplified physics engine for QuantumLeap Pose Engine testing.
Uses a basic MuJoCo model that works reliably.
"""

import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    logger.warning("MuJoCo not available, using synthetic data generation")


class SimplePhysicsEngine:
    """
    Simplified physics engine for testing QuantumLeap Pose Engine.
    Generates synthetic squat motion data.
    """
    
    def __init__(self):
        self.dt = 0.01  # 100Hz simulation
        
    def generate_synthetic_squat_sequence(self, 
                                        sequence_length: int = 200,
                                        squat_depth: float = 0.5,
                                        speed_variation: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a synthetic squat sequence without MuJoCo.
        
        Args:
            sequence_length: Number of timesteps
            squat_depth: Depth of squat (0.3 to 0.8)
            speed_variation: Speed variation factor
            
        Returns:
            Tuple of (joint_positions, joint_rotations, imu_data)
        """
        # Time array
        t = np.linspace(0, sequence_length * self.dt, sequence_length)
        
        # Generate squat motion pattern
        squat_phase = np.sin(2 * np.pi * t / (sequence_length * self.dt * 0.5)) * squat_depth
        squat_phase = np.maximum(squat_phase, 0)  # Only downward motion
        
        # Add speed variation
        speed_factor = 1.0 + speed_variation * np.sin(4 * np.pi * t / (sequence_length * self.dt))
        squat_phase *= speed_factor
        
        # Generate joint positions (17 joints, 3D positions)
        joint_positions = np.zeros((sequence_length, 17, 3))
        
        # Define basic skeleton structure
        # 0: pelvis, 1: spine, 2: head, 3-4: shoulders, 5-6: elbows, 7-8: wrists
        # 9-10: hips, 11-12: knees, 13-14: ankles, 15-16: feet
        
        for i, phase in enumerate(squat_phase):
            # Pelvis (root joint)
            joint_positions[i, 0] = [0, 0, 1.0 - phase]  # Lower during squat
            
            # Spine
            joint_positions[i, 1] = [0, 0, 1.2 - phase * 0.8]
            
            # Head
            joint_positions[i, 2] = [0, 0, 1.6 - phase * 0.8]
            
            # Shoulders
            joint_positions[i, 3] = [-0.2, 0, 1.4 - phase * 0.8]  # Left shoulder
            joint_positions[i, 4] = [0.2, 0, 1.4 - phase * 0.8]   # Right shoulder
            
            # Elbows
            joint_positions[i, 5] = [-0.4, 0, 1.2 - phase * 0.8]  # Left elbow
            joint_positions[i, 6] = [0.4, 0, 1.2 - phase * 0.8]   # Right elbow
            
            # Wrists
            joint_positions[i, 7] = [-0.6, 0, 1.0 - phase * 0.8]  # Left wrist
            joint_positions[i, 8] = [0.6, 0, 1.0 - phase * 0.8]   # Right wrist
            
            # Hips
            joint_positions[i, 9] = [-0.1, 0, 0.9 - phase]   # Left hip
            joint_positions[i, 10] = [0.1, 0, 0.9 - phase]   # Right hip
            
            # Knees (bend more during squat)
            knee_bend = phase * 1.5
            joint_positions[i, 11] = [-0.1, knee_bend, 0.5 - phase * 0.3]  # Left knee
            joint_positions[i, 12] = [0.1, knee_bend, 0.5 - phase * 0.3]   # Right knee
            
            # Ankles
            joint_positions[i, 13] = [-0.1, knee_bend * 0.5, 0.1]  # Left ankle
            joint_positions[i, 14] = [0.1, knee_bend * 0.5, 0.1]   # Right ankle
            
            # Feet
            joint_positions[i, 15] = [-0.1, knee_bend * 0.5, 0.0]  # Left foot
            joint_positions[i, 16] = [0.1, knee_bend * 0.5, 0.0]   # Right foot
        
        # Generate joint rotations (quaternions)
        joint_rotations = np.zeros((sequence_length, 17, 4))
        joint_rotations[:, :, 3] = 1.0  # Identity quaternions (w=1)
        
        # Add some rotation variation for knees and hips during squat
        for i, phase in enumerate(squat_phase):
            # Hip rotations (slight forward lean)
            hip_angle = phase * 0.3
            joint_rotations[i, 9, 0] = np.sin(hip_angle / 2)  # Left hip
            joint_rotations[i, 9, 3] = np.cos(hip_angle / 2)
            joint_rotations[i, 10, 0] = np.sin(hip_angle / 2)  # Right hip
            joint_rotations[i, 10, 3] = np.cos(hip_angle / 2)
            
            # Knee rotations (bend during squat)
            knee_angle = phase * 1.2
            joint_rotations[i, 11, 0] = np.sin(knee_angle / 2)  # Left knee
            joint_rotations[i, 11, 3] = np.cos(knee_angle / 2)
            joint_rotations[i, 12, 0] = np.sin(knee_angle / 2)  # Right knee
            joint_rotations[i, 12, 3] = np.cos(knee_angle / 2)
        
        # Generate IMU data from motion
        imu_data = self._generate_imu_from_motion(joint_positions, joint_rotations)
        
        return joint_positions, joint_rotations, imu_data
    
    def _generate_imu_from_motion(self, positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
        """Generate IMU data from joint motion."""
        sequence_length = positions.shape[0]
        
        # Use pelvis (root joint) for IMU simulation
        pelvis_pos = positions[:, 0, :]  # [seq_len, 3]
        
        # Compute velocities and accelerations
        velocities = np.gradient(pelvis_pos, self.dt, axis=0)
        accelerations = np.gradient(velocities, self.dt, axis=0)
        
        # Add gravity
        accelerations[:, 2] += 9.81
        
        # Compute angular velocities from rotations
        pelvis_rot = rotations[:, 0, :]  # [seq_len, 4] quaternions
        angular_velocities = np.zeros((sequence_length, 3))
        
        for i in range(1, sequence_length):
            # Simple finite difference for angular velocity
            dq = pelvis_rot[i] - pelvis_rot[i-1]
            angular_velocities[i] = dq[:3] / self.dt
        
        # Combine accelerometer and gyroscope data
        imu_data = np.concatenate([accelerations, angular_velocities], axis=1)
        
        # Add realistic noise
        noise_std = 0.1
        imu_data += np.random.normal(0, noise_std, imu_data.shape)
        
        return imu_data
    
    def generate_dataset(self, 
                        num_sequences: int = 1000,
                        sequence_length: int = 200,
                        output_path: str = "data/synthetic_squats.h5") -> None:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_sequences: Number of sequences to generate
            sequence_length: Length of each sequence
            output_path: Path to save HDF5 dataset
        """
        logger.info(f"Generating {num_sequences} synthetic squat sequences...")
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize arrays
        all_joint_positions = []
        all_joint_rotations = []
        all_imu_data = []
        all_exercise_labels = []
        all_form_labels = []
        
        for i in range(num_sequences):
            if i % 100 == 0:
                logger.info(f"Generated {i}/{num_sequences} sequences")
            
            # Randomize squat parameters
            squat_depth = np.random.uniform(0.3, 0.8)
            speed_variation = np.random.uniform(0.1, 0.5)
            
            # Generate sequence
            joint_pos, joint_rot, imu_data = self.generate_synthetic_squat_sequence(
                sequence_length, squat_depth, speed_variation
            )
            
            all_joint_positions.append(joint_pos)
            all_joint_rotations.append(joint_rot)
            all_imu_data.append(imu_data)
            all_exercise_labels.append(0)  # 0 = squat
            all_form_labels.append(2)      # 2 = good form
        
        # Convert to numpy arrays
        all_joint_positions = np.array(all_joint_positions)
        all_joint_rotations = np.array(all_joint_rotations)
        all_imu_data = np.array(all_imu_data)
        all_exercise_labels = np.array(all_exercise_labels)
        all_form_labels = np.array(all_form_labels)
        
        # Save to HDF5
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('joint_positions', data=all_joint_positions)
            f.create_dataset('joint_rotations', data=all_joint_rotations)
            f.create_dataset('imu_data', data=all_imu_data)
            f.create_dataset('exercise_labels', data=all_exercise_labels)
            f.create_dataset('form_labels', data=all_form_labels)
            
            # Add metadata
            f.attrs['num_sequences'] = num_sequences
            f.attrs['sequence_length'] = sequence_length
            f.attrs['dt'] = self.dt
            f.attrs['num_joints'] = 17
            f.attrs['exercise_type'] = 'squat'
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Dataset shape: {all_joint_positions.shape}")


if __name__ == "__main__":
    # Test synthetic data generation
    engine = SimplePhysicsEngine()
    
    # Generate single sequence
    joint_pos, joint_rot, imu_data = engine.generate_synthetic_squat_sequence(200)
    
    print("Synthetic data generation test:")
    print(f"Joint positions shape: {joint_pos.shape}")
    print(f"Joint rotations shape: {joint_rot.shape}")
    print(f"IMU data shape: {imu_data.shape}")
    
    # Generate small dataset
    engine.generate_dataset(100, 200, "test_dataset.h5")
    print("✓ Test dataset generated successfully!")

"""
Physics-based data generation engine for QuantumLeap Pose Engine.
Uses MuJoCo physics simulation to generate biomechanically accurate motion data.
"""

import numpy as np
import mujoco
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysicsDataEngine:
    """
    Physics-first data generation engine for human pose estimation.
    Generates biomechanically accurate motion data using MuJoCo physics simulation.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the physics engine.
        
        Args:
            model_path: Path to MuJoCo model XML file. If None, uses default humanoid model.
        """
        # Model path - use working simple humanoid model
        self.model_path = model_path or str(Path(__file__).parent.parent.parent / "models" / "simple_humanoid.xml")
        self.model = None
        self.data = None
        self._load_model()
        
    def _load_model(self):
        """Load MuJoCo model from XML file."""
        try:
            if Path(self.model_path).exists():
                logger.info(f"Loading MuJoCo model from: {self.model_path}")
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
                self.data = mujoco.MjData(self.model)
                logger.info(f"Model loaded successfully. Bodies: {self.model.nbody}, Joints: {self.model.njnt}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Creating default humanoid model from XML string")
                xml_content = self._get_default_humanoid_xml()
                self.model = mujoco.MjModel.from_xml_string(xml_content)
                self.data = mujoco.MjData(self.model)
        except Exception as e:
            logger.error(f"Failed to load MuJoCo model: {e}")
            raise
    
    def _get_default_humanoid_xml(self) -> str:
        """Get default humanoid model XML for squat simulation."""
        return """
        <mujoco model="humanoid_squat">
            <compiler angle="degree" inertiafromgeom="true"/>
            <default>
                <joint armature="1" damping="1" limited="true"/>
                <geom friction="1 0.1 0.1" rgba="0.8 0.6 0.4 1"/>
            </default>
            
            <worldbody>
                <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
                <geom name="floor" pos="0 0 0" size="40 40 40" type="plane" rgba="0.8 0.9 0.8 1"/>
                
                <body name="torso" pos="0 0 1.4">
                    <geom name="torso" type="capsule" size="0.07 0.1"/>
                    <joint name="root" type="free"/>
                    
                    <!-- Simplified limbs for basic squat motion -->
                    <body name="thigh_l" pos="0.1 0 -0.3">
                        <geom name="thigh_l" type="capsule" size="0.05 0.2"/>
                        <joint name="hip_l" type="hinge" axis="1 0 0" range="-120 30"/>
                        
                        <body name="shin_l" pos="0 0 -0.4">
                            <geom name="shin_l" type="capsule" size="0.04 0.2"/>
                            <joint name="knee_l" type="hinge" axis="1 0 0" range="-150 0"/>
                        </body>
                    </body>
                    
                    <body name="thigh_r" pos="-0.1 0 -0.3">
                        <geom name="thigh_r" type="capsule" size="0.05 0.2"/>
                        <joint name="hip_r" type="hinge" axis="1 0 0" range="-120 30"/>
                        
                        <body name="shin_r" pos="0 0 -0.4">
                            <geom name="shin_r" type="capsule" size="0.04 0.2"/>
                            <joint name="knee_r" type="hinge" axis="1 0 0" range="-150 0"/>
                        </body>
                    </body>
                </body>
            </worldbody>
            
            <actuator>
                <motor joint="hip_l" gear="100"/>
                <motor joint="knee_l" gear="100"/>
                <motor joint="hip_r" gear="100"/>
                <motor joint="knee_r" gear="100"/>
            </actuator>
        </mujoco>
        """
    
    def generate_squat_sequence(self, duration: float = 3.0, timestep: float = 0.01) -> Dict[str, Any]:
        """
        Generate a single squat motion sequence using physics simulation.
        
        Args:
            duration: Duration of squat sequence in seconds
            timestep: Physics simulation timestep
            
        Returns:
            Dictionary containing pose sequence, IMU data, and metadata
        """
        if self.model is None or self.data is None:
            raise RuntimeError("MuJoCo model not loaded")
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose (standing)
        self.data.qpos[2] = 1.4  # torso height
        
        # Generate squat motion
        num_steps = int(duration / timestep)
        poses = []
        imu_data = []
        
        for step in range(num_steps):
            t = step * timestep
            
            # Generate squat motion pattern
            squat_phase = np.sin(2 * np.pi * t / duration)  # One complete squat cycle
            
            # Apply control signals for squat motion
            if hasattr(self.data, 'ctrl') and self.data.ctrl is not None:
                # Hip flexion for squat
                hip_angle = -30 * (1 + squat_phase) / 2  # 0 to -30 degrees
                knee_angle = -60 * (1 + squat_phase) / 2  # 0 to -60 degrees
                
                # Find joint indices
                hip_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_l")
                knee_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_l")
                hip_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip_r")
                knee_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_r")
                
                if hip_l_id >= 0 and len(self.data.ctrl) > 0:
                    self.data.ctrl[0] = hip_angle * 0.1  # Scale for torque
                if knee_l_id >= 0 and len(self.data.ctrl) > 1:
                    self.data.ctrl[1] = knee_angle * 0.1
                if hip_r_id >= 0 and len(self.data.ctrl) > 2:
                    self.data.ctrl[2] = hip_angle * 0.1
                if knee_r_id >= 0 and len(self.data.ctrl) > 3:
                    self.data.ctrl[3] = knee_angle * 0.1
            
            # Step physics simulation
            mujoco.mj_step(self.model, self.data)
            
            # Extract pose data (joint positions and orientations)
            pose_data = self._extract_pose_data()
            poses.append(pose_data)
            
            # Simulate IMU data
            imu_sample = self._simulate_imu_data(t)
            imu_data.append(imu_sample)
        
        return {
            'poses': np.array(poses),
            'imu_data': np.array(imu_data),
            'exercise_label': 0,  # 0 = squat
            'form_label': 1,      # 1 = good form
            'metadata': {
                'duration': duration,
                'timestep': timestep,
                'num_steps': num_steps,
                'exercise_type': 'squat'
            }
        }
    
    def _extract_pose_data(self) -> np.ndarray:
        """Extract pose data from current simulation state."""
        # Extract joint positions and orientations
        joint_positions = self.data.qpos.copy()
        joint_velocities = self.data.qvel.copy()
        
        # Get body positions and orientations
        body_positions = self.data.xpos.copy()
        body_orientations = self.data.xquat.copy()
        
        # Combine into pose vector (simplified for now)
        pose_vector = np.concatenate([
            joint_positions[:min(len(joint_positions), 10)],  # Limit to 10 joints
            joint_velocities[:min(len(joint_velocities), 10)],
            body_positions.flatten()[:min(len(body_positions.flatten()), 20)]  # Limit body positions
        ])
        
        # Pad or truncate to fixed size (e.g., 51 dimensions)
        target_size = 51
        if len(pose_vector) > target_size:
            pose_vector = pose_vector[:target_size]
        elif len(pose_vector) < target_size:
            pose_vector = np.pad(pose_vector, (0, target_size - len(pose_vector)), 'constant')
        
        return pose_vector
    
    def _simulate_imu_data(self, time: float) -> np.ndarray:
        """Simulate IMU sensor data based on current motion state."""
        # Get torso body acceleration and angular velocity
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        
        if torso_id >= 0:
            # Get linear acceleration (approximate from velocity change)
            linear_acc = self.data.cacc[torso_id][:3] if hasattr(self.data, 'cacc') else np.zeros(3)
            # Get angular velocity
            angular_vel = self.data.cvel[torso_id][3:] if hasattr(self.data, 'cvel') else np.zeros(3)
        else:
            # Fallback: simulate based on squat motion
            squat_phase = np.sin(2 * np.pi * time / 3.0)
            linear_acc = np.array([0, 0, -9.81 + squat_phase * 2])  # Gravity + squat motion
            angular_vel = np.array([squat_phase * 0.1, 0, 0])  # Small rotation during squat
        
        # Add realistic noise
        linear_acc += np.random.normal(0, 0.1, 3)
        angular_vel += np.random.normal(0, 0.05, 3)
        
        # Combine into IMU vector [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        imu_vector = np.concatenate([linear_acc, angular_vel])
        
        return imu_vector
    
    def generate_dataset(self, num_sequences: int = 100, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete dataset of squat sequences.
        
        Args:
            num_sequences: Number of squat sequences to generate
            save_path: Optional path to save the dataset
            
        Returns:
            Complete dataset dictionary
        """
        logger.info(f"Generating {num_sequences} squat sequences...")
        
        all_poses = []
        all_imu_data = []
        all_exercise_labels = []
        all_form_labels = []
        
        for i in range(num_sequences):
            if i % 10 == 0:
                logger.info(f"Generated {i}/{num_sequences} sequences")
            
            # Vary squat parameters for diversity
            duration = np.random.uniform(2.5, 4.0)  # Vary squat duration
            
            sequence = self.generate_squat_sequence(duration=duration)
            
            all_poses.append(sequence['poses'])
            all_imu_data.append(sequence['imu_data'])
            all_exercise_labels.append(sequence['exercise_label'])
            all_form_labels.append(sequence['form_label'])
        
        # Create dataset
        dataset = {
            'poses': all_poses,
            'imu_data': all_imu_data,
            'exercise_labels': np.array(all_exercise_labels),
            'form_labels': np.array(all_form_labels),
            'metadata': {
                'num_sequences': num_sequences,
                'exercise_type': 'squat',
                'generation_method': 'mujoco_physics'
            }
        }
        
        # Save if path provided
        if save_path:
            self._save_dataset(dataset, save_path)
        
        logger.info(f"Dataset generation complete: {num_sequences} sequences")
        return dataset
    
    def _save_dataset(self, dataset: Dict[str, Any], save_path: str):
        """Save dataset to file."""
        import h5py
        
        with h5py.File(save_path, 'w') as f:
            # Save pose sequences
            poses_group = f.create_group('poses')
            for i, pose_seq in enumerate(dataset['poses']):
                poses_group.create_dataset(f'sequence_{i}', data=pose_seq)
            
            # Save IMU sequences
            imu_group = f.create_group('imu_data')
            for i, imu_seq in enumerate(dataset['imu_data']):
                imu_group.create_dataset(f'sequence_{i}', data=imu_seq)
            
            # Save labels
            f.create_dataset('exercise_labels', data=dataset['exercise_labels'])
            f.create_dataset('form_labels', data=dataset['form_labels'])
            
            # Save metadata
            metadata_group = f.create_group('metadata')
            for key, value in dataset['metadata'].items():
                if isinstance(value, str):
                    metadata_group.attrs[key] = value
                else:
                    metadata_group.attrs[key] = value
        
        logger.info(f"Dataset saved to: {save_path}")
    
    def test_simulation(self, steps: int = 100) -> bool:
        """Test basic simulation functionality."""
        try:
            logger.info("Testing MuJoCo simulation...")
            
            # Reset simulation
            mujoco.mj_resetData(self.model, self.data)
            
            # Set initial pose
            self.data.qpos[2] = 1.4  # torso height
            
            # Run simulation steps
            for i in range(steps):
                # Apply simple control
                if hasattr(self.data, 'ctrl') and len(self.data.ctrl) > 0:
                    # Simple squat motion
                    phase = np.sin(2 * np.pi * i / steps)
                    for j in range(min(4, len(self.data.ctrl))):
                        self.data.ctrl[j] = phase * 0.1
                
                mujoco.mj_step(self.model, self.data)
                
                # Check for instability
                if np.any(np.isnan(self.data.qpos)) or np.any(np.abs(self.data.qpos) > 100):
                    logger.error("Simulation became unstable")
                    return False
            
            logger.info(f"Simulation test successful: {steps} steps completed")
            return True
            
        except Exception as e:
            logger.error(f"Simulation test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            "model_path": self.model_path,
            "num_bodies": self.model.nbody,
            "num_joints": self.model.njnt,
            "num_actuators": self.model.nu,
            "num_dofs": self.model.nv,
            "body_names": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) 
                          for i in range(self.model.nbody)],
            "joint_names": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                           for i in range(self.model.njnt)]
        }


def main():
    """Test the physics engine."""
    try:
        # Initialize physics engine
        engine = PhysicsDataEngine()
        
        # Print model info
        info = engine.get_model_info()
        print("Model Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test simulation
        if engine.test_simulation(steps=50):
            print("\n✅ Simulation test passed!")
            
            # Generate a sample sequence
            print("\nGenerating sample squat sequence...")
            sequence = engine.generate_squat_sequence(duration=2.0)
            
            print(f"Generated sequence:")
            print(f"  Poses shape: {sequence['poses'].shape}")
            print(f"  IMU data shape: {sequence['imu_data'].shape}")
            print(f"  Exercise label: {sequence['exercise_label']}")
            print(f"  Form label: {sequence['form_label']}")
            
        else:
            print("\n❌ Simulation test failed!")
            
    except Exception as e:
        print(f"❌ Physics engine test failed: {e}")


if __name__ == "__main__":
    main()

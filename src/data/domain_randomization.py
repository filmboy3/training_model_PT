"""
Domain Randomization module for robust sim-to-real transfer.
Implements aggressive randomization of physics, sensor characteristics, and placement.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DomainRandomizer:
    """
    Implements aggressive domain randomization to bridge the sim-to-real gap.
    This is critical for ensuring our physics-first approach generalizes to real IMU data.
    """
    
    def __init__(self, randomization_config: Optional[Dict] = None):
        """
        Initialize domain randomizer with configuration.
        
        Args:
            randomization_config: Configuration dict for randomization parameters
        """
        self.config = randomization_config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default aggressive domain randomization configuration."""
        return {
            'physics': {
                'gravity_variation': 0.1,  # ±10% gravity variation
                'body_mass_variation': 0.3,  # ±30% mass variation
                'limb_length_variation': 0.15,  # ±15% limb length variation
                'joint_friction_variation': 0.5,  # ±50% joint friction
                'ground_friction_variation': 0.3,  # ±30% ground friction
            },
            'sensor': {
                'bias_drift_std': 0.02,  # Accelerometer bias drift (m/s²)
                'scale_factor_variation': 0.02,  # ±2% scale factor variation
                'axis_misalignment_deg': 2.0,  # ±2° axis misalignment
                'temperature_drift_rate': 0.001,  # Temperature drift rate
                'sampling_jitter_std': 0.001,  # Sampling time jitter (seconds)
                'packet_drop_rate': 0.01,  # 1% packet drop rate
                'noise_std_accel': 0.1,  # Accelerometer noise (m/s²)
                'noise_std_gyro': 0.01,  # Gyroscope noise (rad/s)
            },
            'placement': {
                'position_offset_std': 0.05,  # ±5cm position offset
                'rotation_offset_deg': 15.0,  # ±15° rotation offset
                'attachment_looseness': 0.02,  # Simulated attachment looseness
            }
        }
    
    def randomize_physics_params(self, base_params: Dict) -> Dict:
        """
        Apply physics domain randomization.
        
        Args:
            base_params: Base physics parameters
            
        Returns:
            Randomized physics parameters
        """
        randomized = base_params.copy()
        config = self.config['physics']
        
        # Gravity variation
        gravity_factor = 1.0 + np.random.normal(0, config['gravity_variation'])
        randomized['gravity'] = randomized.get('gravity', 9.81) * gravity_factor
        
        # Body mass variation
        if 'body_masses' in randomized:
            mass_factors = 1.0 + np.random.normal(0, config['body_mass_variation'], 
                                                 len(randomized['body_masses']))
            randomized['body_masses'] = randomized['body_masses'] * mass_factors
        
        # Limb length variation
        if 'limb_lengths' in randomized:
            length_factors = 1.0 + np.random.normal(0, config['limb_length_variation'],
                                                   len(randomized['limb_lengths']))
            randomized['limb_lengths'] = randomized['limb_lengths'] * length_factors
        
        # Joint friction variation
        if 'joint_friction' in randomized:
            friction_factors = 1.0 + np.random.normal(0, config['joint_friction_variation'],
                                                     len(randomized['joint_friction']))
            randomized['joint_friction'] = np.maximum(0.001, 
                                                     randomized['joint_friction'] * friction_factors)
        
        # Ground friction variation
        ground_friction_factor = 1.0 + np.random.normal(0, config['ground_friction_variation'])
        randomized['ground_friction'] = randomized.get('ground_friction', 1.0) * ground_friction_factor
        
        return randomized
    
    def simulate_sensor_noise(self, clean_imu_data: np.ndarray, 
                            duration: float) -> np.ndarray:
        """
        Apply realistic sensor noise and artifacts to clean IMU data.
        
        Args:
            clean_imu_data: Clean IMU data [T, 6] (accel_xyz, gyro_xyz)
            duration: Duration of the sequence in seconds
            
        Returns:
            Noisy IMU data with realistic sensor artifacts
        """
        T, channels = clean_imu_data.shape
        assert channels == 6, "Expected 6-channel IMU data (accel + gyro)"
        
        config = self.config['sensor']
        noisy_data = clean_imu_data.clone()
        
        # Split into accelerometer and gyroscope
        accel_data = noisy_data[:, :3]
        gyro_data = noisy_data[:, 3:]
        
        # 1. Bias drift (slowly varying bias)
        accel_bias = self._generate_bias_drift(T, 3, config['bias_drift_std'], duration)
        gyro_bias = self._generate_bias_drift(T, 3, config['bias_drift_std'] * 0.1, duration)
        
        accel_data += accel_bias
        gyro_data += gyro_bias
        
        # 2. Scale factor variation
        accel_scale = 1.0 + np.random.normal(0, config['scale_factor_variation'], 3)
        gyro_scale = 1.0 + np.random.normal(0, config['scale_factor_variation'], 3)
        
        accel_data *= accel_scale[None, :]
        gyro_data *= gyro_scale[None, :]
        
        # 3. Axis misalignment
        accel_data = self._apply_axis_misalignment(accel_data, config['axis_misalignment_deg'])
        gyro_data = self._apply_axis_misalignment(gyro_data, config['axis_misalignment_deg'])
        
        # 4. Temperature drift
        temp_drift_accel = self._generate_temperature_drift(T, 3, config['temperature_drift_rate'])
        temp_drift_gyro = self._generate_temperature_drift(T, 3, config['temperature_drift_rate'])
        
        accel_data += temp_drift_accel
        gyro_data += temp_drift_gyro
        
        # 5. White noise
        accel_noise = np.random.normal(0, config['noise_std_accel'], accel_data.shape)
        gyro_noise = np.random.normal(0, config['noise_std_gyro'], gyro_data.shape)
        
        accel_data += accel_noise
        gyro_data += gyro_noise
        
        # 6. Sampling jitter and packet drops
        noisy_data = np.concatenate([accel_data, gyro_data], axis=1)
        noisy_data = self._apply_sampling_artifacts(noisy_data, config)
        
        return noisy_data
    
    def randomize_sensor_placement(self, base_position: np.ndarray, 
                                 base_rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomize sensor placement to simulate real-world mounting variations.
        
        Args:
            base_position: Base sensor position [3]
            base_rotation: Base sensor rotation matrix [3, 3]
            
        Returns:
            Tuple of (randomized_position, randomized_rotation)
        """
        config = self.config['placement']
        
        # Position offset
        position_offset = np.random.normal(0, config['position_offset_std'], 3)
        new_position = base_position + position_offset
        
        # Rotation offset
        rotation_offset_deg = np.random.normal(0, config['rotation_offset_deg'], 3)
        rotation_offset_rad = np.deg2rad(rotation_offset_deg)
        
        # Create rotation matrices for each axis
        Rx = self._rotation_matrix_x(rotation_offset_rad[0])
        Ry = self._rotation_matrix_y(rotation_offset_rad[1])
        Rz = self._rotation_matrix_z(rotation_offset_rad[2])
        
        # Apply combined rotation
        offset_rotation = Rz @ Ry @ Rx
        new_rotation = offset_rotation @ base_rotation
        
        return new_position, new_rotation
    
    def _generate_bias_drift(self, T: int, channels: int, std: float, duration: float) -> np.ndarray:
        """Generate slowly varying bias drift."""
        # Generate low-frequency noise for bias drift
        freq_cutoff = 0.1  # Hz
        t = np.linspace(0, duration, T)
        
        bias_drift = np.zeros((T, channels))
        for c in range(channels):
            # Generate random walk with low-pass filtering
            white_noise = np.random.normal(0, std, T)
            # Simple exponential smoothing for low-frequency drift
            alpha = freq_cutoff * 2 * np.pi * (duration / T)
            for i in range(1, T):
                bias_drift[i, c] = (1 - alpha) * bias_drift[i-1, c] + alpha * white_noise[i]
        
        return bias_drift
    
    def _generate_temperature_drift(self, T: int, channels: int, drift_rate: float) -> np.ndarray:
        """Generate temperature-induced drift."""
        # Simulate temperature variation over time
        t = np.linspace(0, 1, T)
        temp_variation = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.05 * t)
        
        # Apply temperature drift
        drift = np.zeros((T, channels))
        for c in range(channels):
            drift_coeff = np.random.normal(0, drift_rate)
            drift[:, c] = drift_coeff * temp_variation
        
        return drift
    
    def _apply_axis_misalignment(self, data: np.ndarray, misalignment_deg: float) -> np.ndarray:
        """Apply axis misalignment to sensor data."""
        # Generate small random rotation for misalignment
        angles_deg = np.random.normal(0, misalignment_deg, 3)
        angles_rad = np.deg2rad(angles_deg)
        
        Rx = self._rotation_matrix_x(angles_rad[0])
        Ry = self._rotation_matrix_y(angles_rad[1])
        Rz = self._rotation_matrix_z(angles_rad[2])
        
        R_misalign = Rz @ Ry @ Rx
        
        # Apply rotation to each timestep
        aligned_data = np.zeros_like(data)
        for t in range(data.shape[0]):
            aligned_data[t] = R_misalign @ data[t]
        
        return aligned_data
    
    def _apply_sampling_artifacts(self, data: np.ndarray, config: Dict) -> np.ndarray:
        """Apply sampling jitter and packet drops."""
        T, channels = data.shape
        
        # Packet drops
        drop_mask = np.random.random(T) > config['packet_drop_rate']
        
        # For dropped packets, use linear interpolation
        for c in range(channels):
            valid_indices = np.where(drop_mask)[0]
            if len(valid_indices) < T:
                data[:, c] = np.interp(np.arange(T), valid_indices, data[valid_indices, c])
        
        # Sampling jitter (simulate irregular sampling)
        # This is simplified - in practice, would need to resample at irregular intervals
        jitter = np.random.normal(0, config['sampling_jitter_std'], T)
        
        return data
    
    def _rotation_matrix_x(self, angle: float) -> np.ndarray:
        """Create rotation matrix around X axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
    
    def _rotation_matrix_y(self, angle: float) -> np.ndarray:
        """Create rotation matrix around Y axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
    
    def _rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Create rotation matrix around Z axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])


if __name__ == "__main__":
    # Test domain randomization
    randomizer = DomainRandomizer()
    
    # Test physics randomization
    base_physics = {
        'gravity': 9.81,
        'body_masses': np.array([70.0, 15.0, 10.0]),  # kg
        'limb_lengths': np.array([0.45, 0.45, 0.25]),  # m
        'joint_friction': np.array([0.1, 0.1, 0.05]),
        'ground_friction': 1.0
    }
    
    randomized_physics = randomizer.randomize_physics_params(base_physics)
    print("Physics randomization test:")
    print(f"Original gravity: {base_physics['gravity']:.3f}")
    print(f"Randomized gravity: {randomized_physics['gravity']:.3f}")
    
    # Test sensor noise
    clean_imu = np.random.randn(1000, 6)  # 10 seconds at 100Hz
    noisy_imu = randomizer.simulate_sensor_noise(clean_imu, duration=10.0)
    
    print(f"\nSensor noise test:")
    print(f"Clean IMU std: {np.std(clean_imu, axis=0)}")
    print(f"Noisy IMU std: {np.std(noisy_imu, axis=0)}")
    
    # Test placement randomization
    base_pos = np.array([0.0, 0.0, 0.1])
    base_rot = np.eye(3)
    
    rand_pos, rand_rot = randomizer.randomize_sensor_placement(base_pos, base_rot)
    print(f"\nPlacement randomization test:")
    print(f"Position offset: {rand_pos - base_pos}")
    print(f"Rotation determinant: {np.linalg.det(rand_rot):.6f}")  # Should be ~1.0

"""
IMU Simulator for generating realistic sensor data from physics simulation.
Handles sensor placement, coordinate transformations, and realistic noise models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(__name__)


class IMUSimulator:
    """
    Simulates realistic IMU sensor data from physics simulation.
    Handles multiple sensor placements and realistic noise characteristics.
    """
    
    def __init__(self, 
                 sampling_rate: float = 100.0,
                 sensor_placements: Optional[Dict] = None):
        """
        Initialize IMU simulator.
        
        Args:
            sampling_rate: IMU sampling rate in Hz
            sensor_placements: Dictionary defining sensor placement configurations
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # Default sensor placements (phone in pocket, watch on wrist)
        self.sensor_placements = sensor_placements or {
            'phone': {
                'body': 'torso',
                'position_offset': np.array([0.1, 0.0, -0.2]),  # Front pocket
                'rotation_offset': np.array([0, 0, 0]),  # degrees
                'attachment_type': 'loose'  # loose, tight, rigid
            },
            'watch': {
                'body': 'forearm_left',
                'position_offset': np.array([0.0, 0.0, -0.1]),  # Wrist position
                'rotation_offset': np.array([0, 0, 0]),
                'attachment_type': 'tight'
            }
        }
        
        # Sensor characteristics
        self.sensor_specs = {
            'phone': {
                'accel_range': 16.0,  # ±16g
                'gyro_range': 2000.0,  # ±2000 dps
                'accel_noise_density': 150e-6,  # μg/√Hz
                'gyro_noise_density': 0.01,  # dps/√Hz
                'accel_bias_stability': 0.5e-3,  # mg
                'gyro_bias_stability': 10.0,  # dps/hr
            },
            'watch': {
                'accel_range': 8.0,  # ±8g
                'gyro_range': 1000.0,  # ±1000 dps
                'accel_noise_density': 200e-6,  # μg/√Hz
                'gyro_noise_density': 0.015,  # dps/√Hz
                'accel_bias_stability': 1.0e-3,  # mg
                'gyro_bias_stability': 15.0,  # dps/hr
            }
        }
    
    def simulate_imu_data(self,
                         body_states: Dict[str, Dict],
                         duration: float,
                         sensor_name: str = 'phone') -> Dict[str, np.ndarray]:
        """
        Simulate IMU data for a specific sensor from body motion states.
        
        Args:
            body_states: Dictionary containing body motion data
                        {body_name: {'position': [T,3], 'rotation': [T,4], 'velocity': [T,3], 'angular_velocity': [T,3]}}
            duration: Duration of simulation in seconds
            sensor_name: Name of sensor configuration to use
            
        Returns:
            Dictionary containing simulated IMU data
        """
        if sensor_name not in self.sensor_placements:
            raise ValueError(f"Unknown sensor placement: {sensor_name}")
        
        placement = self.sensor_placements[sensor_name]
        specs = self.sensor_specs[sensor_name]
        
        # Get the body this sensor is attached to
        body_name = placement['body']
        if body_name not in body_states:
            raise ValueError(f"Body '{body_name}' not found in body_states")
        
        body_data = body_states[body_name]
        
        # Transform to sensor coordinate frame
        sensor_data = self._transform_to_sensor_frame(body_data, placement)
        
        # Generate clean IMU measurements
        clean_imu = self._generate_clean_imu(sensor_data, duration)
        
        # Add realistic sensor noise and artifacts
        noisy_imu = self._add_sensor_noise(clean_imu, specs, duration)
        
        # Generate timestamps
        n_samples = clean_imu.shape[0]
        timestamps = np.linspace(0, duration, n_samples)
        
        return {
            'accelerometer': noisy_imu[:, :3],  # m/s²
            'gyroscope': noisy_imu[:, 3:],     # rad/s
            'timestamps': timestamps,
            'sampling_rate': self.sampling_rate,
            'sensor_name': sensor_name,
            'clean_data': clean_imu  # For debugging/analysis
        }
    
    def _transform_to_sensor_frame(self, 
                                  body_data: Dict, 
                                  placement: Dict) -> Dict:
        """
        Transform body motion data to sensor coordinate frame.
        
        Args:
            body_data: Body motion data
            placement: Sensor placement configuration
            
        Returns:
            Transformed sensor motion data
        """
        positions = body_data['position']  # [T, 3]
        rotations = body_data['rotation']  # [T, 4] quaternions
        
        T = positions.shape[0]
        
        # Apply position offset
        pos_offset = placement['position_offset']
        rot_offset_deg = placement['rotation_offset']
        
        # Convert rotation offset to rotation matrix
        rot_offset = R.from_euler('xyz', rot_offset_deg, degrees=True)
        
        # Transform positions and orientations
        sensor_positions = np.zeros_like(positions)
        sensor_rotations = np.zeros_like(rotations)
        
        for t in range(T):
            # Body rotation at time t
            body_rot = R.from_quat(rotations[t])
            
            # Apply sensor mounting offset
            sensor_rot = body_rot * rot_offset
            sensor_rotations[t] = sensor_rot.as_quat()
            
            # Transform position offset to world frame
            world_offset = body_rot.apply(pos_offset)
            sensor_positions[t] = positions[t] + world_offset
        
        # Compute velocities and accelerations
        sensor_velocities = self._compute_velocity(sensor_positions)
        sensor_accelerations = self._compute_acceleration(sensor_velocities)
        sensor_angular_velocities = self._compute_angular_velocity(sensor_rotations)
        
        return {
            'position': sensor_positions,
            'rotation': sensor_rotations,
            'velocity': sensor_velocities,
            'acceleration': sensor_accelerations,
            'angular_velocity': sensor_angular_velocities
        }
    
    def _generate_clean_imu(self, sensor_data: Dict, duration: float) -> np.ndarray:
        """
        Generate clean IMU measurements from sensor motion data.
        
        Args:
            sensor_data: Sensor motion data in sensor frame
            duration: Duration in seconds
            
        Returns:
            Clean IMU data [T, 6] (accel_xyz, gyro_xyz)
        """
        accelerations = sensor_data['acceleration']  # [T, 3]
        angular_velocities = sensor_data['angular_velocity']  # [T, 3]
        rotations = sensor_data['rotation']  # [T, 4]
        
        T = accelerations.shape[0]
        
        # Transform accelerations to sensor body frame and add gravity
        imu_accelerations = np.zeros_like(accelerations)
        gravity_world = np.array([0, 0, -9.81])  # Gravity in world frame
        
        for t in range(T):
            # Get sensor orientation
            sensor_rot = R.from_quat(rotations[t])
            
            # Transform world acceleration to sensor frame
            world_accel = accelerations[t]
            sensor_accel = sensor_rot.inv().apply(world_accel)
            
            # Add gravity in sensor frame
            gravity_sensor = sensor_rot.inv().apply(gravity_world)
            imu_accelerations[t] = sensor_accel - gravity_sensor  # IMU measures specific force
        
        # Angular velocities are already in sensor frame
        imu_gyroscope = angular_velocities.copy()
        
        # Combine into 6-DOF IMU data
        clean_imu = np.concatenate([imu_accelerations, imu_gyroscope], axis=1)
        
        return clean_imu
    
    def _add_sensor_noise(self, 
                         clean_imu: np.ndarray, 
                         specs: Dict, 
                         duration: float) -> np.ndarray:
        """
        Add realistic sensor noise based on sensor specifications.
        
        Args:
            clean_imu: Clean IMU data [T, 6]
            specs: Sensor specifications
            duration: Duration in seconds
            
        Returns:
            Noisy IMU data
        """
        T = clean_imu.shape[0]
        noisy_imu = clean_imu.copy()
        
        # Accelerometer noise
        accel_noise_std = specs['accel_noise_density'] * np.sqrt(self.sampling_rate) * 9.81  # Convert to m/s²
        accel_noise = np.random.normal(0, accel_noise_std, (T, 3))
        noisy_imu[:, :3] += accel_noise
        
        # Gyroscope noise
        gyro_noise_std = np.deg2rad(specs['gyro_noise_density'] * np.sqrt(self.sampling_rate))  # Convert to rad/s
        gyro_noise = np.random.normal(0, gyro_noise_std, (T, 3))
        noisy_imu[:, 3:] += gyro_noise
        
        # Bias stability (slowly varying bias)
        accel_bias_std = specs['accel_bias_stability'] * 9.81 / 1000  # Convert mg to m/s²
        gyro_bias_std = np.deg2rad(specs['gyro_bias_stability'] / 3600)  # Convert dps/hr to rad/s
        
        # Generate slowly varying bias
        accel_bias = self._generate_bias_walk(T, 3, accel_bias_std, duration)
        gyro_bias = self._generate_bias_walk(T, 3, gyro_bias_std, duration)
        
        noisy_imu[:, :3] += accel_bias
        noisy_imu[:, 3:] += gyro_bias
        
        # Apply sensor range limits (clipping)
        accel_limit = specs['accel_range'] * 9.81  # Convert g to m/s²
        gyro_limit = np.deg2rad(specs['gyro_range'])  # Convert dps to rad/s
        
        noisy_imu[:, :3] = np.clip(noisy_imu[:, :3], -accel_limit, accel_limit)
        noisy_imu[:, 3:] = np.clip(noisy_imu[:, 3:], -gyro_limit, gyro_limit)
        
        return noisy_imu
    
    def _compute_velocity(self, positions: np.ndarray) -> np.ndarray:
        """Compute velocity from positions using finite differences."""
        velocities = np.zeros_like(positions)
        
        # Forward difference for first point
        velocities[0] = (positions[1] - positions[0]) / self.dt
        
        # Central difference for middle points
        for t in range(1, len(positions) - 1):
            velocities[t] = (positions[t + 1] - positions[t - 1]) / (2 * self.dt)
        
        # Backward difference for last point
        velocities[-1] = (positions[-1] - positions[-2]) / self.dt
        
        return velocities
    
    def _compute_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        """Compute acceleration from velocities using finite differences."""
        accelerations = np.zeros_like(velocities)
        
        # Forward difference for first point
        accelerations[0] = (velocities[1] - velocities[0]) / self.dt
        
        # Central difference for middle points
        for t in range(1, len(velocities) - 1):
            accelerations[t] = (velocities[t + 1] - velocities[t - 1]) / (2 * self.dt)
        
        # Backward difference for last point
        accelerations[-1] = (velocities[-1] - velocities[-2]) / self.dt
        
        return accelerations
    
    def _compute_angular_velocity(self, quaternions: np.ndarray) -> np.ndarray:
        """Compute angular velocity from quaternion sequence."""
        T = quaternions.shape[0]
        angular_velocities = np.zeros((T, 3))
        
        for t in range(T - 1):
            q1 = R.from_quat(quaternions[t])
            q2 = R.from_quat(quaternions[t + 1])
            
            # Compute relative rotation
            q_rel = q2 * q1.inv()
            
            # Convert to axis-angle and scale by time
            axis_angle = q_rel.as_rotvec()
            angular_velocities[t] = axis_angle / self.dt
        
        # Use last computed value for final timestep
        angular_velocities[-1] = angular_velocities[-2]
        
        return angular_velocities
    
    def _generate_bias_walk(self, T: int, channels: int, std: float, duration: float) -> np.ndarray:
        """Generate random walk bias for sensor drift."""
        # Generate random walk
        steps = np.random.normal(0, std * np.sqrt(self.dt), (T, channels))
        bias = np.cumsum(steps, axis=0)
        
        # Apply low-pass filter to make it more realistic
        alpha = 0.01  # Low-pass filter coefficient
        filtered_bias = np.zeros_like(bias)
        
        for t in range(1, T):
            filtered_bias[t] = alpha * bias[t] + (1 - alpha) * filtered_bias[t - 1]
        
        return filtered_bias


def simulate_multi_sensor_imu(body_states: Dict[str, Dict],
                             duration: float,
                             sensor_configs: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Simulate IMU data for multiple sensors simultaneously.
    
    Args:
        body_states: Body motion data
        duration: Duration in seconds
        sensor_configs: List of sensor configurations to simulate
        
    Returns:
        Dictionary of IMU data for each sensor
    """
    if sensor_configs is None:
        sensor_configs = ['phone', 'watch']
    
    simulator = IMUSimulator()
    multi_sensor_data = {}
    
    for sensor_name in sensor_configs:
        try:
            imu_data = simulator.simulate_imu_data(body_states, duration, sensor_name)
            multi_sensor_data[sensor_name] = imu_data
            logger.info(f"Generated IMU data for {sensor_name}")
        except Exception as e:
            logger.warning(f"Failed to generate IMU data for {sensor_name}: {e}")
    
    return multi_sensor_data


if __name__ == "__main__":
    # Test IMU simulation
    simulator = IMUSimulator()
    
    # Create dummy body motion data
    duration = 5.0
    T = int(duration * 100)  # 100 Hz
    
    # Simple sinusoidal motion for testing
    t = np.linspace(0, duration, T)
    positions = np.column_stack([
        0.1 * np.sin(2 * np.pi * 0.5 * t),  # x
        np.zeros(T),                        # y
        1.0 + 0.2 * np.sin(2 * np.pi * 1.0 * t)  # z (vertical motion)
    ])
    
    # Simple rotation (quaternions)
    angles = 0.1 * np.sin(2 * np.pi * 0.3 * t)
    rotations = np.column_stack([
        np.zeros(T),  # qx
        np.zeros(T),  # qy
        np.sin(angles / 2),  # qz
        np.cos(angles / 2)   # qw
    ])
    
    body_states = {
        'torso': {
            'position': positions,
            'rotation': rotations
        }
    }
    
    # Simulate phone IMU
    imu_data = simulator.simulate_imu_data(body_states, duration, 'phone')
    
    print(f"Generated IMU data:")
    print(f"Accelerometer shape: {imu_data['accelerometer'].shape}")
    print(f"Gyroscope shape: {imu_data['gyroscope'].shape}")
    print(f"Accelerometer stats: mean={np.mean(imu_data['accelerometer'], axis=0):.3f}, std={np.std(imu_data['accelerometer'], axis=0):.3f}")
    print(f"Gyroscope stats: mean={np.mean(imu_data['gyroscope'], axis=0):.3f}, std={np.std(imu_data['gyroscope'], axis=0):.3f}")

"""
Extract sample synthetic squat data in human-readable format for auditing.
"""

import h5py
import numpy as np
import json
from pathlib import Path

def extract_sample_sequences(dataset_path, num_samples=3):
    """Extract sample sequences in readable format."""
    
    samples = []
    
    with h5py.File(dataset_path, 'r') as f:
        # Get metadata
        metadata = dict(f['metadata'].attrs)
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in metadata.items():
            if hasattr(value, 'item'):
                metadata[key] = value.item()
        
        print(f"Dataset Metadata:")
        print(f"  Total sequences: {metadata.get('num_sequences', 'unknown')}")
        print(f"  Exercise type: {metadata.get('exercise_type', 'unknown')}")
        print(f"  Generation method: {metadata.get('generation_method', 'unknown')}")
        print(f"  Generation time: {metadata.get('generation_time', 'unknown'):.2f} seconds")
        print()
        
        # Extract sample sequences
        for i in range(min(num_samples, metadata.get('num_sequences', 0))):
            print(f"=== SEQUENCE {i} ===")
            
            # Load pose and IMU data
            poses = f['poses'][f'sequence_{i}'][:]
            imu_data = f['imu_data'][f'sequence_{i}'][:]
            exercise_label = f['exercise_labels'][i]
            form_label = f['form_labels'][i]
            
            print(f"Sequence length: {len(poses)} timesteps")
            print(f"Pose dimensions: {poses.shape}")
            print(f"IMU dimensions: {imu_data.shape}")
            print(f"Exercise label: {exercise_label} (0=squat)")
            print(f"Form label: {form_label} (1=good form)")
            print()
            
            # Show first few timesteps
            print("First 5 timesteps:")
            print("Timestep | Pose Vector (first 10 dims) | IMU Data [accel_xyz, gyro_xyz]")
            print("-" * 80)
            
            for t in range(min(5, len(poses))):
                pose_sample = poses[t][:10]  # First 10 dimensions
                imu_sample = imu_data[t]
                
                pose_str = "[" + ", ".join([f"{x:6.3f}" for x in pose_sample]) + ", ...]"
                imu_str = "[" + ", ".join([f"{x:6.3f}" for x in imu_sample]) + "]"
                
                print(f"{t:8d} | {pose_str:35s} | {imu_str}")
            
            print()
            
            # Show middle timesteps
            mid_point = len(poses) // 2
            print(f"Middle 3 timesteps (around t={mid_point}):")
            print("Timestep | Pose Vector (first 10 dims) | IMU Data [accel_xyz, gyro_xyz]")
            print("-" * 80)
            
            for t in range(max(0, mid_point-1), min(len(poses), mid_point+2)):
                pose_sample = poses[t][:10]
                imu_sample = imu_data[t]
                
                pose_str = "[" + ", ".join([f"{x:6.3f}" for x in pose_sample]) + ", ...]"
                imu_str = "[" + ", ".join([f"{x:6.3f}" for x in imu_sample]) + "]"
                
                print(f"{t:8d} | {pose_str:35s} | {imu_str}")
            
            print()
            
            # Statistical summary
            print("Statistical Summary:")
            print(f"  Pose vector range: [{poses.min():.3f}, {poses.max():.3f}]")
            print(f"  Pose vector mean: {poses.mean():.3f}, std: {poses.std():.3f}")
            print(f"  IMU accel range: [{imu_data[:,:3].min():.3f}, {imu_data[:,:3].max():.3f}]")
            print(f"  IMU gyro range: [{imu_data[:,3:].min():.3f}, {imu_data[:,3:].max():.3f}]")
            print()
            
            # Create sample for JSON export
            sample_data = {
                "sequence_id": i,
                "length": len(poses),
                "exercise_label": int(exercise_label),
                "form_label": int(form_label),
                "pose_shape": list(poses.shape),
                "imu_shape": list(imu_data.shape),
                "first_5_timesteps": {
                    "poses": poses[:5].tolist(),
                    "imu_data": imu_data[:5].tolist()
                },
                "statistics": {
                    "pose_min": float(poses.min()),
                    "pose_max": float(poses.max()),
                    "pose_mean": float(poses.mean()),
                    "pose_std": float(poses.std()),
                    "imu_accel_range": [float(imu_data[:,:3].min()), float(imu_data[:,:3].max())],
                    "imu_gyro_range": [float(imu_data[:,3:].min()), float(imu_data[:,3:].max())]
                }
            }
            
            samples.append(sample_data)
            
            print("=" * 80)
            print()
    
    return samples, metadata

if __name__ == "__main__":
    dataset_path = "data/production_squats_10k.h5"
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        exit(1)
    
    print("Extracting sample synthetic squat data for auditing...")
    print("=" * 80)
    print()
    
    # Extract samples
    samples, metadata = extract_sample_sequences(dataset_path, num_samples=3)
    
    # Save to JSON for easy inspection
    output_data = {
        "dataset_metadata": metadata,
        "sample_sequences": samples,
        "data_format_description": {
            "pose_vector": "51-dimensional vector containing joint positions, velocities, and body positions from MuJoCo simulation",
            "imu_data": "6-dimensional vector [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] simulated from torso motion",
            "exercise_label": "0 = squat exercise",
            "form_label": "1 = good form quality",
            "physics_engine": "MuJoCo with simple_humanoid.xml model",
            "generation_rate": "~118 sequences per second"
        }
    }
    
    # Save readable sample
    with open("sample_synthetic_data.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Sample data saved to: sample_synthetic_data.json")
    print("This file can be easily audited by another LLM or human reviewer.")

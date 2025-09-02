"""
Generate large-scale synthetic dataset using MuJoCo physics engine.
Creates 10,000+ squat sequences for production training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.physics_engine import PhysicsDataEngine
import numpy as np
import h5py
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_production_dataset():
    """Generate large-scale production dataset."""
    
    # Initialize physics engine
    logger.info("Initializing MuJoCo physics engine...")
    engine = PhysicsDataEngine()
    
    # Test engine first
    if not engine.test_simulation(steps=50):
        logger.error("Physics engine test failed!")
        return False
    
    logger.info("Physics engine validated successfully!")
    
    # Dataset parameters
    num_sequences = 10000  # Large-scale dataset
    batch_size = 100       # Process in batches to manage memory
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Output file
    output_file = data_dir / "production_squats_10k.h5"
    
    logger.info(f"Generating {num_sequences} squat sequences...")
    logger.info(f"Output file: {output_file}")
    
    start_time = time.time()
    
    # Generate dataset in batches
    with h5py.File(output_file, 'w') as f:
        # Create datasets for storing sequences
        poses_group = f.create_group('poses')
        imu_group = f.create_group('imu_data')
        
        # Create arrays for labels
        exercise_labels = []
        form_labels = []
        
        for batch_idx in range(0, num_sequences, batch_size):
            batch_end = min(batch_idx + batch_size, num_sequences)
            current_batch_size = batch_end - batch_idx
            
            logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(num_sequences-1)//batch_size + 1} "
                       f"(sequences {batch_idx}-{batch_end-1})")
            
            batch_poses = []
            batch_imu = []
            batch_exercise_labels = []
            batch_form_labels = []
            
            for seq_idx in range(current_batch_size):
                global_idx = batch_idx + seq_idx
                
                # Vary squat parameters for diversity
                duration = np.random.uniform(2.0, 5.0)  # 2-5 second squats
                
                # Generate sequence
                try:
                    sequence = engine.generate_squat_sequence(duration=duration)
                    
                    batch_poses.append(sequence['poses'])
                    batch_imu.append(sequence['imu_data'])
                    batch_exercise_labels.append(sequence['exercise_label'])
                    batch_form_labels.append(sequence['form_label'])
                    
                except Exception as e:
                    logger.warning(f"Failed to generate sequence {global_idx}: {e}")
                    continue
                
                # Progress update
                if global_idx % 500 == 0 and global_idx > 0:
                    elapsed = time.time() - start_time
                    rate = global_idx / elapsed
                    eta = (num_sequences - global_idx) / rate
                    logger.info(f"Progress: {global_idx}/{num_sequences} ({global_idx/num_sequences*100:.1f}%) "
                               f"Rate: {rate:.1f} seq/s, ETA: {eta/60:.1f} min")
            
            # Save batch to HDF5
            for i, (poses, imu) in enumerate(zip(batch_poses, batch_imu)):
                seq_id = batch_idx + i
                poses_group.create_dataset(f'sequence_{seq_id}', data=poses, compression='gzip')
                imu_group.create_dataset(f'sequence_{seq_id}', data=imu, compression='gzip')
            
            # Accumulate labels
            exercise_labels.extend(batch_exercise_labels)
            form_labels.extend(batch_form_labels)
        
        # Save labels
        f.create_dataset('exercise_labels', data=np.array(exercise_labels))
        f.create_dataset('form_labels', data=np.array(form_labels))
        
        # Save metadata
        metadata_group = f.create_group('metadata')
        metadata_group.attrs['num_sequences'] = len(exercise_labels)
        metadata_group.attrs['exercise_type'] = 'squat'
        metadata_group.attrs['generation_method'] = 'mujoco_physics'
        metadata_group.attrs['model_path'] = engine.model_path
        metadata_group.attrs['generation_time'] = time.time() - start_time
    
    total_time = time.time() - start_time
    logger.info(f"Dataset generation complete!")
    logger.info(f"Generated {len(exercise_labels)} sequences in {total_time/60:.1f} minutes")
    logger.info(f"Average rate: {len(exercise_labels)/total_time:.1f} sequences/second")
    logger.info(f"Dataset saved to: {output_file}")
    
    return True

def validate_dataset():
    """Validate the generated dataset."""
    data_file = Path("data/production_squats_10k.h5")
    
    if not data_file.exists():
        logger.error(f"Dataset file not found: {data_file}")
        return False
    
    logger.info("Validating generated dataset...")
    
    with h5py.File(data_file, 'r') as f:
        # Check structure
        assert 'poses' in f, "Missing poses group"
        assert 'imu_data' in f, "Missing imu_data group"
        assert 'exercise_labels' in f, "Missing exercise_labels"
        assert 'form_labels' in f, "Missing form_labels"
        assert 'metadata' in f, "Missing metadata"
        
        # Check data
        num_sequences = len(f['exercise_labels'])
        logger.info(f"Dataset contains {num_sequences} sequences")
        
        # Sample a few sequences to validate shapes
        for i in range(min(5, num_sequences)):
            poses = f['poses'][f'sequence_{i}'][:]
            imu = f['imu_data'][f'sequence_{i}'][:]
            
            logger.info(f"Sequence {i}: poses {poses.shape}, imu {imu.shape}")
            
            # Validate shapes
            assert poses.shape[1] == 51, f"Expected pose dim 51, got {poses.shape[1]}"
            assert imu.shape[1] == 6, f"Expected IMU dim 6, got {imu.shape[1]}"
        
        # Check metadata
        metadata = f['metadata']
        logger.info(f"Metadata: {dict(metadata.attrs)}")
    
    logger.info("✅ Dataset validation successful!")
    return True

if __name__ == "__main__":
    logger.info("Starting large-scale synthetic dataset generation...")
    
    try:
        # Generate dataset
        success = generate_production_dataset()
        
        if success:
            # Validate dataset
            validate_dataset()
            logger.info("🎉 Production dataset generation and validation complete!")
        else:
            logger.error("❌ Dataset generation failed!")
            
    except Exception as e:
        logger.error(f"❌ Error during dataset generation: {e}")
        raise

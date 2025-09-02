"""
Debug HDF5 structure to understand the data organization.
"""

import h5py
import numpy as np

def debug_hdf5():
    with h5py.File("data/production_squats_10k.h5", 'r') as f:
        print("Full HDF5 structure:")
        
        def print_structure(name, obj):
            print(f"{name}: {type(obj)}")
            if hasattr(obj, 'shape'):
                print(f"  Shape: {obj.shape}")
            if hasattr(obj, 'attrs'):
                for attr in obj.attrs:
                    print(f"  Attr {attr}: {obj.attrs[attr]}")
        
        f.visititems(print_structure)
        
        # Check specific sequences
        print("\nChecking first few sequences:")
        for i in range(3):
            if f'poses/{i}' in f:
                poses = f[f'poses/{i}'][:]
                imu = f[f'imu_data/{i}'][:]
                print(f"Sequence {i}: poses {poses.shape}, imu {imu.shape}")

if __name__ == "__main__":
    debug_hdf5()

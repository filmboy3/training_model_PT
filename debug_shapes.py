"""
Debug script to check actual tensor shapes in the dataset.
"""

import h5py
import torch
from simple_dataset_loader import ProductionDataset

def debug_shapes():
    print("🔍 Debugging dataset shapes...")
    
    # Check HDF5 structure directly
    with h5py.File("data/production_squats_10k.h5", 'r') as f:
        print("\nHDF5 file structure:")
        for key in f.keys():
            if hasattr(f[key], 'shape'):
                print(f"  {key}: {f[key].shape}")
            else:
                print(f"  {key}: {type(f[key])}")
                if hasattr(f[key], 'attrs'):
                    for attr_key in f[key].attrs.keys():
                        print(f"    {attr_key}: {f[key].attrs[attr_key]}")
    
    # Check dataset loader
    dataset = ProductionDataset("data/production_squats_10k.h5", max_seq_length=200)
    print(f"\nDataset length: {len(dataset)}")
    
    # Check first sample
    sample = dataset[0]
    print("\nSample shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value} (type: {type(value)})")
    
    # Check a few more samples
    for i in [1, 2, 3]:
        sample = dataset[i]
        print(f"\nSample {i} shapes:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")

if __name__ == "__main__":
    debug_shapes()

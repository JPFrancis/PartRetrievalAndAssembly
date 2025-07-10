#!/usr/bin/env python3
"""
Test script to verify Kaedim pickle file loading
"""

import sys
import os
sys.path.append('src')

from config import global_args
from data_manager import load_kaedim_dataset, get_parts, get_shapes

def test_pickle_loading():
    """Test loading the Kaedim pickle file"""
    
    # Set the pickle file path
    pickle_file = 'data/kaedim-chair-eval-dataset.pickle'
    
    print("Testing pickle file loading...")
    print(f"Pickle file: {pickle_file}")
    
    # Test direct loading
    try:
        part_meshes, part_vol_pcs, part_sur_pcs, shape_meshes, shape_vol_pcs, shape_sur_pcs = load_kaedim_dataset(pickle_file)
        print(f"✓ Successfully loaded {len(part_vol_pcs)} parts and {len(shape_vol_pcs)} shapes")
        
        if len(part_vol_pcs) > 0:
            print(f"  First part shape: {part_vol_pcs[0].shape}")
        if len(shape_vol_pcs) > 0:
            print(f"  First shape shape: {shape_vol_pcs[0].shape}")
            
    except Exception as e:
        print(f"✗ Error loading pickle file: {e}")
        return False
    
    # Test through the get_parts function
    try:
        # Temporarily set the pickle file in global_args
        global_args.pickle_file = pickle_file
        
        part_meshes, part_vol_pcs, part_sur_pcs = get_parts('', 'partnet', 'chair', 5, [], False)
        print(f"✓ Successfully loaded {len(part_vol_pcs)} parts through get_parts")
        
    except Exception as e:
        print(f"✗ Error in get_parts: {e}")
        return False
    
    # Test through the get_shapes function
    try:
        shape_meshes, shape_vol_pcs, shape_sur_pcs = get_shapes('', 'partnet', 'chair', [], 5, False)
        print(f"✓ Successfully loaded {len(shape_vol_pcs)} shapes through get_shapes")
        
    except Exception as e:
        print(f"✗ Error in get_shapes: {e}")
        return False
    
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_pickle_loading()
    sys.exit(0 if success else 1) 
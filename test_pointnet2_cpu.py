#!/usr/bin/env python3
"""
Test script for CPU-only PointNet2 feature extraction
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import sys

from pointnet2_feature_extractor_cpu import PointNet2Backbone, preprocess_point_cloud

def test_model():
    """Test the PointNet2 model with random data"""
    print("Testing PointNet2 model...")
    
    # Create random point cloud data
    batch_size = 2
    num_points = 1000
    spatial_dims = 3
    feature_dims = 3
    
    # Random coordinates and features
    xyz = torch.randn(batch_size, spatial_dims, num_points)
    features = torch.randn(batch_size, feature_dims, num_points)
    
    # Initialize model
    model = PointNet2Backbone(
        spatial_dims=spatial_dims, 
        feature_dims=feature_dims, 
        feature_dim=128
    )
    model.eval()
    
    # Test forward pass
    with torch.no_grad():
        output = model(xyz, features)
    
    print(f"Input shape: xyz={xyz.shape}, features={features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [{batch_size}, 128, {num_points}]")
    
    if output.shape == (batch_size, 128, 512):  # SA layers downsample to 512 points
        print("✓ Model test passed!")
        return True
    else:
        print("✗ Model test failed!")
        return False

def test_preprocessing():
    """Test point cloud preprocessing"""
    print("\nTesting point cloud preprocessing...")
    
    # Create random point cloud
    num_points = 500
    points = np.random.randn(num_points, 6)  # 3 spatial + 3 feature dimensions
    points[:, :3] *= 10  # Scale spatial coordinates
    
    # Test preprocessing
    processed_points, center_coords = preprocess_point_cloud(points, normalize=True, center=True)
    
    print(f"Original points shape: {points.shape}")
    print(f"Processed points shape: {processed_points.shape}")
    print(f"Center coordinates: {center_coords}")
    
    # Check that centering worked
    centered_coords = processed_points[:, :3]
    mean_coords = np.mean(centered_coords, axis=0)
    print(f"Mean of centered coordinates: {mean_coords}")
    
    if np.allclose(mean_coords, np.zeros(3), atol=1e-6):
        print("✓ Preprocessing test passed!")
        return True
    else:
        print("✗ Preprocessing test failed!")
        return False

def test_feature_extraction():
    """Test feature extraction from file"""
    print("\nTesting feature extraction from file...")
    
    # Create sample data file
    sample_file = Path("test_data.h5")
    num_points = 300
    points = np.random.randn(num_points, 9)  # 9 features
    points[:, :3] *= 5  # Scale spatial coordinates
    
    with h5py.File(sample_file, 'w') as f:
        f.create_dataset('fused_detections', data=points)
    
    # Test feature extraction
    model = PointNet2Backbone(spatial_dims=3, feature_dims=3, feature_dim=64)
    model.eval()
    
    from pointnet2_feature_extractor_cpu import extract_features_from_file
    
    try:
        output_path = extract_features_from_file(
            model, str(sample_file), "test_output", device='cpu'
        )
        
        # Check output file
        with h5py.File(output_path, 'r') as f:
            features = f['features'][:]
            original_points = f['original_points'][:]
            center = f['center'][:]
        
        print(f"Extracted features shape: {features.shape}")
        print(f"Original points shape: {original_points.shape}")
        print(f"Center coordinates: {center}")
        
        if features.shape[1] == 128:  # feature_dim
            print("✓ Feature extraction test passed!")
            return True
        else:
            print("✗ Feature extraction test failed!")
            return False
            
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        return False
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()
        test_output_dir = Path("test_output")
        if test_output_dir.exists():
            import shutil
            shutil.rmtree(test_output_dir)

def main():
    """Run all tests"""
    print("Running CPU-only PointNet2 feature extraction tests...\n")
    
    tests = [
        test_model,
        test_preprocessing,
        test_feature_extraction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! CPU-only PointNet2 feature extraction is ready to use.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    main()
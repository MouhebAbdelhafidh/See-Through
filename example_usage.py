#!/usr/bin/env python3
"""
Example usage of PointNet2 feature extraction
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from pointnet2_feature_extractor_cpu import PointNet2Backbone, extract_features_from_file

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic PointNet2 Feature Extraction Example ===\n")
    
    # Initialize model
    model = PointNet2Backbone(spatial_dims=3, feature_dims=3, feature_dim=128)
    model.eval()
    
    # Create sample point cloud data
    num_points = 1000
    xyz = torch.randn(1, 3, num_points)  # [batch, 3, num_points]
    features = torch.randn(1, 3, num_points)  # [batch, 3, num_points]
    
    print(f"Input point cloud: {xyz.shape}")
    print(f"Input features: {features.shape}")
    
    # Extract features
    with torch.no_grad():
        extracted_features = model(xyz, features)
    
    print(f"Extracted features: {extracted_features.shape}")
    print("✓ Basic usage example completed!\n")

def example_file_processing():
    """Example of processing HDF5 files"""
    print("=== File Processing Example ===\n")
    
    # Create sample data directory
    sample_dir = Path("SampleData")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample HDF5 file
    sample_file = sample_dir / "sample_data.h5"
    num_points = 500
    points = np.random.randn(num_points, 9)  # 9 features: x, y, z, vr, rcs, v, r, num_merged, etc.
    points[:, :3] *= 10  # Scale spatial coordinates
    
    with h5py.File(sample_file, 'w') as f:
        f.create_dataset('fused_detections', data=points)
    
    print(f"Created sample data: {sample_file}")
    print(f"Point cloud shape: {points.shape}")
    
    # Initialize model
    model = PointNet2Backbone(spatial_dims=3, feature_dims=3, feature_dim=64)
    model.eval()
    
    # Process file
    output_dir = Path("ExampleOutput")
    output_path = extract_features_from_file(
        model, str(sample_file), str(output_dir), device='cpu'
    )
    
    # Load and display results
    with h5py.File(output_path, 'r') as f:
        features = f['features'][:]
        original_points = f['original_points'][:]
        center = f['center'][:]
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Original points shape: {original_points.shape}")
    print(f"Center coordinates: {center}")
    print("✓ File processing example completed!\n")

def example_batch_processing():
    """Example of processing multiple files"""
    print("=== Batch Processing Example ===\n")
    
    # Create sample data directory
    sample_dir = Path("BatchSampleData")
    sample_dir.mkdir(exist_ok=True)
    
    # Create multiple sample files
    num_files = 3
    for i in range(num_files):
        sample_file = sample_dir / f"data_{i:03d}.h5"
        num_points = np.random.randint(300, 800)
        points = np.random.randn(num_points, 9)
        points[:, :3] *= np.random.uniform(5, 15)
        
        with h5py.File(sample_file, 'w') as f:
            f.create_dataset('fused_detections', data=points)
        
        print(f"Created {sample_file} with {num_points} points")
    
    # Process all files
    from pointnet2_feature_extractor_cpu import main
    import sys
    
    # Set up command line arguments
    sys.argv = [
        'pointnet2_feature_extractor_cpu.py',
        '--input_dir', str(sample_dir),
        '--output_dir', 'BatchOutput',
        '--feature_dim', '64',
        '--device', 'cpu'
    ]
    
    print("Processing files...")
    main()
    print("✓ Batch processing example completed!\n")

def example_custom_model():
    """Example of using a custom model configuration"""
    print("=== Custom Model Example ===\n")
    
    # Create custom model with different parameters
    model = PointNet2Backbone(
        spatial_dims=3,
        feature_dims=3,
        feature_dim=256  # Larger feature dimension
    )
    model.eval()
    
    # Create sample data
    num_points = 800
    xyz = torch.randn(1, 3, num_points)
    features = torch.randn(1, 3, num_points)
    
    # Extract features
    with torch.no_grad():
        extracted_features = model(xyz, features)
    
    print(f"Custom model output shape: {extracted_features.shape}")
    print(f"Feature dimension: {extracted_features.shape[1]}")
    print("✓ Custom model example completed!\n")

def main():
    """Run all examples"""
    print("PointNet2 Feature Extraction Examples\n")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_file_processing()
        example_custom_model()
        
        print("All examples completed successfully!")
        print("\nTo run feature extraction on your data:")
        print("python pointnet2_feature_extractor_cpu.py --input_dir YourDataDir --output_dir ExtractedFeatures")
        
    except Exception as e:
        print(f"Error running examples: {e}")

if __name__ == "__main__":
    main()
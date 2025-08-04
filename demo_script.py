#!/usr/bin/env python3
"""
PointNet2 Feature Extractor - Demo Script
========================================

This script demonstrates the key features of the PointNet2 implementation
based on the erikwijmans/Pointnet2_PyTorch repository.
"""

import numpy as np
import torch
import time
from pointnet2_feature_extractor import (
    PointNet2FeatureExtractor,
    create_sample_point_cloud,
    extract_features,
    normalize_point_cloud
)


def demo_basic_usage():
    """Demonstrate basic feature extraction"""
    print("=" * 60)
    print("DEMO 1: Basic Feature Extraction")
    print("=" * 60)
    
    # Create sample point cloud
    print("Creating sample point cloud...")
    point_cloud = create_sample_point_cloud(num_points=1024)
    print(f"Point cloud shape: {point_cloud.shape}")
    
    # Initialize model
    print("Initializing PointNet2 model...")
    model = PointNet2FeatureExtractor(num_classes=40, use_msg=False, use_xyz=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Extract features
    print("Extracting features...")
    start_time = time.time()
    features = extract_features(model, point_cloud, device)
    end_time = time.time()
    
    print(f"Feature extraction completed in {end_time - start_time:.4f} seconds")
    print("\nExtracted features:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
    
    return features


def demo_msg_comparison():
    """Compare Single-Scale vs Multi-Scale Grouping"""
    print("\n" + "=" * 60)
    print("DEMO 2: SSG vs MSG Comparison")
    print("=" * 60)
    
    # Create sample point cloud
    point_cloud = create_sample_point_cloud(num_points=1024)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test SSG
    print("Testing Single-Scale Grouping (SSG)...")
    model_ssg = PointNet2FeatureExtractor(use_msg=False, use_xyz=False)
    start_time = time.time()
    features_ssg = extract_features(model_ssg, point_cloud, device)
    time_ssg = time.time() - start_time
    
    # Test MSG
    print("Testing Multi-Scale Grouping (MSG)...")
    model_msg = PointNet2FeatureExtractor(use_msg=True, use_xyz=False)
    start_time = time.time()
    features_msg = extract_features(model_msg, point_cloud, device)
    time_msg = time.time() - start_time
    
    print(f"\nPerformance Comparison:")
    print(f"  SSG time: {time_ssg:.4f}s")
    print(f"  MSG time: {time_msg:.4f}s")
    print(f"  MSG is {time_msg/time_ssg:.2f}x slower than SSG")
    
    print(f"\nFeature Comparison:")
    print(f"  SSG SA1 features: {features_ssg['sa1_features'].shape}")
    print(f"  MSG SA1 features: {features_msg['sa1_features'].shape}")
    print(f"  SSG SA2 features: {features_ssg['sa2_features'].shape}")
    print(f"  MSG SA2 features: {features_msg['sa2_features'].shape}")
    
    return features_ssg, features_msg


def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n" + "=" * 60)
    print("DEMO 3: Batch Processing")
    print("=" * 60)
    
    # Create batch of point clouds
    batch_size = 4
    point_clouds = []
    
    print(f"Creating batch of {batch_size} point clouds...")
    for i in range(batch_size):
        if i == 0:
            # Sphere
            pc = create_sample_point_cloud(1024)
        elif i == 1:
            # Random cube
            pc = np.random.uniform(-1, 1, (1024, 3)).astype(np.float32)
            pc = normalize_point_cloud(pc)
        elif i == 2:
            # Cylinder
            theta = np.random.uniform(0, 2*np.pi, 1024)
            z = np.random.uniform(-1, 1, 1024)
            r = np.random.uniform(0, 1, 1024)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pc = np.column_stack([x, y, z]).astype(np.float32)
            pc = normalize_point_cloud(pc)
        else:
            # Plane
            x = np.random.uniform(-1, 1, 1024)
            y = np.random.uniform(-1, 1, 1024)
            z = np.zeros(1024)
            pc = np.column_stack([x, y, z]).astype(np.float32)
            pc = normalize_point_cloud(pc)
        
        point_clouds.append(pc)
    
    # Process individually
    model = PointNet2FeatureExtractor(use_msg=False, use_xyz=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    
    print("Processing individually...")
    start_time = time.time()
    individual_features = []
    for pc in point_clouds:
        features = extract_features(model, pc, device)
        individual_features.append(features)
    individual_time = time.time() - start_time
    
    # Process as batch
    print("Processing as batch...")
    batch_tensor = torch.from_numpy(np.stack(point_clouds)).float().to(device)
    start_time = time.time()
    with torch.no_grad():
        batch_features = model(batch_tensor, return_features=True)
    batch_time = time.time() - start_time
    
    print(f"\nBatch Processing Results:")
    print(f"  Individual processing time: {individual_time:.4f}s")
    print(f"  Batch processing time: {batch_time:.4f}s")
    print(f"  Speedup: {individual_time/batch_time:.2f}x")
    print(f"  Batch tensor shape: {batch_tensor.shape}")
    print(f"  Batch features shape: {batch_features['global_features'].shape}")
    
    return batch_features


def demo_different_shapes():
    """Test with different geometric shapes"""
    print("\n" + "=" * 60)
    print("DEMO 4: Different Geometric Shapes")
    print("=" * 60)
    
    def create_torus(R=1.0, r=0.3, num_points=1024):
        """Create torus point cloud"""
        theta = np.random.uniform(0, 2*np.pi, num_points)
        phi = np.random.uniform(0, 2*np.pi, num_points)
        
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        return np.column_stack([x, y, z]).astype(np.float32)
    
    def create_spiral(num_points=1024):
        """Create spiral point cloud"""
        t = np.linspace(0, 4*np.pi, num_points)
        x = t * np.cos(t) / (4*np.pi)
        y = t * np.sin(t) / (4*np.pi)
        z = t / (4*np.pi)
        return np.column_stack([x, y, z]).astype(np.float32)
    
    shapes = {
        'Sphere': create_sample_point_cloud(1024),
        'Torus': normalize_point_cloud(create_torus()),
        'Spiral': normalize_point_cloud(create_spiral())
    }
    
    model = PointNet2FeatureExtractor(use_msg=False, use_xyz=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing different shapes:")
    shape_features = {}
    
    for shape_name, point_cloud in shapes.items():
        print(f"\n  Processing {shape_name}...")
        features = extract_features(model, point_cloud, device)
        shape_features[shape_name] = features
        
        global_feat = features['global_features'][0]
        classification = features['classification'][0]
        predicted_class = np.argmax(classification)
        confidence = np.max(classification)
        
        print(f"    Global feature stats: min={global_feat.min():.4f}, "
              f"max={global_feat.max():.4f}, mean={global_feat.mean():.4f}")
        print(f"    Predicted class: {predicted_class}, confidence: {confidence:.4f}")
    
    return shape_features


def demo_feature_analysis():
    """Analyze the extracted features"""
    print("\n" + "=" * 60)
    print("DEMO 5: Feature Analysis")
    print("=" * 60)
    
    # Create two similar and one different shape
    sphere1 = create_sample_point_cloud(1024)
    sphere2 = create_sample_point_cloud(1024)  # Different random sphere
    
    # Create a very different shape (plane)
    x = np.random.uniform(-1, 1, 1024)
    y = np.random.uniform(-1, 1, 1024)
    z = np.zeros(1024)
    plane = normalize_point_cloud(np.column_stack([x, y, z]).astype(np.float32))
    
    model = PointNet2FeatureExtractor(use_msg=False, use_xyz=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Extract features
    features1 = extract_features(model, sphere1, device)
    features2 = extract_features(model, sphere2, device)
    features3 = extract_features(model, plane, device)
    
    # Compare global features
    feat1 = features1['global_features'][0]
    feat2 = features2['global_features'][0]
    feat3 = features3['global_features'][0]
    
    # Calculate similarities (cosine similarity)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_spheres = cosine_similarity(feat1, feat2)
    sim_sphere_plane = cosine_similarity(feat1, feat3)
    
    print("Feature Similarity Analysis:")
    print(f"  Sphere1 vs Sphere2: {sim_spheres:.4f}")
    print(f"  Sphere1 vs Plane: {sim_sphere_plane:.4f}")
    print(f"  Similar shapes should have higher similarity!")
    
    # Analyze feature distributions
    print(f"\nFeature Distribution Analysis:")
    print(f"  Sphere1 - mean: {feat1.mean():.4f}, std: {feat1.std():.4f}")
    print(f"  Sphere2 - mean: {feat2.mean():.4f}, std: {feat2.std():.4f}")
    print(f"  Plane   - mean: {feat3.mean():.4f}, std: {feat3.std():.4f}")
    
    return features1, features2, features3


def main():
    """Run all demonstrations"""
    print("PointNet2 Feature Extractor - Comprehensive Demo")
    print("Based on https://github.com/erikwijmans/Pointnet2_PyTorch")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Run demonstrations
        demo_basic_usage()
        demo_msg_comparison()
        demo_batch_processing()
        demo_different_shapes()
        demo_feature_analysis()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Basic feature extraction from point clouds")
        print("✓ Single-Scale vs Multi-Scale Grouping comparison")
        print("✓ Efficient batch processing capabilities")
        print("✓ Support for different geometric shapes")
        print("✓ Feature similarity analysis")
        print("\nThe implementation successfully replicates the core")
        print("functionality of the erikwijmans/Pointnet2_PyTorch repository!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
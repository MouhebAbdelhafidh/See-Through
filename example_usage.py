#!/usr/bin/env python3
"""
Example Usage of PointNet2 Feature Extractor
===========================================

This script demonstrates different ways to use the PointNet2 feature extractor
for various point cloud processing tasks.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pointnet2_feature_extractor import (
    PointNet2FeatureExtractor, 
    extract_features, 
    create_sample_point_cloud,
    normalize_point_cloud,
    sample_point_cloud
)


def example_1_basic_feature_extraction():
    """Example 1: Basic feature extraction from a sample point cloud"""
    print("=" * 60)
    print("Example 1: Basic Feature Extraction")
    print("=" * 60)
    
    # Create a sample point cloud (sphere)
    point_cloud = create_sample_point_cloud(num_points=1024, add_normals=False)
    print(f"Created sample point cloud: {point_cloud.shape}")
    
    # Initialize the model
    model = PointNet2FeatureExtractor(num_classes=40, use_msg=False, use_xyz=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Extract features
    features = extract_features(model, point_cloud, device)
    
    # Print results
    print("\nExtracted Features:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
    
    global_features = features['global_features'][0]
    print(f"\nGlobal Feature Statistics:")
    print(f"  Min: {global_features.min():.4f}")
    print(f"  Max: {global_features.max():.4f}")
    print(f"  Mean: {global_features.mean():.4f}")
    print(f"  Std: {global_features.std():.4f}")
    
    return features


def example_2_multi_scale_grouping():
    """Example 2: Using Multi-Scale Grouping (MSG) version"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Scale Grouping (MSG)")
    print("=" * 60)
    
    # Create a more complex point cloud (torus shape)
    def create_torus(R=1.0, r=0.3, num_points=1024):
        theta = np.random.uniform(0, 2*np.pi, num_points)
        phi = np.random.uniform(0, 2*np.pi, num_points)
        
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        return np.column_stack([x, y, z]).astype(np.float32)
    
    point_cloud = create_torus(num_points=1024)
    point_cloud = normalize_point_cloud(point_cloud)
    print(f"Created torus point cloud: {point_cloud.shape}")
    
    # Initialize MSG model
    model = PointNet2FeatureExtractor(num_classes=40, use_msg=True, use_xyz=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Extract features
    features = extract_features(model, point_cloud, device)
    
    print("\nMSG Features:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
    
    return features


def example_3_with_normals():
    """Example 3: Feature extraction with normal vectors"""
    print("\n" + "=" * 60)
    print("Example 3: Feature Extraction with Normals")
    print("=" * 60)
    
    # Create point cloud with normals
    point_cloud = create_sample_point_cloud(num_points=1024, add_normals=True)
    print(f"Created point cloud with normals: {point_cloud.shape}")
    
    # Initialize model with normal support
    model = PointNet2FeatureExtractor(num_classes=40, use_msg=False, use_xyz=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Extract features
    features = extract_features(model, point_cloud, device)
    
    print("\nFeatures with Normals:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
    
    return features


def example_4_batch_processing():
    """Example 4: Batch processing multiple point clouds"""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)
    
    # Create multiple point clouds
    batch_size = 4
    point_clouds = []
    
    for i in range(batch_size):
        if i == 0:
            pc = create_sample_point_cloud(1024)  # Sphere
        elif i == 1:
            # Cube
            pc = np.random.uniform(-1, 1, (1024, 3)).astype(np.float32)
        elif i == 2:
            # Cylinder
            theta = np.random.uniform(0, 2*np.pi, 1024)
            z = np.random.uniform(-1, 1, 1024)
            r = np.random.uniform(0, 1, 1024)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pc = np.column_stack([x, y, z]).astype(np.float32)
        else:
            # Plane
            x = np.random.uniform(-1, 1, 1024)
            y = np.random.uniform(-1, 1, 1024)
            z = np.zeros(1024)
            pc = np.column_stack([x, y, z]).astype(np.float32)
        
        pc = normalize_point_cloud(pc)
        point_clouds.append(pc)
    
    # Initialize model
    model = PointNet2FeatureExtractor(num_classes=40, use_msg=False, use_xyz=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    
    # Batch processing
    batch_tensor = torch.from_numpy(np.stack(point_clouds)).float().to(device)
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    with torch.no_grad():
        batch_features = model(batch_tensor, return_features=True)
    
    print("\nBatch Features:")
    for key, value in batch_features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Convert to numpy
    batch_features_np = {}
    for key, value in batch_features.items():
        if isinstance(value, torch.Tensor):
            batch_features_np[key] = value.cpu().numpy()
    
    return batch_features_np


def example_5_feature_visualization():
    """Example 5: Visualizing extracted features"""
    print("\n" + "=" * 60)
    print("Example 5: Feature Visualization")
    print("=" * 60)
    
    # Create point cloud
    point_cloud = create_sample_point_cloud(num_points=1024, add_normals=False)
    
    # Extract features
    model = PointNet2FeatureExtractor(num_classes=40, use_msg=False, use_xyz=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = extract_features(model, point_cloud, device)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Original point cloud
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                c=point_cloud[:, 2], cmap='viridis', s=1)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: SA1 points
    sa1_xyz = features['sa1_xyz'][0].T  # [3, N] -> [N, 3]
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(sa1_xyz[:, 0], sa1_xyz[:, 1], sa1_xyz[:, 2], 
                c=sa1_xyz[:, 2], cmap='plasma', s=5)
    ax2.set_title('SA1 Sampled Points (512)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Plot 3: SA2 points
    sa2_xyz = features['sa2_xyz'][0].T  # [3, N] -> [N, 3]
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.scatter(sa2_xyz[:, 0], sa2_xyz[:, 1], sa2_xyz[:, 2], 
                c=sa2_xyz[:, 2], cmap='coolwarm', s=10)
    ax3.set_title('SA2 Sampled Points (128)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # Plot 4: Global feature histogram
    ax4 = fig.add_subplot(2, 3, 4)
    global_features = features['global_features'][0]
    ax4.hist(global_features, bins=50, alpha=0.7, color='blue')
    ax4.set_title('Global Features Distribution')
    ax4.set_xlabel('Feature Value')
    ax4.set_ylabel('Frequency')
    
    # Plot 5: Feature heatmap (first 100 features)
    ax5 = fig.add_subplot(2, 3, 5)
    feature_matrix = global_features[:100].reshape(10, 10)
    im = ax5.imshow(feature_matrix, cmap='hot', interpolation='nearest')
    ax5.set_title('Global Features Heatmap (First 100)')
    plt.colorbar(im, ax=ax5)
    
    # Plot 6: Classification scores
    ax6 = fig.add_subplot(2, 3, 6)
    classification = features['classification'][0]
    top_classes = np.argsort(classification)[-10:][::-1]
    ax6.bar(range(10), classification[top_classes])
    ax6.set_title('Top 10 Classification Scores')
    ax6.set_xlabel('Class Index')
    ax6.set_ylabel('Score')
    ax6.set_xticks(range(10))
    ax6.set_xticklabels([str(i) for i in top_classes])
    
    plt.tight_layout()
    plt.savefig('pointnet2_features_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'pointnet2_features_visualization.png'")
    
    return features


def example_6_custom_point_cloud():
    """Example 6: Loading and processing custom point cloud data"""
    print("\n" + "=" * 60)
    print("Example 6: Custom Point Cloud Processing")
    print("=" * 60)
    
    # Create a custom point cloud (spiral)
    def create_spiral(num_points=1024):
        t = np.linspace(0, 4*np.pi, num_points)
        x = t * np.cos(t) / (4*np.pi)
        y = t * np.sin(t) / (4*np.pi)
        z = t / (4*np.pi)
        return np.column_stack([x, y, z]).astype(np.float32)
    
    point_cloud = create_spiral(1024)
    point_cloud = normalize_point_cloud(point_cloud)
    
    # Save as different formats for demonstration
    np.save('sample_spiral.npy', point_cloud)
    np.savetxt('sample_spiral.txt', point_cloud)
    
    print(f"Created and saved spiral point cloud: {point_cloud.shape}")
    
    # Load and process
    from pointnet2_feature_extractor import load_point_cloud
    
    loaded_pc = load_point_cloud('sample_spiral.npy')
    print(f"Loaded point cloud: {loaded_pc.shape}")
    
    # Extract features
    model = PointNet2FeatureExtractor(num_classes=40, use_msg=False, use_xyz=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = extract_features(model, loaded_pc, device)
    
    print("\nCustom Point Cloud Features:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
    
    # Save features
    np.save('spiral_features.npy', features)
    print("Features saved as 'spiral_features.npy'")
    
    return features


def main():
    """Run all examples"""
    print("PointNet2 Feature Extractor - Example Usage")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Run examples
        example_1_basic_feature_extraction()
        example_2_multi_scale_grouping()
        example_3_with_normals()
        example_4_batch_processing()
        example_5_feature_visualization()
        example_6_custom_point_cloud()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
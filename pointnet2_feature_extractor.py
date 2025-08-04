#!/usr/bin/env python3
"""
PointNet2 Feature Extractor
===========================

This script implements a PointNet2-based feature extractor for 3D point clouds.
Based on the architecture from "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"

Key Features:
- Set Abstraction modules for hierarchical feature learning
- Support for both Single-Scale Grouping (SSG) and Multi-Scale Grouping (MSG)
- Feature extraction at multiple scales
- Support for both classification and segmentation tasks

Usage:
    python pointnet2_feature_extractor.py --input_file point_cloud.npy --output_file features.npy
    
Requirements:
    - PyTorch
    - NumPy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from typing import Tuple, Optional, List
import time


def timeit(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate Euclidean distance between each two points.
    
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
        
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index points according to the indices.
    
    Args:
        points: input points data, [B, N, C]
        idx: sample indices data, [B, S]
        
    Returns:
        new_points: indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) algorithm.
    
    Args:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
        
    Returns:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query for local neighborhood.
    
    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
        
    Returns:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, 
                     points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and group points.
    
    Args:
        npoint: number of sample points
        radius: search radius in local region
        nsample: max sample number in local region
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
        
    Returns:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    
    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and group all points (for global feature extraction).
    
    Args:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
        
    Returns:
        new_xyz: sampled points position data, [B, 1, N, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """PointNet Set Abstraction Module"""
    
    def __init__(self, npoint: Optional[int], radius: float, nsample: int, 
                 in_channel: int, mlp: List[int], group_all: bool = False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            
        Returns:
            new_xyz: sampled points position data, [B, C, S]
            new_points: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """PointNet Set Abstraction Module with Multi-Scale Grouping"""
    
    def __init__(self, npoint: int, radius_list: List[float], nsample_list: List[int], 
                 in_channel: int, mlp_list: List[List[int]]):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
    
    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            
        Returns:
            new_xyz: sampled points position data, [B, C, S]
            new_points: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)
        
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNet2FeatureExtractor(nn.Module):
    """
    PointNet2 Feature Extractor for point cloud processing.
    
    This model implements the hierarchical feature learning approach from PointNet++,
    which can extract both local and global features from point clouds.
    """
    
    def __init__(self, num_classes: int = 40, use_msg: bool = False, use_xyz: bool = True):
        super(PointNet2FeatureExtractor, self).__init__()
        self.use_msg = use_msg
        self.use_xyz = use_xyz
        
        in_channel = 6 if use_xyz else 3
        
        if use_msg:
            # Multi-Scale Grouping version
            self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 
                                                 in_channel - 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
            self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 
                                                 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
            self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        else:
            # Single-Scale Grouping version
            self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                              in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
            self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, 
                                              in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
            self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, 
                                              in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # Classification head (optional)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, xyz: torch.Tensor, return_features: bool = True) -> dict:
        """
        Forward pass of the feature extractor.
        
        Args:
            xyz: Input point cloud [B, N, 3] or [B, N, 6] if normals included
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - 'global_features': Global features [B, 1024]
                - 'sa1_features': First level features [B, 128, 512]
                - 'sa2_features': Second level features [B, 256, 128] 
                - 'classification': Classification logits [B, num_classes]
                - 'sa1_xyz': First level coordinates [B, 3, 512]
                - 'sa2_xyz': Second level coordinates [B, 3, 128]
        """
        B, N, C = xyz.shape
        
        if self.use_xyz and C > 3:
            norm = xyz[:, :, 3:]
            xyz_only = xyz[:, :, :3]
        else:
            norm = None
            xyz_only = xyz[:, :, :3]
        
        # Prepare input
        if norm is not None:
            l0_points = norm.transpose(2, 1).contiguous()
        else:
            l0_points = None
        l0_xyz = xyz_only.transpose(2, 1).contiguous()
        
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global features
        global_features = l3_points.view(B, 1024)
        
        # Classification
        x = self.drop1(F.relu(self.bn1(self.fc1(global_features))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        classification = self.fc3(x)
        
        result = {
            'global_features': global_features,
            'classification': classification
        }
        
        if return_features:
            result.update({
                'sa1_features': l1_points,
                'sa2_features': l2_points,
                'sa1_xyz': l1_xyz,
                'sa2_xyz': l2_xyz,
                'sa3_xyz': l3_xyz
            })
        
        return result


def load_point_cloud(file_path: str) -> np.ndarray:
    """
    Load point cloud from various file formats.
    
    Args:
        file_path: Path to the point cloud file
        
    Returns:
        Point cloud array of shape [N, 3] or [N, 6]
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        return np.load(file_path)
    elif ext == '.txt':
        return np.loadtxt(file_path)
    elif ext == '.ply':
        # Simple PLY loader (you might want to use a proper library like open3d)
        points = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            data_start = False
            for line in lines:
                if line.startswith('end_header'):
                    data_start = True
                    continue
                if data_start:
                    coords = line.strip().split()
                    if len(coords) >= 3:
                        points.append([float(x) for x in coords[:6]])  # x, y, z, nx, ny, nz
        return np.array(points)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def normalize_point_cloud(pc: np.ndarray) -> np.ndarray:
    """
    Normalize point cloud to unit sphere.
    
    Args:
        pc: Point cloud array [N, C]
        
    Returns:
        Normalized point cloud
    """
    xyz = pc[:, :3]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    xyz = xyz / m
    
    if pc.shape[1] > 3:
        return np.concatenate([xyz, pc[:, 3:]], axis=1)
    else:
        return xyz


def sample_point_cloud(pc: np.ndarray, num_points: int = 1024) -> np.ndarray:
    """
    Sample or pad point cloud to desired number of points.
    
    Args:
        pc: Point cloud array [N, C]
        num_points: Target number of points
        
    Returns:
        Sampled point cloud [num_points, C]
    """
    N = pc.shape[0]
    if N >= num_points:
        # Random sampling
        indices = np.random.choice(N, num_points, replace=False)
        return pc[indices]
    else:
        # Pad with repetition
        indices = np.random.choice(N, num_points, replace=True)
        return pc[indices]


@timeit
def extract_features(model: PointNet2FeatureExtractor, point_cloud: np.ndarray, 
                    device: str = 'cuda', batch_size: int = 1) -> dict:
    """
    Extract features from point cloud using PointNet2.
    
    Args:
        model: PointNet2 feature extractor model
        point_cloud: Point cloud array [N, C]
        device: Device to run inference on
        batch_size: Batch size for processing
        
    Returns:
        Dictionary containing extracted features
    """
    model.eval()
    model.to(device)
    
    # Prepare input
    pc_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(pc_tensor, return_features=True)
    
    # Convert to numpy
    result = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.cpu().numpy()
        else:
            result[key] = value
    
    return result


def create_sample_point_cloud(num_points: int = 1024, add_normals: bool = False) -> np.ndarray:
    """
    Create a sample point cloud for testing.
    
    Args:
        num_points: Number of points to generate
        add_normals: Whether to add normal vectors
        
    Returns:
        Sample point cloud array
    """
    # Generate points on a unit sphere
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    points = np.column_stack([x, y, z])
    
    if add_normals:
        # For a sphere, normals are the same as positions
        normals = points.copy()
        points = np.column_stack([points, normals])
    
    return points.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='PointNet2 Feature Extractor')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input point cloud file (.npy, .txt, .ply)')
    parser.add_argument('--output_file', type=str, default='features.npy',
                        help='Output features file')
    parser.add_argument('--model_type', type=str, choices=['ssg', 'msg'], default='ssg',
                        help='Model type: ssg (Single-Scale Grouping) or msg (Multi-Scale Grouping)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample from input')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--use_normals', action='store_true',
                        help='Use normal vectors if available')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with sample point cloud')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PointNet2 Feature Extractor")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model type: {args.model_type.upper()}")
    print(f"Number of points: {args.num_points}")
    
    # Create model
    use_msg = args.model_type == 'msg'
    model = PointNet2FeatureExtractor(num_classes=40, use_msg=use_msg, use_xyz=args.use_normals)
    
    if args.demo or args.input_file is None:
        print("\nRunning demo with sample point cloud...")
        point_cloud = create_sample_point_cloud(args.num_points, args.use_normals)
        print(f"Generated sample point cloud: {point_cloud.shape}")
    else:
        print(f"\nLoading point cloud from: {args.input_file}")
        point_cloud = load_point_cloud(args.input_file)
        print(f"Original point cloud shape: {point_cloud.shape}")
        
        # Normalize and sample
        point_cloud = normalize_point_cloud(point_cloud)
        point_cloud = sample_point_cloud(point_cloud, args.num_points)
        print(f"Processed point cloud shape: {point_cloud.shape}")
    
    # Extract features
    print("\nExtracting features...")
    features = extract_features(model, point_cloud, args.device)
    
    print("\nExtracted features:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Save features
    print(f"\nSaving features to: {args.output_file}")
    np.save(args.output_file, features)
    
    # Print feature statistics
    global_features = features['global_features'][0]  # Remove batch dimension
    print(f"\nGlobal feature statistics:")
    print(f"  Shape: {global_features.shape}")
    print(f"  Min: {global_features.min():.4f}")
    print(f"  Max: {global_features.max():.4f}")
    print(f"  Mean: {global_features.mean():.4f}")
    print(f"  Std: {global_features.std():.4f}")
    
    # Classification results
    if 'classification' in features:
        classification = features['classification'][0]
        predicted_class = np.argmax(classification)
        confidence = np.max(classification)
        print(f"\nClassification results:")
        print(f"  Predicted class: {predicted_class}")
        print(f"  Confidence: {confidence:.4f}")
    
    print("\nFeature extraction completed successfully!")


if __name__ == "__main__":
    main()
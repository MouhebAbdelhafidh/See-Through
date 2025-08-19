import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import h5py
import torch

def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS) for point clouds.
    Args:
        xyz: (B, N, 2) point coordinates
        npoint: number of samples
    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, xyz.shape[2])
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Gather points using indices.
    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K) indices
    Returns:
        new_points: (B, S, C) or (B, S, K, C)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def ball_query(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, S, _ = new_xyz.shape
    N = xyz.shape[1]

    # Compute pairwise distances (B, S, N)
    dist = torch.cdist(new_xyz, xyz)

    # Initialize group_idx with all indices (B, S, N)
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).expand(B, S, N)

    # Create mask for points beyond radius (B, S, N)
    mask = dist > radius

    # Replace masked indices with first point's index
    group_first = group_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, N)
    group_idx = torch.where(mask, group_first, group_idx)

    # Sort by distance and take first nsample points
    dist_sorted, sort_idx = torch.sort(dist, dim=-1)
    group_idx = torch.gather(group_idx, -1, sort_idx)[:, :, :nsample]

    return group_idx

class PointNetSetAbstraction(nn.Module):
    """Fixed Set Abstraction Layer"""

    def __init__(self, npoint, radius, nsample, in_channels, out_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        # MLP layers with batch normalization
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, xyz, features):
        B, N, _ = xyz.shape
        
        # Farthest Point Sampling
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        
        # Ball Query
        group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        
        # Grouping
        grouped_xyz = index_points(xyz, group_idx)  # (B, npoint, nsample, 2)
        grouped_features = index_points(features, group_idx)  # (B, npoint, nsample, feature_dim)
        
        # Normalize coordinates
        grouped_xyz -= new_xyz.unsqueeze(2)
        
        # Combine features
        combined = torch.cat([grouped_xyz, grouped_features], dim=-1)  # (B, npoint, nsample, 2 + feature_dim)
        combined = combined.permute(0, 3, 1, 2)  # (B, channels, npoint, nsample)
        
        # Process with MLP
        new_features = self.mlp(combined)
        new_features = torch.max(new_features, 3)[0]  # (B, out_channels, npoint)
        
        return new_xyz, new_features.permute(0, 2, 1)  # (B, npoint, out_channels)

class PointNet2Backbone(nn.Module):
    """Final Working Backbone"""
    def __init__(self, feature_channels=5):
        super().__init__()
        # SA1: 2 (xyz) + 5 (features) = 7 input channels
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channels=2 + feature_channels, out_channels=128
        )
        
        # SA2: 128 (features) + 2 (xyz) = 130 input channels
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channels=2 + 128, out_channels=256
        )
        
        # SA3: 256 + 2 = 258 input channels
        self.sa3 = PointNetSetAbstraction(
            npoint=1, radius=100, nsample=256,
            in_channels=2 + 256, out_channels=1024
        )
        
    def forward(self, xyz, features):
        # Input verification
        print(f"Input shapes - xyz: {xyz.shape}, features: {features.shape}")
        
        l0_xyz = xyz
        l0_features = features
        
        # SA1
        l1_xyz, l1_features = self.sa1(l0_xyz, l0_features)
        
        # SA2
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        
        # SA3
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        
        return l3_features.squeeze(1), l1_features, {
            'l0_xyz': l0_xyz,
            'l0_features': l0_features,
            'l1_xyz': l1_xyz,
            'l1_features': l1_features,
            'l2_xyz': l2_xyz,
            'l2_features': l2_features,
            'l3_xyz': l3_xyz,
            'l3_features': l3_features
        }

# PointNet2FeatureExtractor = PointNet2Backbone

# file_path = "FusedData/sequence_2_fused.h5"

# if not os.path.exists(file_path):
#     print(f"File not found: {file_path}")
#     exit()

# model = PointNet2FeatureExtractor(feature_channels=3)
# model.eval()

# features_list = []
# selected_fields = ['x_cc', 'y_cc', 'vr', 'vr_compensated', 'rcs']

# inputs_info = []  

# with h5py.File(file_path, 'r') as f:
#     fused_detections = f['fused_detections']
#     for i in range(len(fused_detections)):
#         record = fused_detections[i]
        
#         # Get original input: (x_cc, y_cc, vr, vr_compensated, rcs)
#         point = np.array([record[field] for field in selected_fields])  # shape (5,)
#         inputs_info.append(point.copy())  # store input for traceability
        
#         xyz_np = point[:2].reshape(1, 2)
#         feat_np = point[2:].reshape(1, 3)
        
#         xyz = torch.tensor(xyz_np, dtype=torch.float32).unsqueeze(0)
#         features = torch.tensor(feat_np, dtype=torch.float32).unsqueeze(0)
        
#         with torch.no_grad():
#             output, _, _ = model(xyz, features)
        
#         features_list.append(output.squeeze(0))

# # Stack results
# all_features = torch.stack(features_list)
# inputs_info = np.stack(inputs_info)  # shape (num_points, 5)
# print("Feature vector 0:")
# print(all_features[0])

# print("\nInput used for feature vector 0:")
# print(f"x_cc = {inputs_info[0][0]:.4f}")
# print(f"y_cc = {inputs_info[0][1]:.4f}")
# print(f"vr = {inputs_info[0][2]:.4f}")
# print(f"vr_compensated = {inputs_info[0][3]:.4f}")
# print(f"rcs = {inputs_info[0][4]:.4f}")

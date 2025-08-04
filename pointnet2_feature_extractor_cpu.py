import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from pathlib import Path

# CPU-only PointNet2 implementation
def ball_query(xyz, new_xyz, radius, nsample):
    """
    Ball query implemented in PyTorch (CPU version)
    """
    sqrdists = torch.cdist(new_xyz, xyz)
    _, group_idx = torch.topk(-sqrdists, nsample, dim=-1)
    mask = sqrdists.gather(-1, group_idx) < radius**2
    group_idx[~mask] = 0
    return group_idx

def farthest_point_sample(xyz, npoint):
    """
    Farthest point sampling implemented in PyTorch
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

def index_points(points, idx):
    """
    Index points using indices
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

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sampling and grouping for set abstraction layer
    """
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    
    idx = ball_query(xyz, new_xyz, radius, nsample)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Sample all points and group for global feature
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
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
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

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
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
        
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        
        # Adjust the first conv layer to handle the actual input channels
        if len(self.mlp_convs) > 0:
            # Create a temporary conv layer for the first layer
            temp_conv = nn.Conv2d(new_points.shape[1], self.mlp_convs[0].out_channels, 1).to(new_points.device)
            new_points = F.relu(self.mlp_bns[0](temp_conv(new_points)))
            
            # Apply remaining layers
            for i in range(1, len(self.mlp_convs)):
                conv = self.mlp_convs[i]
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
        else:
            # No conv layers, just max pooling
            pass
        
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = torch.cdist(xyz1, xyz2)
            dists, idx = torch.topk(dists, 3, dim=-1, largest=False)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
            
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        return new_points

class PointNet2Backbone(nn.Module):
    def __init__(self, spatial_dims=3, feature_dims=3, feature_dim=128):
        super().__init__()
        # First SA layer processes feature_dims channels
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=feature_dims, mlp=[64, 64, 128], group_all=False
        )
        # Second SA layer
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        # Third SA layer (global)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, feature_dim], group_all=True
        )
        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=feature_dim + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])

    def forward(self, xyz, features):
        # Set Abstraction Layers
        l1_xyz, l1_features = self.sa1(xyz, features)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        
        # Feature Propagation Layers
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        return l1_features

def preprocess_point_cloud(points, normalize=True, center=True):
    """
    Preprocess point cloud data
    Args:
        points: Point cloud data [N, D]
        normalize: Whether to normalize coordinates
        center: Whether to center the point cloud
    Returns:
        processed_points: Processed point cloud
        center_coords: Center coordinates (if centering was applied)
    """
    if center:
        center_coords = np.mean(points[:, :3], axis=0)
        points[:, :3] -= center_coords
    else:
        center_coords = np.zeros(3)
    
    if normalize:
        # Normalize to unit sphere
        max_dist = np.max(np.linalg.norm(points[:, :3], axis=1))
        if max_dist > 0:
            points[:, :3] /= max_dist
    
    return points, center_coords

def extract_features_from_file(model, input_file, output_dir, device='cpu'):
    """
    Extract features from a single HDF5 file
    """
    print(f"Processing {input_file}...")
    
    # Load data
    with h5py.File(input_file, 'r') as f:
        if 'fused_detections' in f:
            data = f['fused_detections'][:]
        elif 'points' in f:
            data = f['points'][:]
        else:
            # Try to find any dataset
            key = list(f.keys())[0]
            data = f[key][:]
    
    # Convert structured array if needed
    if hasattr(data, 'dtype') and data.dtype.names:
        data = np.array([list(item) for item in data])
    
    # Ensure proper shape
    if len(data.shape) == 1:
        data = data.reshape(-1, data.shape[0])
    
    # Define which dimensions to use
    spatial_dims = data[:, :3]  # x, y, z coordinates
    feature_dims = data[:, [3, 4, 7]] if data.shape[1] >= 8 else data[:, 3:6]  # Use available features
    
    # Preprocess point cloud
    processed_points, center_coords = preprocess_point_cloud(
        np.concatenate([spatial_dims, feature_dims], axis=1)
    )
    
    # Convert to tensor
    xyz = torch.tensor(spatial_dims, dtype=torch.float32).unsqueeze(0).transpose(2, 1)
    features = torch.tensor(feature_dims, dtype=torch.float32).unsqueeze(0).transpose(2, 1)
    
    # Move to device
    xyz = xyz.to(device)
    features = features.to(device)
    model = model.to(device)
    
    # Extract features
    with torch.no_grad():
        point_features = model(xyz, features)
    
    # Save features
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_file))
    
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('features', data=point_features.squeeze(0).transpose(1, 0).cpu().numpy())
        f_out.create_dataset('original_points', data=data)
        f_out.create_dataset('center', data=center_coords)
        f_out.create_dataset('processed_points', data=processed_points)
        f_out.attrs['feature_dim'] = point_features.shape[1]
        f_out.attrs['num_points'] = point_features.shape[2]
    
    print(f"Saved features to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Extract features using PointNet2 (CPU version)')
    parser.add_argument('--input_dir', type=str, default='FusedData', 
                       help='Input directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, default='ExtractedFeatures', 
                       help='Output directory for extracted features')
    parser.add_argument('--feature_dim', type=int, default=128, 
                       help='Dimension of extracted features')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to pre-trained model weights')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Initialize model
    model = PointNet2Backbone(spatial_dims=3, feature_dims=3, feature_dim=args.feature_dim)
    
    # Load pre-trained weights if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pre-trained weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    
    model.eval()
    
    # Process files
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all HDF5 files
    h5_files = list(input_dir.glob("*.h5")) + list(input_dir.glob("*.hdf5"))
    
    if not h5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(h5_files)} HDF5 files to process")
    
    processed = 0
    for file_path in tqdm(h5_files, desc="Processing files"):
        try:
            extract_features_from_file(
                model, str(file_path), str(output_dir), args.device
            )
            processed += 1
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Save model
    model_save_path = output_dir / "pointnet2_feature_extractor_cpu.pth"
    torch.save(model.state_dict(), model_save_path)
    
    print(f"\nSuccessfully processed {processed}/{len(h5_files)} files")
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List


class FarthestPointSample(nn.Module):
    """
    Farthest Point Sampling (FPS) implementation
    """
    def __init__(self, npoint: int):
        super(FarthestPointSample, self).__init__()
        self.npoint = npoint

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Input:
            xyz: pointcloud data, [B, N, C]
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        
        centroids = torch.zeros(B, self.npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        
        # Randomly select first point
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        
        for i in range(self.npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        
        return centroids


class BallQuery(nn.Module):
    """
    Ball Query for grouping points within a radius
    """
    def __init__(self, radius: float, nsample: int):
        super(BallQuery, self).__init__()
        self.radius = radius
        self.nsample = nsample

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        Input:
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        B, N, C = xyz.shape
        S = new_xyz.shape[1]
        
        # Calculate pairwise distances
        sqrdists = torch.cdist(new_xyz, xyz)  # [B, S, N]
        
        # Get the nsample closest points
        _, group_idx = torch.topk(-sqrdists, self.nsample, dim=-1)
        
        # Create mask for points within radius
        mask = sqrdists.gather(-1, group_idx) < self.radius ** 2
        
        # Replace points outside radius with first point
        group_idx[~mask] = group_idx[:, :, 0:1].expand(-1, -1, self.nsample)[~mask]
        
        return group_idx


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Layer
    """
    def __init__(self, npoint: int, radius: float, nsample: int, 
                 in_channel: int, mlp: List[int], group_all: bool = False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # MLP layers
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]

        if self.group_all:
            new_xyz, new_points = self._sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self._sample_and_group(xyz, points)
        
        # MLP
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, C, S]
        return new_xyz, new_points

    def _sample_and_group(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample and group points
        """
        fps = FarthestPointSample(self.npoint)
        ball_query = BallQuery(self.radius, self.nsample)
        
        new_xyz = index_points(xyz, fps(xyz))  # [B, npoint, C]
        idx = ball_query(xyz, new_xyz)  # [B, npoint, nsample]
        
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, xyz.shape[2])
        
        if points is not None:
            grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm
        
        return new_xyz, new_points

    def _sample_and_group_all(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample and group all points (for global feature)
        """
        device = xyz.device
        B, N, C = xyz.shape
        
        new_xyz = torch.zeros(B, 1, C).to(device)
        grouped_xyz = xyz.view(B, 1, N, C)
        
        if points is not None:
            grouped_points = points.view(B, 1, N, -1)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
        
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation Layer
    """
    def __init__(self, in_channel: int, mlp: List[int]):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, 
                points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        """
        Input:
            xyz1: target points position data, [B, C, N]
            xyz2: source points position data, [B, C, S]
            points1: target points data, [B, D, N]
            points2: source points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, C]
        points1 = points1.permute(0, 2, 1)  # [B, N, D]
        points2 = points2.permute(0, 2, 1)  # [B, S, D]
        
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # Find 3 nearest neighbors
            dists = torch.cdist(xyz1, xyz2)  # [B, N, S]
            dists, idx = torch.topk(dists, 3, dim=-1, largest=False)  # [B, N, 3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        new_points = new_points.permute(0, 2, 1)  # [B, D', N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
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


class PointNetPlusPlus(nn.Module):
    """
    Complete PointNet++ implementation for radar point cloud processing
    """
    def __init__(self, num_classes: int = 13, feature_dim: int = 128, 
                 spatial_dims: int = 2, feature_dims: int = 5):
        super(PointNetPlusPlus, self).__init__()
        
        self.spatial_dims = spatial_dims  # x_cc, y_cc
        self.feature_dims = feature_dims  # rcs, vr, vr_compensated, num_merged, sensor_id
        
        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=spatial_dims + feature_dims, mlp=[64, 64, 128]
        )
        
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + spatial_dims, mlp=[128, 128, 256]
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + spatial_dims, mlp=[256, 512, 1024], group_all=True
        )
        
        # Feature Propagation layers
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + feature_dims, mlp=[128, 128, 128])
        
        # Classification head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        
        # Segmentation head
        self.seg_conv1 = nn.Conv1d(128, 128, 1)
        self.seg_bn1 = nn.BatchNorm1d(128)
        self.seg_conv2 = nn.Conv1d(128, 128, 1)
        self.seg_bn2 = nn.BatchNorm1d(128)
        self.seg_conv3 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            xyz: point coordinates, [B, spatial_dims, N]
            features: point features, [B, feature_dims, N]
        Return:
            classification_output: [B, num_classes]
            segmentation_output: [B, num_classes, N]
        """
        # Combine spatial and feature dimensions
        points = torch.cat([xyz, features], dim=1)  # [B, spatial_dims + feature_dims, N]
        
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, features, l1_points)
        
        # Classification head
        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)
        x = self.conv2(x)
        classification_output = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        
        # Segmentation head
        seg_x = F.relu(self.seg_bn1(self.seg_conv1(l0_points)))
        seg_x = F.relu(self.seg_bn2(self.seg_conv2(seg_x)))
        segmentation_output = self.seg_conv3(seg_x)
        
        return classification_output, segmentation_output


class RadarPointCloudDataset(Dataset):
    """
    Dataset class for radar point cloud data
    """
    def __init__(self, h5_file_path: str, max_points: int = 1024, 
                 features: List[str] = ['rcs', 'vr', 'vr_compensated', 'num_merged', 'sensor_id']):
        self.h5_file_path = h5_file_path
        self.max_points = max_points
        self.features = features
        
        # Load data
        with h5py.File(h5_file_path, 'r') as f:
            self.frames = f['frames'][:]
            self.detections = f['detections'][:]
        
        # Create frame indices
        self.frame_indices = []
        for i, frame in enumerate(self.frames):
            start_idx = frame['detection_start_idx']
            end_idx = frame['detection_end_idx']
            num_points = end_idx - start_idx
            
            if num_points > 0:
                self.frame_indices.append(i)

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        frame_idx = self.frame_indices[idx]
        frame = self.frames[frame_idx]
        
        start_idx = frame['detection_start_idx']
        end_idx = frame['detection_end_idx']
        
        # Get point data
        points = self.detections[start_idx:end_idx]
        
        # Extract features
        xyz = np.stack([points['x_cc'], points['y_cc']], axis=1)  # [N, 2]
        
        feature_list = []
        for feature in self.features:
            if feature == 'num_merged':
                feature_list.append(points[feature].astype(np.float32))
            else:
                feature_list.append(points[feature])
        
        features = np.stack(feature_list, axis=1)  # [N, feature_dims]
        labels = points['label_id']  # [N]
        
        # Handle variable number of points
        num_points = len(points)
        if num_points > self.max_points:
            # Randomly sample points
            indices = np.random.choice(num_points, self.max_points, replace=False)
            xyz = xyz[indices]
            features = features[indices]
            labels = labels[indices]
        elif num_points < self.max_points:
            # Pad with zeros
            pad_size = self.max_points - num_points
            xyz = np.pad(xyz, ((0, pad_size), (0, 0)), mode='constant')
            features = np.pad(features, ((0, pad_size), (0, 0)), mode='constant')
            labels = np.pad(labels, (0, pad_size), mode='constant', constant_values=-1)
        
        # Convert to tensors
        xyz = torch.FloatTensor(xyz).transpose(0, 1)  # [2, N]
        features = torch.FloatTensor(features).transpose(0, 1)  # [feature_dims, N]
        labels = torch.LongTensor(labels)  # [N]
        
        return xyz, features, labels


def train_pointnet_plus_plus(model: PointNetPlusPlus, train_loader: DataLoader, 
                           num_epochs: int = 100, learning_rate: float = 0.001,
                           device: str = 'cuda'):
    """
    Training function for PointNet++
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    segmentation_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (xyz, features, labels) in enumerate(train_loader):
            xyz = xyz.to(device)
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            classification_output, segmentation_output = model(xyz, features)
            
            # Calculate losses
            # For classification, use the most common label in the point cloud
            unique_labels, counts = torch.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                most_common_label = unique_labels[torch.argmax(counts)]
                classification_loss = classification_criterion(classification_output, most_common_label.unsqueeze(0))
            else:
                classification_loss = torch.tensor(0.0).to(device)
            
            # For segmentation, use all valid labels
            valid_mask = labels != -1
            if valid_mask.sum() > 0:
                segmentation_loss = segmentation_criterion(
                    segmentation_output[:, :, valid_mask], 
                    labels[valid_mask]
                )
            else:
                segmentation_loss = torch.tensor(0.0).to(device)
            
            # Total loss
            loss = classification_loss + segmentation_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')


def main():
    """
    Example usage of PointNet++ for radar point cloud processing
    """
    # Model parameters
    num_classes = 13  # Number of object classes in RadarScenes
    feature_dim = 128
    spatial_dims = 2  # x_cc, y_cc
    feature_dims = 5  # rcs, vr, vr_compensated, num_merged, sensor_id
    
    # Create model
    model = PointNetPlusPlus(
        num_classes=num_classes,
        feature_dim=feature_dim,
        spatial_dims=spatial_dims,
        feature_dims=feature_dims
    )
    
    print("PointNet++ model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass
    batch_size = 4
    num_points = 1024
    
    xyz = torch.randn(batch_size, spatial_dims, num_points)
    features = torch.randn(batch_size, feature_dims, num_points)
    
    classification_output, segmentation_output = model(xyz, features)
    
    print(f"Classification output shape: {classification_output.shape}")
    print(f"Segmentation output shape: {segmentation_output.shape}")
    
    # Example dataset usage
    # dataset = RadarPointCloudDataset('path_to_your_data.h5')
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # train_pointnet_plus_plus(model, dataloader)


if __name__ == "__main__":
    main()
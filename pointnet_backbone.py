import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class PointNetPlusPlus(nn.Module):
    """
    PointNet++ backbone for radar point cloud feature extraction.
    
    Input data format:
    - xyz: point coordinates [x_cc, y_cc, 0] (we add z=0 for 2D radar data)
    - features: [rcs, vr, vr_compensated, num_merged, sensor_id]
    - labels: label_id
    
    The model extracts hierarchical features that can be used with:
    - VoteNet head for classification
    - PointNet++ head for segmentation
    """
    
    def __init__(self, num_features=5, num_classes=None, use_xyz=True):
        super(PointNetPlusPlus, self).__init__()
        
        self.num_features = num_features  # rcs, vr, vr_compensated, num_merged, sensor_id
        self.num_classes = num_classes
        self.use_xyz = use_xyz
        
        # Input channel: 3 (xyz) + num_features if use_xyz, else just num_features
        input_channel = 3 + num_features if use_xyz else num_features
        
        # Set Abstraction layers for hierarchical feature extraction
        self.sa1 = PointNetSetAbstraction(
            npoint=512, 
            radius=0.2, 
            nsample=32, 
            in_channel=input_channel, 
            mlp=[64, 64, 128], 
            group_all=False
        )
        
        self.sa2 = PointNetSetAbstraction(
            npoint=128, 
            radius=0.4, 
            nsample=64, 
            in_channel=128 + 3, 
            mlp=[128, 128, 256], 
            group_all=False
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None, 
            in_channel=256 + 3, 
            mlp=[256, 512, 1024], 
            group_all=True
        )
        
        # Feature Propagation layers for upsampling (useful for segmentation)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + input_channel, mlp=[128, 128, 128])
        
        # Classification head (optional, for direct classification)
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, xyz, features, return_intermediate=False):
        """
        Forward pass of PointNet++
        
        Args:
            xyz: Point coordinates [B, 3, N] where 3rd dim is 0 for 2D radar
            features: Point features [B, num_features, N]
            return_intermediate: If True, return intermediate features for segmentation
            
        Returns:
            If return_intermediate=False: global features [B, 1024]
            If return_intermediate=True: tuple of (global_features, intermediate_features)
        """
        B, _, N = xyz.shape
        
        # Combine xyz and features if using xyz
        if self.use_xyz:
            points = torch.cat([xyz, features], dim=1)  # [B, 3+num_features, N]
        else:
            points = features
            
        # Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global features
        global_features = l3_points.view(B, -1)  # [B, 1024]
        
        if return_intermediate:
            # Feature Propagation for segmentation
            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(xyz, l1_xyz, points, l1_points)
            
            intermediate_features = {
                'l0_xyz': xyz,
                'l0_points': l0_points,
                'l1_xyz': l1_xyz,
                'l1_points': l1_points,
                'l2_xyz': l2_xyz,
                'l2_points': l2_points,
                'l3_xyz': l3_xyz,
                'l3_points': l3_points,
                'global_features': global_features
            }
            
            return global_features, intermediate_features
        
        return global_features
    
    def get_loss(self, pred, target, smoothing=False):
        """Compute classification loss"""
        if smoothing:
            eps = 0.2
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, target, reduction='mean')
        return loss


class RadarPointNetPlusPlus(PointNetPlusPlus):
    """
    Specialized PointNet++ for radar data processing.
    Handles the specific format of radar point clouds.
    """
    
    def __init__(self, num_classes=None, normalize_features=True):
        super(RadarPointNetPlusPlus, self).__init__(
            num_features=5,  # rcs, vr, vr_compensated, num_merged, sensor_id
            num_classes=num_classes,
            use_xyz=True
        )
        self.normalize_features = normalize_features
        
        # Feature normalization layers
        if normalize_features:
            self.feature_norm = nn.BatchNorm1d(5)
    
    def preprocess_radar_data(self, data_dict):
        """
        Preprocess radar data from the .h5 format
        
        Args:
            data_dict: Dictionary containing:
                - x_cc, y_cc: coordinates
                - rcs, vr, vr_compensated, num_merged, sensor_id: features
                - label_id: labels (optional)
        
        Returns:
            xyz: [B, 3, N] coordinates with z=0
            features: [B, 5, N] normalized features
            labels: [B, N] labels if provided
        """
        device = next(self.parameters()).device
        
        # Extract coordinates (add z=0 for 2D radar data)
        x_cc = data_dict['x_cc']
        y_cc = data_dict['y_cc']
        z_cc = torch.zeros_like(x_cc)  # Add z=0 for 3D compatibility
        
        xyz = torch.stack([x_cc, y_cc, z_cc], dim=1)  # [B, 3, N]
        
        # Extract features
        features = torch.stack([
            data_dict['rcs'],
            data_dict['vr'],
            data_dict['vr_compensated'],
            data_dict['num_merged'].float(),
            data_dict['sensor_id'].float()
        ], dim=1)  # [B, 5, N]
        
        # Normalize features if enabled
        if self.normalize_features:
            B, C, N = features.shape
            features = features.view(-1, C)
            features = self.feature_norm(features)
            features = features.view(B, C, N)
        
        # Extract labels if available
        labels = data_dict.get('label_id', None)
        
        return xyz.to(device), features.to(device), labels.to(device) if labels is not None else None
    
    def forward(self, data_dict, return_intermediate=False):
        """
        Forward pass for radar data
        
        Args:
            data_dict: Dictionary containing radar data
            return_intermediate: Whether to return intermediate features
            
        Returns:
            Global features or (global_features, intermediate_features)
        """
        xyz, features, labels = self.preprocess_radar_data(data_dict)
        return super().forward(xyz, features, return_intermediate)


def get_radar_pointnet_model(num_classes=None, **kwargs):
    """
    Factory function to create a radar-specific PointNet++ model
    
    Args:
        num_classes: Number of classes for classification head
        **kwargs: Additional arguments for the model
        
    Returns:
        RadarPointNetPlusPlus model
    """
    return RadarPointNetPlusPlus(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Example usage
    model = RadarPointNetPlusPlus(num_classes=10)
    
    # Create dummy radar data
    batch_size = 4
    num_points = 1024
    
    dummy_data = {
        'x_cc': torch.randn(batch_size, num_points),
        'y_cc': torch.randn(batch_size, num_points),
        'rcs': torch.randn(batch_size, num_points),
        'vr': torch.randn(batch_size, num_points),
        'vr_compensated': torch.randn(batch_size, num_points),
        'num_merged': torch.randint(0, 100, (batch_size, num_points)),
        'sensor_id': torch.randint(0, 256, (batch_size, num_points)),
        'label_id': torch.randint(0, 10, (batch_size, num_points))
    }
    
    # Forward pass
    global_features = model(dummy_data)
    print(f"Global features shape: {global_features.shape}")
    
    # Forward pass with intermediate features
    global_features, intermediate = model(dummy_data, return_intermediate=True)
    print(f"Global features shape: {global_features.shape}")
    print(f"Intermediate features keys: {intermediate.keys()}")
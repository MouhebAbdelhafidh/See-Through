#!/usr/bin/env python3
"""
Example script demonstrating how to use PointNet++ backbone for fusion with other architectures
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

# Import our PointNet++ backbone
from pointnet_plus_plus_backbone import PointNetPlusPlusBackbone, create_pointnet_plus_plus_model


class VotNetHead(nn.Module):
    """
    Example VotNet head that can be fused with PointNet++ backbone
    """
    def __init__(self, input_dim: int, num_classes: int = 13):
        super(VotNetHead, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # VotNet-style head
        self.vote_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3D vote
        )
        
        self.objectness_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Objectness score
        )
        
        self.classification_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for VotNet head
        
        Args:
            features: Input features [B, N, feature_dim]
        
        Returns:
            votes: 3D votes [B, N, 3]
            objectness: Objectness scores [B, N, 1]
            classification: Classification scores [B, N, num_classes]
        """
        votes = self.vote_mlp(features)
        objectness = self.objectness_mlp(features)
        classification = self.classification_mlp(features)
        
        return votes, objectness, classification


class FusedPointNetVotNet(nn.Module):
    """
    Example of fusing PointNet++ backbone with VotNet head
    """
    def __init__(self, spatial_dims: int = 2, feature_dims: int = 5, 
                 feature_dim: int = 128, num_classes: int = 13):
        super(FusedPointNetVotNet, self).__init__()
        
        # PointNet++ backbone
        self.pointnet_backbone = PointNetPlusPlusBackbone(
            spatial_dims=spatial_dims,
            feature_dims=feature_dims,
            feature_dim=feature_dim
        )
        
        # VotNet head
        self.votnet_head = VotNetHead(
            input_dim=feature_dim,
            num_classes=num_classes
        )
        
        # Additional fusion layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # Global + local features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            xyz: Point coordinates [B, spatial_dims, N]
            features: Point features [B, feature_dims, N]
        
        Returns:
            votes: 3D votes [B, N, 3]
            objectness: Objectness scores [B, N, 1]
            classification: Classification scores [B, N, num_classes]
        """
        # Extract features using PointNet++ backbone
        point_features, global_features, skip_connections = self.pointnet_backbone(xyz, features)
        
        # Transpose for VotNet head
        point_features = point_features.transpose(1, 2)  # [B, N, feature_dim]
        
        # Fuse global and local features
        global_features_expanded = global_features.unsqueeze(1).expand(-1, point_features.shape[1], -1)
        fused_features = torch.cat([point_features, global_features_expanded], dim=-1)
        fused_features = self.fusion_mlp(fused_features)
        
        # Pass through VotNet head
        votes, objectness, classification = self.votnet_head(fused_features)
        
        return votes, objectness, classification


class MultiTaskFusionModel(nn.Module):
    """
    Example of a multi-task model using PointNet++ backbone with different heads
    """
    def __init__(self, spatial_dims: int = 2, feature_dims: int = 5, 
                 feature_dim: int = 128, num_classes: int = 13):
        super(MultiTaskFusionModel, self).__init__()
        
        # Create PointNet++ backbone and heads
        self.backbone, self.classification_head, self.segmentation_head = create_pointnet_plus_plus_model(
            num_classes=num_classes,
            feature_dim=feature_dim,
            spatial_dims=spatial_dims,
            feature_dims=feature_dims,
            task='both'
        )
        
        # Additional VotNet-style head
        self.votnet_head = VotNetHead(
            input_dim=feature_dim,
            num_classes=num_classes
        )
        
        # Task weights for multi-task learning
        self.task_weights = nn.Parameter(torch.ones(3))  # classification, segmentation, votnet

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> dict:
        """
        Forward pass for multi-task model
        
        Args:
            xyz: Point coordinates [B, spatial_dims, N]
            features: Point features [B, feature_dims, N]
        
        Returns:
            Dictionary containing all outputs
        """
        # Get backbone features
        point_features, global_features, skip_connections = self.backbone(xyz, features)
        
        # Classification output
        classification_output = self.classification_head(xyz, features)
        
        # Segmentation output
        segmentation_output = self.segmentation_head(xyz, features)
        
        # VotNet output
        point_features_for_votnet = point_features.transpose(1, 2)  # [B, N, feature_dim]
        votes, objectness, votnet_classification = self.votnet_head(point_features_for_votnet)
        
        return {
            'classification': classification_output,
            'segmentation': segmentation_output,
            'votes': votes,
            'objectness': objectness,
            'votnet_classification': votnet_classification,
            'global_features': global_features,
            'point_features': point_features
        }


def example_usage():
    """Example usage of the fused models"""
    print("PointNet++ Fusion Examples")
    print("=" * 40)
    
    # Model parameters
    batch_size = 4
    num_points = 1024
    spatial_dims = 2  # x_cc, y_cc
    feature_dims = 5  # rcs, vr, vr_compensated, num_merged, sensor_id
    feature_dim = 128
    num_classes = 13
    
    # Create dummy data
    xyz = torch.randn(batch_size, spatial_dims, num_points)
    features = torch.randn(batch_size, feature_dims, num_points)
    
    print(f"Input shapes:")
    print(f"  xyz: {xyz.shape}")
    print(f"  features: {features.shape}")
    
    # Example 1: PointNet++ + VotNet fusion
    print("\n1. PointNet++ + VotNet Fusion:")
    fused_model = FusedPointNetVotNet(
        spatial_dims=spatial_dims,
        feature_dims=feature_dims,
        feature_dim=feature_dim,
        num_classes=num_classes
    )
    
    votes, objectness, classification = fused_model(xyz, features)
    print(f"  Votes shape: {votes.shape}")
    print(f"  Objectness shape: {objectness.shape}")
    print(f"  Classification shape: {classification.shape}")
    
    # Example 2: Multi-task model
    print("\n2. Multi-task Model:")
    multi_task_model = MultiTaskFusionModel(
        spatial_dims=spatial_dims,
        feature_dims=feature_dims,
        feature_dim=feature_dim,
        num_classes=num_classes
    )
    
    outputs = multi_task_model(xyz, features)
    print(f"  Classification: {outputs['classification'].shape}")
    print(f"  Segmentation: {outputs['segmentation'].shape}")
    print(f"  Votes: {outputs['votes'].shape}")
    print(f"  Objectness: {outputs['objectness'].shape}")
    print(f"  Global features: {outputs['global_features'].shape}")
    print(f"  Point features: {outputs['point_features'].shape}")
    
    # Example 3: Using backbone only
    print("\n3. Backbone Only:")
    backbone = PointNetPlusPlusBackbone(
        spatial_dims=spatial_dims,
        feature_dims=feature_dims,
        feature_dim=feature_dim
    )
    
    point_features, global_features, skip_connections = backbone(xyz, features)
    print(f"  Point features: {point_features.shape}")
    print(f"  Global features: {global_features.shape}")
    print(f"  Skip connections: {len(skip_connections)}")
    
    print("\n" + "=" * 40)
    print("Examples completed successfully!")
    print("\nYou can now use these models for your radar point cloud processing tasks.")


def create_custom_head_example():
    """Example of creating a custom head for the PointNet++ backbone"""
    print("\nCustom Head Example:")
    print("-" * 20)
    
    class CustomDetectionHead(nn.Module):
        """Custom detection head example"""
        def __init__(self, backbone, num_classes: int = 13):
            super().__init__()
            self.backbone = backbone
            feature_dim = backbone.get_feature_dim()
            
            # Detection-specific layers
            self.bbox_head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 7)  # 3D bbox (center + size + rotation)
            )
            
            self.confidence_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            self.classification_head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> dict:
            """Forward pass"""
            _, global_features, _ = self.backbone(xyz, features)
            
            bbox = self.bbox_head(global_features)
            confidence = self.confidence_head(global_features)
            classification = self.classification_head(global_features)
            
            return {
                'bbox': bbox,
                'confidence': confidence,
                'classification': classification
            }
    
    # Create backbone and custom head
    backbone = PointNetPlusPlusBackbone(
        spatial_dims=2,
        feature_dims=5,
        feature_dim=128
    )
    
    custom_head = CustomDetectionHead(backbone, num_classes=13)
    
    # Test with dummy data
    xyz = torch.randn(2, 2, 1024)
    features = torch.randn(2, 5, 1024)
    
    outputs = custom_head(xyz, features)
    print(f"  Bbox shape: {outputs['bbox'].shape}")
    print(f"  Confidence shape: {outputs['confidence'].shape}")
    print(f"  Classification shape: {outputs['classification'].shape}")


if __name__ == "__main__":
    example_usage()
    create_custom_head_example()
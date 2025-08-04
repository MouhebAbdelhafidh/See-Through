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

# Add PointNet2_PyTorch to path
POINTNET2_PATH = "Pointnet2_PyTorch"
if not os.path.exists(POINTNET2_PATH):
    print(f"Cloning PointNet2_PyTorch repository to {POINTNET2_PATH}...")
    os.system(f"git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git {POINTNET2_PATH}")

sys.path.append(POINTNET2_PATH)

try:
    from pointnet2_ops import pointnet2_utils
    from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule
except ImportError as e:
    print(f"Error importing PointNet2 modules: {e}")
    print("Please ensure PointNet2_PyTorch is properly installed")
    sys.exit(1)


class PointNet2FeatureExtractor(nn.Module):
    """
    PointNet2-based feature extractor using the official PointNet2_PyTorch implementation
    """
    def __init__(self, input_channels=3, feature_dim=128):
        super().__init__()
        
        # Set Abstraction layers
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[input_channels, 64, 64, 128], [input_channels, 64, 96, 128]],
                bn=True
            )
        )
        
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[128 + 3, 128, 128, 256], [128 + 3, 128, 196, 256]],
                bn=True
            )
        )
        
        self.SA_modules.append(
            PointnetSAModule(
                npoint=None,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[256 + 3, 256, 512, 1024], [256 + 3, 256, 512, 1024]],
                bn=True
            )
        )
        
        # Feature Propagation layers
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + 1024, 256, 256])
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[128 + 256, 256, feature_dim])
        )
        
        self.feature_dim = feature_dim

    def forward(self, xyz, features=None):
        """
        Forward pass through PointNet2
        Args:
            xyz: Point coordinates [B, N, 3]
            features: Point features [B, N, C] (optional)
        Returns:
            features: Extracted features [B, N, feature_dim]
        """
        if features is None:
            features = xyz
        
        xyz_list = []
        features_list = []
        
        # Set Abstraction layers
        for i, sa_module in enumerate(self.SA_modules):
            xyz, features = sa_module(xyz, features)
            xyz_list.append(xyz)
            features_list.append(features)
        
        # Feature Propagation layers
        for i, fp_module in enumerate(self.FP_modules):
            features = fp_module(
                xyz_list[-(i + 2)], xyz_list[-1], 
                features_list[-(i + 2)], features_list[-1]
            )
        
        return features


class PointNet2Backbone(nn.Module):
    """
    Complete PointNet2 backbone for feature extraction
    """
    def __init__(self, spatial_dims=3, feature_dims=3, feature_dim=128):
        super().__init__()
        
        # Combine spatial and feature dimensions
        input_channels = spatial_dims + feature_dims
        
        self.feature_extractor = PointNet2FeatureExtractor(
            input_channels=input_channels, 
            feature_dim=feature_dim
        )
        
        self.spatial_dims = spatial_dims
        self.feature_dims = feature_dims
        self.feature_dim = feature_dim

    def forward(self, xyz, features):
        """
        Forward pass
        Args:
            xyz: Point coordinates [B, 3, N]
            features: Point features [B, feature_dims, N]
        Returns:
            features: Extracted features [B, feature_dim, N]
        """
        # Transpose to [B, N, C] format for PointNet2
        xyz = xyz.transpose(2, 1)  # [B, N, 3]
        features = features.transpose(2, 1)  # [B, N, feature_dims]
        
        # Combine spatial and feature information
        combined_features = torch.cat([xyz, features], dim=-1)
        
        # Extract features
        extracted_features = self.feature_extractor(xyz, combined_features)
        
        # Transpose back to [B, C, N] format
        return extracted_features.transpose(2, 1)


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


def extract_features_from_file(model, input_file, output_dir, device='cuda'):
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
    parser = argparse.ArgumentParser(description='Extract features using PointNet2')
    parser.add_argument('--input_dir', type=str, default='FusedData', 
                       help='Input directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, default='ExtractedFeatures', 
                       help='Output directory for extracted features')
    parser.add_argument('--feature_dim', type=int, default=128, 
                       help='Dimension of extracted features')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda/cpu)')
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
    model_save_path = output_dir / "pointnet2_feature_extractor.pth"
    torch.save(model.state_dict(), model_save_path)
    
    print(f"\nSuccessfully processed {processed}/{len(h5_files)} files")
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
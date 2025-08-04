import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
import os
from sklearn.preprocessing import StandardScaler
import pickle


class RadarPointCloudDataset(Dataset):
    """
    Dataset class for radar point cloud data from H5 files
    """
    def __init__(self, h5_file_path: str, max_points: int = 1024, 
                 features: List[str] = ['rcs', 'vr', 'vr_compensated', 'num_merged', 'sensor_id'],
                 normalize_features: bool = True, normalize_xyz: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialize the dataset
        
        Args:
            h5_file_path: Path to the H5 file
            max_points: Maximum number of points per frame
            features: List of feature names to extract
            normalize_features: Whether to normalize features
            normalize_xyz: Whether to normalize xyz coordinates
            cache_dir: Directory to cache preprocessed data
        """
        self.h5_file_path = h5_file_path
        self.max_points = max_points
        self.features = features
        self.normalize_features = normalize_features
        self.normalize_xyz = normalize_xyz
        self.cache_dir = cache_dir
        
        # Load data
        self._load_data()
        
        # Initialize normalizers
        self._initialize_normalizers()

    def _load_data(self):
        """Load data from H5 file"""
        with h5py.File(self.h5_file_path, 'r') as f:
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
        
        print(f"Loaded {len(self.frame_indices)} frames from {self.h5_file_path}")

    def _initialize_normalizers(self):
        """Initialize feature and coordinate normalizers"""
        if self.normalize_features or self.normalize_xyz:
            # Compute statistics from a subset of frames
            sample_size = min(100, len(self.frame_indices))
            sample_indices = np.random.choice(self.frame_indices, sample_size, replace=False)
            
            feature_samples = []
            xyz_samples = []
            
            for idx in sample_indices:
                frame = self.frames[idx]
                start_idx = frame['detection_start_idx']
                end_idx = frame['detection_end_idx']
                points = self.detections[start_idx:end_idx]
                
                # Extract features
                feature_list = []
                for feature in self.features:
                    if feature == 'num_merged':
                        feature_list.append(points[feature].astype(np.float32))
                    else:
                        feature_list.append(points[feature])
                
                features = np.stack(feature_list, axis=1)
                xyz = np.stack([points['x_cc'], points['y_cc']], axis=1)
                
                feature_samples.append(features)
                xyz_samples.append(xyz)
            
            # Compute statistics
            all_features = np.concatenate(feature_samples, axis=0)
            all_xyz = np.concatenate(xyz_samples, axis=0)
            
            if self.normalize_features:
                self.feature_scaler = StandardScaler()
                self.feature_scaler.fit(all_features)
            
            if self.normalize_xyz:
                self.xyz_scaler = StandardScaler()
                self.xyz_scaler.fit(all_xyz)

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
        
        # Normalize if requested
        if self.normalize_xyz:
            xyz = self.xyz_scaler.transform(xyz)
        
        if self.normalize_features:
            features = self.feature_scaler.transform(features)
        
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


class RadarDataLoader:
    """
    Data loader utility for radar point cloud data
    """
    def __init__(self, data_dir: str, batch_size: int = 32, max_points: int = 1024,
                 features: List[str] = ['rcs', 'vr', 'vr_compensated', 'num_merged', 'sensor_id'],
                 normalize_features: bool = True, normalize_xyz: bool = True,
                 num_workers: int = 4):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing H5 files
            batch_size: Batch size for training
            max_points: Maximum number of points per frame
            features: List of feature names to extract
            normalize_features: Whether to normalize features
            normalize_xyz: Whether to normalize xyz coordinates
            num_workers: Number of worker processes
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_points = max_points
        self.features = features
        self.normalize_features = normalize_features
        self.normalize_xyz = normalize_xyz
        self.num_workers = num_workers
        
        # Find H5 files
        self.h5_files = self._find_h5_files()
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _find_h5_files(self) -> List[str]:
        """Find all H5 files in the data directory"""
        h5_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
        return sorted(h5_files)

    def create_datasets(self, train_split: float = 0.7, val_split: float = 0.15, 
                       test_split: float = 0.15, random_seed: int = 42):
        """
        Create train/val/test datasets
        
        Args:
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            random_seed: Random seed for reproducibility
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        np.random.seed(random_seed)
        
        # Shuffle files
        file_indices = np.random.permutation(len(self.h5_files))
        
        # Split files
        train_end = int(len(self.h5_files) * train_split)
        val_end = train_end + int(len(self.h5_files) * val_split)
        
        train_files = [self.h5_files[i] for i in file_indices[:train_end]]
        val_files = [self.h5_files[i] for i in file_indices[train_end:val_end]]
        test_files = [self.h5_files[i] for i in file_indices[val_end:]]
        
        # Create datasets
        if train_files:
            self.train_dataset = RadarPointCloudDataset(
                train_files[0],  # For now, use first file as training
                max_points=self.max_points,
                features=self.features,
                normalize_features=self.normalize_features,
                normalize_xyz=self.normalize_xyz
            )
        
        if val_files:
            self.val_dataset = RadarPointCloudDataset(
                val_files[0],  # For now, use first file as validation
                max_points=self.max_points,
                features=self.features,
                normalize_features=self.normalize_features,
                normalize_xyz=self.normalize_xyz
            )
        
        if test_files:
            self.test_dataset = RadarPointCloudDataset(
                test_files[0],  # For now, use first file as testing
                max_points=self.max_points,
                features=self.features,
                normalize_features=self.normalize_features,
                normalize_xyz=self.normalize_xyz
            )
        
        print(f"Created datasets:")
        print(f"  Train: {len(self.train_dataset) if self.train_dataset else 0} samples")
        print(f"  Val: {len(self.val_dataset) if self.val_dataset else 0} samples")
        print(f"  Test: {len(self.test_dataset) if self.test_dataset else 0} samples")

    def get_data_loaders(self) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """
        Get data loaders for train/val/test sets
        
        Returns:
            train_loader, val_loader, test_loader
        """
        train_loader = None
        val_loader = None
        test_loader = None
        
        if self.train_dataset:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        if self.test_dataset:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        return train_loader, val_loader, test_loader


def visualize_point_cloud(xyz: torch.Tensor, features: torch.Tensor, labels: torch.Tensor, 
                         title: str = "Point Cloud Visualization"):
    """
    Visualize a point cloud (for debugging purposes)
    
    Args:
        xyz: Point coordinates [2, N]
        features: Point features [feature_dims, N]
        labels: Point labels [N]
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        xyz_np = xyz.numpy()
        labels_np = labels.numpy()
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(xyz_np[0], xyz_np[1], c=labels_np, cmap='tab10', s=1)
        plt.colorbar(scatter)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(title)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


def get_class_distribution(dataset: RadarPointCloudDataset) -> Dict[int, int]:
    """
    Get class distribution from dataset
    
    Args:
        dataset: RadarPointCloudDataset instance
    
    Returns:
        Dictionary mapping class_id to count
    """
    class_counts = {}
    
    for i in range(len(dataset)):
        _, _, labels = dataset[i]
        unique_labels, counts = torch.unique(labels[labels != -1], return_counts=True)
        
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            if label in class_counts:
                class_counts[label] += count
            else:
                class_counts[label] = count
    
    return class_counts


def main():
    """
    Example usage of the radar data loader
    """
    # Example usage
    data_dir = "path/to/your/data"  # Replace with your data directory
    
    # Create data loader
    data_loader = RadarDataLoader(
        data_dir=data_dir,
        batch_size=8,
        max_points=1024,
        features=['rcs', 'vr', 'vr_compensated', 'num_merged', 'sensor_id'],
        normalize_features=True,
        normalize_xyz=True
    )
    
    # Create datasets
    data_loader.create_datasets(train_split=0.7, val_split=0.15, test_split=0.15)
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Example iteration
    if train_loader:
        for batch_idx, (xyz, features, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  xyz shape: {xyz.shape}")
            print(f"  features shape: {features.shape}")
            print(f"  labels shape: {labels.shape}")
            
            # Visualize first batch
            if batch_idx == 0:
                visualize_point_cloud(xyz[0], features[0], labels[0], "Training Sample")
            
            break
    
    # Get class distribution
    if data_loader.train_dataset:
        class_dist = get_class_distribution(data_loader.train_dataset)
        print(f"Class distribution: {class_dist}")


if __name__ == "__main__":
    main()
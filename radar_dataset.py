import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
from pathlib import Path


class RadarPointCloudDataset(Dataset):
    """
    PyTorch Dataset for loading radar point cloud data from .h5 files.
    
    Expected data format in .h5 files:
    - x_cc: x coordinate
    - y_cc: y coordinate  
    - vr: radial velocity
    - vr_compensated: compensated radial velocity
    - rcs: radar cross section
    - label_id: label for each point
    - track_id: tracking ID
    - num_merged: number of merged detections
    - sensor_id: sensor identifier
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        subset: str = 'train',
        max_points: int = 2048,
        normalize_coords: bool = True,
        augment: bool = False,
        cache_data: bool = False
    ):
        """
        Initialize the radar dataset.
        
        Args:
            data_path: Path to the .h5 file or directory containing .h5 files
            subset: Dataset subset ('train', 'val', 'test')
            max_points: Maximum number of points per sample
            normalize_coords: Whether to normalize coordinates
            augment: Whether to apply data augmentation
            cache_data: Whether to cache data in memory
        """
        self.data_path = Path(data_path)
        self.subset = subset
        self.max_points = max_points
        self.normalize_coords = normalize_coords
        self.augment = augment
        self.cache_data = cache_data
        
        # Field names in the .h5 file
        self.coord_fields = ['x_cc', 'y_cc']
        self.feature_fields = ['rcs', 'vr', 'vr_compensated', 'num_merged', 'sensor_id']
        self.label_field = 'label_id'
        self.extra_fields = ['track_id']
        
        # Initialize data storage
        self.data_files = []
        self.cached_data = {} if cache_data else None
        
        # Load file paths
        self._load_file_paths()
        
        # Calculate normalization statistics if needed
        if self.normalize_coords:
            self._calculate_normalization_stats()
    
    def _load_file_paths(self):
        """Load all .h5 file paths."""
        if self.data_path.is_file() and self.data_path.suffix == '.h5':
            self.data_files = [self.data_path]
        elif self.data_path.is_dir():
            self.data_files = list(self.data_path.glob('*.h5'))
            if self.subset in ['train', 'val', 'test']:
                # Filter files based on subset (assuming naming convention)
                subset_files = []
                for file in self.data_files:
                    if self.subset in file.stem.lower():
                        subset_files.append(file)
                if subset_files:
                    self.data_files = subset_files
        else:
            raise ValueError(f"Data path {self.data_path} is not a valid file or directory")
        
        if not self.data_files:
            raise ValueError(f"No .h5 files found in {self.data_path}")
        
        print(f"Found {len(self.data_files)} .h5 files for subset '{self.subset}'")
    
    def _calculate_normalization_stats(self):
        """Calculate mean and std for coordinate normalization."""
        print("Calculating normalization statistics...")
        all_coords = []
        
        for file_path in self.data_files[:min(5, len(self.data_files))]:  # Sample a few files
            with h5py.File(file_path, 'r') as f:
                coords = np.stack([f[field][:] for field in self.coord_fields], axis=-1)
                all_coords.append(coords)
        
        all_coords = np.concatenate(all_coords, axis=0)
        self.coord_mean = np.mean(all_coords, axis=0)
        self.coord_std = np.std(all_coords, axis=0)
        
        print(f"Coordinate mean: {self.coord_mean}")
        print(f"Coordinate std: {self.coord_std}")
    
    def _load_sample(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load a single sample from an .h5 file."""
        if self.cache_data and str(file_path) in self.cached_data:
            return self.cached_data[str(file_path)]
        
        with h5py.File(file_path, 'r') as f:
            data = {}
            
            # Load coordinates
            for field in self.coord_fields:
                data[field] = f[field][:]
            
            # Load features
            for field in self.feature_fields:
                data[field] = f[field][:]
            
            # Load labels
            if self.label_field in f:
                data[self.label_field] = f[self.label_field][:]
            
            # Load extra fields
            for field in self.extra_fields:
                if field in f:
                    data[field] = f[field][:]
        
        if self.cache_data:
            self.cached_data[str(file_path)] = data
        
        return data
    
    def _sample_points(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sample or pad points to match max_points."""
        num_points = len(data[self.coord_fields[0]])
        
        if num_points > self.max_points:
            # Random sampling
            indices = np.random.choice(num_points, self.max_points, replace=False)
        elif num_points < self.max_points:
            # Padding by repeating points
            indices = np.random.choice(num_points, self.max_points, replace=True)
        else:
            indices = np.arange(num_points)
        
        # Apply sampling to all fields
        sampled_data = {}
        for key, value in data.items():
            sampled_data[key] = value[indices]
        
        return sampled_data
    
    def _normalize_coordinates(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize coordinates using pre-calculated statistics."""
        if not self.normalize_coords:
            return data
        
        for i, field in enumerate(self.coord_fields):
            data[field] = (data[field] - self.coord_mean[i]) / (self.coord_std[i] + 1e-8)
        
        return data
    
    def _augment_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply data augmentation."""
        if not self.augment:
            return data
        
        # Random rotation around z-axis (for 2D radar data)
        angle = np.random.uniform(0, 2 * np.pi)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        
        x_cc = data['x_cc'] * cos_angle - data['y_cc'] * sin_angle
        y_cc = data['x_cc'] * sin_angle + data['y_cc'] * cos_angle
        
        data['x_cc'] = x_cc
        data['y_cc'] = y_cc
        
        # Random jittering of features
        noise_factor = 0.01
        for field in self.feature_fields:
            if field not in ['num_merged', 'sensor_id']:  # Don't jitter integer fields
                noise = np.random.normal(0, noise_factor, data[field].shape)
                data[field] = data[field] + noise
        
        # Random scaling of coordinates
        scale_factor = np.random.uniform(0.9, 1.1)
        data['x_cc'] *= scale_factor
        data['y_cc'] *= scale_factor
        
        return data
    
    def _to_tensors(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        tensor_data = {}
        
        for key, value in data.items():
            if key in ['num_merged', 'sensor_id', 'label_id', 'track_id']:
                tensor_data[key] = torch.from_numpy(value.astype(np.int64))
            else:
                tensor_data[key] = torch.from_numpy(value.astype(np.float32))
        
        return tensor_data
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing the processed sample data
        """
        file_path = self.data_files[idx]
        
        # Load raw data
        data = self._load_sample(file_path)
        
        # Sample/pad points
        data = self._sample_points(data)
        
        # Normalize coordinates
        data = self._normalize_coordinates(data)
        
        # Apply augmentation
        data = self._augment_data(data)
        
        # Convert to tensors
        data = self._to_tensors(data)
        
        return data


def radar_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for radar data.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched data dictionary
    """
    if not batch:
        return {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    # Stack tensors for each key
    batched_data = {}
    for key in keys:
        tensors = [sample[key] for sample in batch]
        batched_data[key] = torch.stack(tensors, dim=0)
    
    return batched_data


def create_radar_dataloader(
    data_path: Union[str, Path],
    subset: str = 'train',
    batch_size: int = 32,
    max_points: int = 2048,
    normalize_coords: bool = True,
    augment: bool = False,
    cache_data: bool = False,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for radar point cloud data.
    
    Args:
        data_path: Path to data files
        subset: Dataset subset
        batch_size: Batch size
        max_points: Maximum points per sample
        normalize_coords: Whether to normalize coordinates
        augment: Whether to apply augmentation
        cache_data: Whether to cache data
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    dataset = RadarPointCloudDataset(
        data_path=data_path,
        subset=subset,
        max_points=max_points,
        normalize_coords=normalize_coords,
        augment=augment,
        cache_data=cache_data
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=radar_collate_fn,
        pin_memory=True
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    # Create a synthetic .h5 file for testing
    def create_test_h5_file(filename: str, num_points: int = 1000):
        """Create a test .h5 file with synthetic radar data."""
        with h5py.File(filename, 'w') as f:
            # Coordinates
            f.create_dataset('x_cc', data=np.random.uniform(-50, 50, num_points))
            f.create_dataset('y_cc', data=np.random.uniform(-50, 50, num_points))
            
            # Features
            f.create_dataset('rcs', data=np.random.uniform(0, 2, num_points))
            f.create_dataset('vr', data=np.random.uniform(-10, 10, num_points))
            f.create_dataset('vr_compensated', data=np.random.uniform(-10, 10, num_points))
            f.create_dataset('num_merged', data=np.random.randint(1, 100, num_points))
            f.create_dataset('sensor_id', data=np.random.randint(0, 256, num_points))
            
            # Labels
            f.create_dataset('label_id', data=np.random.randint(0, 10, num_points))
            
            # Extra
            f.create_dataset('track_id', data=np.random.randint(-1, 1000, num_points))
    
    # Create test files
    test_dir = Path('test_radar_data')
    test_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        create_test_h5_file(test_dir / f'train_sample_{i}.h5')
    
    # Test the dataset
    try:
        dataset = RadarPointCloudDataset(
            data_path=test_dir,
            subset='train',
            max_points=512,
            normalize_coords=True,
            augment=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test a sample
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        for key, value in sample.items():
            print(f"{key}: {value.shape} ({value.dtype})")
        
        # Test dataloader
        dataloader = create_radar_dataloader(
            data_path=test_dir,
            subset='train',
            batch_size=2,
            max_points=512,
            num_workers=0  # Use 0 for testing
        )
        
        for batch in dataloader:
            print("\nBatch keys:", batch.keys())
            for key, value in batch.items():
                print(f"{key}: {value.shape}")
            break
            
    except Exception as e:
        print(f"Error testing dataset: {e}")
    
    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
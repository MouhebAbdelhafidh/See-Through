# PointNet2 Feature Extraction

This implementation uses the official [PointNet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) repository to extract features from point cloud data.

## Overview

PointNet2 is a hierarchical neural network that processes point clouds by grouping points and abstracting local features. This implementation provides:

- **PointNet2Backbone**: A complete PointNet2 architecture for feature extraction
- **Automatic preprocessing**: Point cloud centering and normalization
- **Batch processing**: Process multiple HDF5 files efficiently
- **Flexible input**: Support for various point cloud formats

## Features

- Uses the official PointNet2_PyTorch implementation
- Supports both spatial coordinates (x, y, z) and additional features
- Automatic point cloud preprocessing (centering, normalization)
- GPU acceleration support
- Batch processing of multiple files
- HDF5 input/output format

## Installation

### 1. Setup Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the setup script to install PointNet2_PyTorch
python setup_pointnet2.py
```

### 2. Verify Installation

```bash
# Run tests to verify everything works
python test_pointnet2.py
```

## Usage

### Basic Feature Extraction

```bash
# Extract features from HDF5 files
python pointnet2_feature_extractor.py \
    --input_dir FusedData \
    --output_dir ExtractedFeatures \
    --feature_dim 128 \
    --device cuda
```

### Command Line Arguments

- `--input_dir`: Directory containing HDF5 files (default: 'FusedData')
- `--output_dir`: Output directory for extracted features (default: 'ExtractedFeatures')
- `--feature_dim`: Dimension of extracted features (default: 128)
- `--device`: Device to use - 'cuda' or 'cpu' (default: 'cuda')
- `--model_path`: Path to pre-trained model weights (optional)

### Input Data Format

The script expects HDF5 files with point cloud data. Supported formats:

1. **Fused detections format** (recommended):
   ```python
   # HDF5 file with dataset 'fused_detections'
   # Shape: [N, 9] where N is number of points
   # Columns: [x, y, z, vr_compensated, rcs, v, r, num_merged, ...]
   ```

2. **Generic point format**:
   ```python
   # HDF5 file with dataset 'points'
   # Shape: [N, D] where D is number of features
   ```

3. **Any dataset format**:
   ```python
   # HDF5 file with any dataset
   # The first dataset found will be used
   ```

### Output Format

Each processed file creates an HDF5 file with:

- `features`: Extracted features [N, feature_dim]
- `original_points`: Original point cloud data
- `center`: Center coordinates used for preprocessing
- `processed_points`: Preprocessed point cloud data
- Attributes: `feature_dim`, `num_points`

## Architecture

### PointNet2Backbone

The main model class that combines:
- **Spatial dimensions**: x, y, z coordinates
- **Feature dimensions**: Additional point features (e.g., velocity, RCS)
- **Set Abstraction layers**: Hierarchical feature extraction
- **Feature Propagation layers**: Upsampling and feature refinement

### Preprocessing

1. **Centering**: Center point cloud around origin
2. **Normalization**: Scale to unit sphere (optional)
3. **Feature combination**: Combine spatial and feature dimensions

## Example Usage

### Python API

```python
import torch
from pointnet2_feature_extractor import PointNet2Backbone

# Initialize model
model = PointNet2Backbone(spatial_dims=3, feature_dims=3, feature_dim=128)
model.eval()

# Prepare data
xyz = torch.randn(1, 3, 1000)  # [batch, 3, num_points]
features = torch.randn(1, 3, 1000)  # [batch, features, num_points]

# Extract features
with torch.no_grad():
    extracted_features = model(xyz, features)
    # Shape: [1, 128, 1000]
```

### Processing Custom Data

```python
import numpy as np
import h5py
from pointnet2_feature_extractor import extract_features_from_file, PointNet2Backbone

# Create model
model = PointNet2Backbone(spatial_dims=3, feature_dims=3, feature_dim=64)

# Process single file
output_path = extract_features_from_file(
    model, "input_data.h5", "output_dir", device='cuda'
)
```

## Performance

- **GPU acceleration**: Significantly faster with CUDA
- **Batch processing**: Process multiple files efficiently
- **Memory efficient**: Processes files one at a time
- **Progress tracking**: Shows progress with tqdm

## Troubleshooting

### Common Issues

1. **PointNet2_PyTorch not found**:
   ```bash
   python setup_pointnet2.py
   ```

2. **CUDA out of memory**:
   - Use `--device cpu` for CPU processing
   - Reduce batch size or feature dimension

3. **Import errors**:
   ```bash
   pip install -e PointNet2_PyTorch
   ```

### Testing

Run the test suite to verify installation:

```bash
python test_pointnet2.py
```

## Advanced Usage

### Custom Model Architecture

```python
from pointnet2_feature_extractor import PointNet2FeatureExtractor

# Custom feature extractor
extractor = PointNet2FeatureExtractor(
    input_channels=6,  # 3 spatial + 3 features
    feature_dim=256
)
```

### Preprocessing Options

```python
from pointnet2_feature_extractor import preprocess_point_cloud

# Custom preprocessing
processed_points, center = preprocess_point_cloud(
    points, normalize=True, center=True
)
```

## File Structure

```
.
├── pointnet2_feature_extractor.py  # Main feature extraction script
├── setup_pointnet2.py              # Setup script
├── test_pointnet2.py               # Test script
├── requirements.txt                 # Python dependencies
├── README_PointNet2.md            # This file
├── PointNet2_PyTorch/             # PointNet2_PyTorch repository
├── FusedData/                     # Input data directory
└── ExtractedFeatures/             # Output directory
```

## Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- h5py >= 3.1.0
- scipy >= 1.7.0
- tqdm >= 4.62.0
- PointNet2_PyTorch (automatically installed)

## License

This implementation uses the PointNet2_PyTorch repository which is licensed under the MIT License.
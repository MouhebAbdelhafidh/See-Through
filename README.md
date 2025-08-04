# PointNet2 Feature Extraction

This repository provides a complete implementation for extracting features from point clouds using PointNet2 architecture. The implementation includes both a CPU-only version and integration with the official PointNet2_PyTorch repository.

## Overview

PointNet2 is a hierarchical neural network that processes point clouds by grouping points and abstracting local features. This implementation provides:

- **CPU-only implementation**: Works without CUDA compilation
- **Official PointNet2_PyTorch integration**: Uses the official repository when available
- **Automatic preprocessing**: Point cloud centering and normalization
- **Batch processing**: Process multiple HDF5 files efficiently
- **Flexible input**: Support for various point cloud formats

## Features

- ✅ CPU-only PointNet2 implementation (no CUDA required)
- ✅ Official PointNet2_PyTorch integration (when CUDA is available)
- ✅ Automatic point cloud preprocessing (centering, normalization)
- ✅ GPU acceleration support (when available)
- ✅ Batch processing of multiple files
- ✅ HDF5 input/output format
- ✅ Comprehensive testing suite
- ✅ Example usage scripts

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv pointnet2_env
source pointnet2_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run setup (optional - for CUDA version)
python setup_pointnet2.py
```

### 2. Test Installation

```bash
# Test CPU-only version
python test_pointnet2_cpu.py

# Test with examples
python example_usage.py
```

### 3. Extract Features

```bash
# Basic usage
python pointnet2_feature_extractor_cpu.py \
    --input_dir FusedData \
    --output_dir ExtractedFeatures \
    --feature_dim 128 \
    --device cpu
```

## File Structure

```
.
├── pointnet2_feature_extractor_cpu.py    # CPU-only implementation
├── pointnet2_feature_extractor.py        # CUDA version (requires PointNet2_PyTorch)
├── test_pointnet2_cpu.py                 # CPU version tests
├── test_pointnet2.py                     # CUDA version tests
├── example_usage.py                      # Usage examples
├── setup_pointnet2.py                    # Setup script
├── requirements.txt                       # Python dependencies
├── README.md                             # This file
├── README_PointNet2.md                   # Detailed documentation
├── pointnet2_env/                        # Virtual environment
├── Pointnet2_PyTorch/                    # Official repository (if cloned)
├── FusedData/                            # Input data directory
└── ExtractedFeatures/                    # Output directory
```

## Usage

### Basic Feature Extraction

```python
import torch
from pointnet2_feature_extractor_cpu import PointNet2Backbone

# Initialize model
model = PointNet2Backbone(spatial_dims=3, feature_dims=3, feature_dim=128)
model.eval()

# Prepare data
xyz = torch.randn(1, 3, 1000)  # [batch, 3, num_points]
features = torch.randn(1, 3, 1000)  # [batch, features, num_points]

# Extract features
with torch.no_grad():
    extracted_features = model(xyz, features)
    # Shape: [1, 128, 512] (downsampled to 512 points)
```

### Processing HDF5 Files

```bash
# Process single file
python pointnet2_feature_extractor_cpu.py \
    --input_dir FusedData \
    --output_dir ExtractedFeatures \
    --feature_dim 128 \
    --device cpu
```

### Command Line Arguments

- `--input_dir`: Directory containing HDF5 files (default: 'FusedData')
- `--output_dir`: Output directory for extracted features (default: 'ExtractedFeatures')
- `--feature_dim`: Dimension of extracted features (default: 128)
- `--device`: Device to use - 'cpu' or 'cuda' (default: 'cpu')
- `--model_path`: Path to pre-trained model weights (optional)

## Input Data Format

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

## Output Format

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

## Performance

- **CPU processing**: Works on any system without CUDA
- **GPU acceleration**: Significantly faster with CUDA (when available)
- **Batch processing**: Process multiple files efficiently
- **Memory efficient**: Processes files one at a time
- **Progress tracking**: Shows progress with tqdm

## Testing

Run the test suite to verify installation:

```bash
# Test CPU version
python test_pointnet2_cpu.py

# Test CUDA version (if available)
python test_pointnet2.py
```

## Examples

Run the example script to see various usage patterns:

```bash
python example_usage.py
```

This demonstrates:
- Basic model usage
- File processing
- Batch processing
- Custom model configurations

## Troubleshooting

### Common Issues

1. **CUDA not available**:
   - Use the CPU-only version: `pointnet2_feature_extractor_cpu.py`
   - Set `--device cpu`

2. **Memory issues**:
   - Reduce feature dimension: `--feature_dim 64`
   - Use CPU processing: `--device cpu`

3. **Import errors**:
   - Ensure virtual environment is activated
   - Install dependencies: `pip install -r requirements.txt`

### Testing

```bash
# Run all tests
python test_pointnet2_cpu.py

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- h5py >= 3.1.0
- scipy >= 1.7.0
- tqdm >= 4.62.0
- PointNet2_PyTorch (optional, for CUDA version)

## License

This implementation uses the PointNet2_PyTorch repository which is licensed under the MIT License.

## Citation

If you use this implementation, please cite the original PointNet2 paper:

```bibtex
@inproceedings{qi2017pointnet++,
  title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5099--5108},
  year={2017}
}
```

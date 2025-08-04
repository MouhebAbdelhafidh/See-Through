# PointNet2 Feature Extractor

A PyTorch implementation of PointNet2 (PointNet++) for 3D point cloud feature extraction, based on the paper "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space".

## Features

- **Complete PointNet2 Implementation**: Full implementation of the PointNet2 architecture with Set Abstraction modules
- **Multiple Variants**: Support for both Single-Scale Grouping (SSG) and Multi-Scale Grouping (MSG) versions
- **Hierarchical Feature Learning**: Extract features at multiple scales (local and global)
- **Flexible Input**: Support for point clouds with or without normal vectors
- **Batch Processing**: Efficient batch processing of multiple point clouds
- **Multiple File Formats**: Load point clouds from .npy, .txt, and .ply files
- **Feature Visualization**: Built-in visualization tools for understanding extracted features

## Architecture

The implementation includes:

1. **Farthest Point Sampling (FPS)**: For selecting representative points
2. **Ball Query**: For local neighborhood search
3. **Set Abstraction Modules**: For hierarchical feature learning
4. **Multi-Scale Grouping**: For capturing features at different scales
5. **Classification Head**: For point cloud classification tasks

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0 (for visualization)
- Open3D >= 0.13.0 (optional, for advanced point cloud processing)

### Quick Setup

```bash
# Clone or download the files
git clone <repository-url>
cd pointnet2-feature-extractor

# Install dependencies
pip install torch torchvision numpy matplotlib

# Test the installation
python pointnet2_feature_extractor.py --demo
```

## Usage

### Basic Feature Extraction

```python
import numpy as np
from pointnet2_feature_extractor import PointNet2FeatureExtractor, extract_features

# Load your point cloud (N x 3 or N x 6 array)
point_cloud = np.load('your_point_cloud.npy')

# Initialize the model
model = PointNet2FeatureExtractor(num_classes=40, use_msg=False, use_xyz=False)

# Extract features
features = extract_features(model, point_cloud, device='cuda')

# Access different feature levels
global_features = features['global_features']      # [1, 1024] - Global features
sa1_features = features['sa1_features']            # [1, 128, 512] - First level
sa2_features = features['sa2_features']            # [1, 256, 128] - Second level
classification = features['classification']        # [1, 40] - Classification scores
```

### Command Line Interface

```bash
# Basic usage with demo data
python pointnet2_feature_extractor.py --demo

# Process your own point cloud
python pointnet2_feature_extractor.py --input_file point_cloud.npy --output_file features.npy

# Use Multi-Scale Grouping
python pointnet2_feature_extractor.py --input_file point_cloud.npy --model_type msg

# Include normal vectors
python pointnet2_feature_extractor.py --input_file point_cloud.npy --use_normals

# Specify number of points and device
python pointnet2_feature_extractor.py --input_file point_cloud.npy --num_points 2048 --device cuda
```

### Advanced Usage Examples

See `example_usage.py` for comprehensive examples including:

1. **Basic Feature Extraction**: Simple feature extraction from sample data
2. **Multi-Scale Grouping**: Using the MSG variant for better feature capture
3. **Normal Vector Support**: Including surface normals in feature extraction
4. **Batch Processing**: Processing multiple point clouds simultaneously
5. **Feature Visualization**: Visualizing extracted features and hierarchical sampling
6. **Custom Point Clouds**: Loading and processing custom point cloud data

```bash
python example_usage.py
```

## Model Variants

### Single-Scale Grouping (SSG)
- Faster inference
- Single radius for local neighborhood
- Good for general-purpose feature extraction

```python
model = PointNet2FeatureExtractor(use_msg=False)
```

### Multi-Scale Grouping (MSG)
- Better feature representation
- Multiple radii for capturing different scales
- Higher computational cost

```python
model = PointNet2FeatureExtractor(use_msg=True)
```

## Input Formats

The feature extractor supports various input formats:

### Point Cloud Formats
- **NumPy arrays** (.npy): Direct loading with `np.load()`
- **Text files** (.txt): Space-separated coordinates
- **PLY files** (.ply): Simple PLY format support

### Point Cloud Structure
- **XYZ only**: `[N, 3]` - Just coordinates
- **XYZ + Normals**: `[N, 6]` - Coordinates + normal vectors
- **Custom features**: `[N, C]` - Coordinates + additional features

### Data Preprocessing
- **Normalization**: Points are normalized to unit sphere
- **Sampling**: Point clouds are sampled/padded to desired number of points
- **Centering**: Point clouds are centered at origin

## Output Features

The feature extractor returns a dictionary with the following keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `global_features` | `[B, 1024]` | Global point cloud features |
| `sa1_features` | `[B, 128, 512]` | First level Set Abstraction features |
| `sa2_features` | `[B, 256, 128]` | Second level Set Abstraction features |
| `classification` | `[B, num_classes]` | Classification logits |
| `sa1_xyz` | `[B, 3, 512]` | First level sampled coordinates |
| `sa2_xyz` | `[B, 3, 128]` | Second level sampled coordinates |
| `sa3_xyz` | `[B, 3, 1]` | Final level coordinates |

## Applications

This feature extractor can be used for various 3D point cloud tasks:

### Classification
- Object recognition in point clouds
- Shape classification
- Scene understanding

### Feature Analysis
- Point cloud similarity measurement
- Clustering and retrieval
- Dimensionality reduction

### Preprocessing
- Feature extraction for downstream tasks
- Point cloud representation learning
- Data augmentation

## Performance

### Computational Complexity
- **SSG Model**: ~1.2M parameters
- **MSG Model**: ~1.7M parameters
- **Inference Time**: ~50ms per point cloud (1024 points, GPU)
- **Memory Usage**: ~2GB GPU memory for batch size 32

### Scalability
- Supports point clouds from 512 to 4096 points
- Batch processing for multiple point clouds
- GPU acceleration for fast inference

## Customization

### Model Architecture
```python
# Custom number of classes
model = PointNet2FeatureExtractor(num_classes=10)

# Custom feature dimensions
class CustomPointNet2(PointNet2FeatureExtractor):
    def __init__(self, custom_features=512):
        super().__init__()
        # Modify the architecture as needed
        self.custom_layer = nn.Linear(1024, custom_features)
```

### Data Loading
```python
def custom_loader(file_path):
    # Implement your custom point cloud loader
    # Return numpy array of shape [N, C]
    pass

# Use with the feature extractor
point_cloud = custom_loader('custom_format.xyz')
features = extract_features(model, point_cloud)
```

## Visualization

The package includes visualization tools for understanding the feature extraction process:

```python
from example_usage import example_5_feature_visualization
features = example_5_feature_visualization()
```

This generates visualizations showing:
- Original point cloud
- Hierarchical sampling (SA1, SA2 points)
- Feature distributions
- Classification scores

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or number of points
   - Use CPU instead of GPU: `--device cpu`

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check PyTorch installation with CUDA support

3. **Point Cloud Loading Issues**
   - Verify file format and structure
   - Check point cloud dimensions (should be [N, 3] or [N, 6])

4. **Feature Dimension Mismatch**
   - Ensure point cloud has correct number of channels
   - Check `use_xyz` parameter setting

### Performance Tips

1. **Use GPU**: Significant speedup with CUDA-enabled PyTorch
2. **Batch Processing**: Process multiple point clouds together
3. **Point Sampling**: Use appropriate number of points (1024-2048 optimal)
4. **Model Selection**: Use SSG for speed, MSG for accuracy

## References

- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
- [Original PointNet2 Implementation](https://github.com/charlesq34/pointnet2)
- [Erik Wijmans' PyTorch Implementation](https://github.com/erikwijmans/Pointnet2_PyTorch)

## License

This implementation is provided under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd pointnet2-feature-extractor

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Format code
black *.py
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{qi2017pointnet++,
    title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
    author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
    booktitle={Advances in Neural Information Processing Systems},
    pages={5099--5108},
    year={2017}
}
```

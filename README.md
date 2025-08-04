# PointNet++ for Radar Point Cloud Processing

A PyTorch implementation of PointNet++ specifically designed for radar point cloud data processing. This implementation serves as a backbone for extracting hierarchical features from radar data that can be fused with VoteNet heads for classification or PointNet++ heads for segmentation.

## Features

- **Radar-Specific Design**: Tailored for 2D radar data with features like RCS, radial velocity, and sensor metadata
- **Hierarchical Feature Extraction**: Multi-level feature extraction suitable for both classification and segmentation
- **Flexible Architecture**: Can be used as a backbone for various downstream tasks
- **Data Augmentation**: Built-in augmentation techniques for radar data
- **Comprehensive Training**: Complete training pipeline with checkpointing and visualization

## Data Format

The implementation expects radar data in HDF5 (.h5) format with the following fields:

### Coordinates
- `x_cc`: X coordinate in Cartesian coordinates
- `y_cc`: Y coordinate in Cartesian coordinates

### Features
- `rcs`: Radar Cross Section
- `vr`: Radial velocity
- `vr_compensated`: Compensated radial velocity
- `num_merged`: Number of merged detections
- `sensor_id`: Sensor identifier

### Labels
- `label_id`: Point-wise labels for classification/segmentation

### Additional Fields
- `track_id`: Tracking ID (optional)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd pointnet-radar
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Ensure your radar data is in HDF5 format with the expected fields. You can create synthetic data for testing:

```bash
python train_pointnet.py --data_path ./synthetic_data --create_synthetic --mode classification
```

### 2. Training for Classification

Train PointNet++ with a classification head:

```bash
python train_pointnet.py \
    --data_path /path/to/your/radar/data \
    --mode classification \
    --num_classes 5 \
    --batch_size 16 \
    --epochs 50 \
    --lr 0.001
```

### 3. Training for Feature Extraction

Train PointNet++ as a feature extraction backbone:

```bash
python train_pointnet.py \
    --data_path /path/to/your/radar/data \
    --mode feature_extraction \
    --batch_size 16 \
    --epochs 50 \
    --lr 0.001
```

## Usage Examples

### Basic Feature Extraction

```python
import torch
from pointnet_backbone import RadarPointNetPlusPlus

# Create model
model = RadarPointNetPlusPlus(num_classes=None)  # Feature extraction only

# Example radar data
batch_data = {
    'x_cc': torch.randn(4, 1024),      # batch_size=4, num_points=1024
    'y_cc': torch.randn(4, 1024),
    'rcs': torch.randn(4, 1024),
    'vr': torch.randn(4, 1024),
    'vr_compensated': torch.randn(4, 1024),
    'num_merged': torch.randint(0, 100, (4, 1024)),
    'sensor_id': torch.randint(0, 256, (4, 1024)),
}

# Extract global features
global_features = model(batch_data)
print(f"Global features shape: {global_features.shape}")  # [4, 1024]

# Extract intermediate features for segmentation/fusion
global_features, intermediate = model(batch_data, return_intermediate=True)
print("Available intermediate features:")
for key, value in intermediate.items():
    if torch.is_tensor(value):
        print(f"  {key}: {value.shape}")
```

### Using with Custom Dataset

```python
from radar_dataset import RadarPointCloudDataset, create_radar_dataloader

# Create dataset
dataset = RadarPointCloudDataset(
    data_path='/path/to/h5/files',
    subset='train',
    max_points=2048,
    normalize_coords=True,
    augment=True
)

# Create data loader
dataloader = create_radar_dataloader(
    data_path='/path/to/h5/files',
    subset='train',
    batch_size=32,
    max_points=2048,
    normalize_coords=True,
    augment=True,
    num_workers=4
)

# Use in training loop
for batch in dataloader:
    features = model(batch)
    # ... training code
```

## Architecture

The PointNet++ architecture consists of:

1. **Set Abstraction Layers**: Hierarchical feature extraction using farthest point sampling and local region grouping
2. **Feature Propagation Layers**: Upsampling for dense prediction tasks
3. **Classification Head**: Optional head for direct classification

### Network Structure

```
Input: Radar Points [B, N, 8]  # 3 coords + 5 features
  ↓
SA1: [B, 512, 128]  # Sample 512 points, 128-dim features
  ↓
SA2: [B, 128, 256]  # Sample 128 points, 256-dim features
  ↓
SA3: [B, 1, 1024]   # Global features, 1024-dim
  ↓
Global Features: [B, 1024]
```

## Model Variants

### 1. Classification Model
```python
model = RadarPointNetPlusPlus(num_classes=10)
predictions = model.classifier(global_features)
```

### 2. Feature Extraction Backbone
```python
model = RadarPointNetPlusPlus(num_classes=None)
features = model(data)  # For fusion with other networks
```

### 3. Segmentation-Ready Model
```python
model = RadarPointNetPlusPlus(num_classes=None)
global_feat, intermediate = model(data, return_intermediate=True)
# Use intermediate features for point-wise predictions
```

## Integration with Other Architectures

### VoteNet Integration
```python
# Extract features for VoteNet
global_features, intermediate = model(radar_data, return_intermediate=True)

# Use l0_points for vote generation
vote_features = intermediate['l0_points']  # [B, 128, N]
# Feed to VoteNet voting module...
```

### Segmentation Head Integration
```python
# Extract multi-level features
global_features, intermediate = model(radar_data, return_intermediate=True)

# Use all levels for segmentation
seg_features = intermediate['l0_points']  # Point-wise features
# Feed to segmentation head...
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | Required | Path to radar data directory |
| `--mode` | `classification` | Training mode: `classification` or `feature_extraction` |
| `--num_classes` | `5` | Number of classes for classification |
| `--batch_size` | `16` | Batch size |
| `--max_points` | `2048` | Maximum points per sample |
| `--epochs` | `50` | Number of training epochs |
| `--lr` | `0.001` | Learning rate |
| `--weight_decay` | `1e-4` | Weight decay |
| `--device` | `cuda` | Device to use (cuda/cpu) |
| `--save_dir` | `checkpoints` | Directory to save checkpoints |
| `--resume` | `None` | Path to checkpoint to resume from |

## Data Augmentation

The implementation includes radar-specific data augmentation:

- **Rotation**: Random rotation around z-axis
- **Scaling**: Random coordinate scaling
- **Jittering**: Gaussian noise on continuous features
- **Normalization**: Coordinate normalization based on dataset statistics

## File Structure

```
├── pointnet_utils.py       # Core PointNet++ utilities (FPS, ball query, etc.)
├── pointnet_backbone.py    # Main PointNet++ model implementation
├── radar_dataset.py        # Dataset and data loading utilities
├── train_pointnet.py       # Training script with full pipeline
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Performance Tips

1. **Memory Usage**: Adjust `max_points` and `batch_size` based on GPU memory
2. **Data Loading**: Use `num_workers > 0` for faster data loading
3. **Feature Normalization**: Enable coordinate normalization for better convergence
4. **Augmentation**: Use augmentation for training, disable for validation

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{pointnet_radar_2024,
  title={PointNet++ for Radar Point Cloud Processing},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/pointnet-radar}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

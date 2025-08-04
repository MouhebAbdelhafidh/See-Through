# PointNet++ Implementation for Radar Point Cloud Processing

This repository contains a complete implementation of PointNet++ from scratch using PyTorch, specifically designed for processing radar point cloud data from mmWave radars.

## Overview

PointNet++ is a hierarchical neural network that applies PointNet recursively on a nested partitioning of the input point set. This implementation is tailored for radar point cloud data with the following features:

- **Spatial dimensions**: `x_cc`, `y_cc` (2D coordinates)
- **Feature dimensions**: `rcs`, `vr`, `vr_compensated`, `num_merged`, `sensor_id`
- **Label**: `label_id` (for classification/segmentation)

## Data Format

The implementation expects radar point cloud data in H5 format with the following structure:

```python
# Example record (first entry):
(11.524101, -2.2880847, -2.515373, 2.6073835, 0.8275838, 0, -1, 63, 255)

# All fields of the first record:
x_cc: 11.524101257324219
y_cc: -2.2880847454071045
vr: -2.5153729915618896
vr_compensated: 2.6073834896087646
rcs: 0.8275837898254395
label_id: 0
track_id: -1
num_merged: 63
sensor_id: 255
```

## Files Structure

```
├── pointnet_plus_plus.py              # Complete PointNet++ implementation
├── pointnet_plus_plus_backbone.py     # Backbone-only implementation for fusion
├── radar_data_loader.py               # Data loading utilities
├── train_pointnet_plus_plus.py        # Training script
├── requirements.txt                    # Dependencies
└── README_PointNet++.md              # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have PyTorch installed with CUDA support (if using GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Basic Model Creation

```python
from pointnet_plus_plus_backbone import create_pointnet_plus_plus_model

# Create model with backbone and heads
backbone, classification_head, segmentation_head = create_pointnet_plus_plus_model(
    num_classes=13,
    feature_dim=128,
    spatial_dims=2,  # x_cc, y_cc
    feature_dims=5,  # rcs, vr, vr_compensated, num_merged, sensor_id
    task='both'  # 'classification', 'segmentation', or 'both'
)
```

### 2. Data Loading

```python
from radar_data_loader import RadarDataLoader

# Create data loader
data_loader = RadarDataLoader(
    data_dir="path/to/your/h5/files",
    batch_size=16,
    max_points=1024,
    features=['rcs', 'vr', 'vr_compensated', 'num_merged', 'sensor_id'],
    normalize_features=True,
    normalize_xyz=True
)

# Create datasets
data_loader.create_datasets(train_split=0.7, val_split=0.15, test_split=0.15)

# Get data loaders
train_loader, val_loader, test_loader = data_loader.get_data_loaders()
```

### 3. Training

```bash
python train_pointnet_plus_plus.py \
    --data_dir /path/to/your/data \
    --batch_size 16 \
    --max_points 1024 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --feature_dim 128 \
    --num_classes 13 \
    --task both \
    --save_dir checkpoints
```

### 4. Using the Backbone for Fusion

The backbone can be easily integrated with other architectures like VotNet:

```python
from pointnet_plus_plus_backbone import PointNetPlusPlusBackbone

# Create backbone
backbone = PointNetPlusPlusBackbone(
    spatial_dims=2,
    feature_dims=5,
    feature_dim=128
)

# Forward pass
xyz = torch.randn(4, 2, 1024)  # [B, spatial_dims, N]
features = torch.randn(4, 5, 1024)  # [B, feature_dims, N]

point_features, global_features, skip_connections = backbone(xyz, features)

# Use these features for fusion with other architectures
print(f"Point features: {point_features.shape}")
print(f"Global features: {global_features.shape}")
```

## Model Architecture

### PointNet++ Components

1. **Farthest Point Sampling (FPS)**: Samples points that are farthest from each other
2. **Ball Query**: Groups points within a radius around sampled points
3. **Set Abstraction**: Processes grouped points through MLPs
4. **Feature Propagation**: Upsamples features from coarse to fine levels

### Architecture Details

- **Input**: Point coordinates (x_cc, y_cc) + features (rcs, vr, vr_compensated, num_merged, sensor_id)
- **Set Abstraction Layers**: 3 levels with decreasing number of points (1024 → 512 → 128 → 1)
- **Feature Propagation Layers**: 3 levels for upsampling
- **Output**: Classification and/or segmentation predictions

### Hyperparameters

- **Radius**: 0.2, 0.4 for different abstraction levels
- **Number of samples**: 32, 64 for ball query
- **Feature dimensions**: 64, 128, 256, 512, 1024 for MLPs
- **Dropout**: 0.5 for regularization

## Training Details

### Loss Functions

- **Classification Loss**: CrossEntropyLoss for point cloud classification
- **Segmentation Loss**: CrossEntropyLoss with ignore_index=-1 for point-wise segmentation

### Optimization

- **Optimizer**: Adam with learning rate 0.001
- **Scheduler**: StepLR with step_size=20, gamma=0.7
- **Weight Decay**: 1e-4 for regularization

### Data Augmentation

- **Point Sampling**: Random sampling when points exceed max_points
- **Normalization**: StandardScaler for features and coordinates
- **Padding**: Zero padding for frames with insufficient points

## Integration with Other Architectures

### VotNet Integration

```python
# Example: Using PointNet++ backbone with VotNet
from pointnet_plus_plus_backbone import PointNetPlusPlusBackbone

# Create PointNet++ backbone
pointnet_backbone = PointNetPlusPlusBackbone(
    spatial_dims=2,
    feature_dims=5,
    feature_dim=128
)

# Extract features
point_features, global_features, skip_connections = pointnet_backbone(xyz, features)

# Use these features in VotNet
# ... VotNet implementation using point_features and global_features
```

### Custom Head Integration

```python
# Create custom classification head
class CustomClassificationHead(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(backbone.get_feature_dim(), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, xyz, features):
        _, global_features, _ = self.backbone(xyz, features)
        return self.classifier(global_features)
```

## Performance Optimization

### Memory Efficiency

- **Gradient Checkpointing**: Enable for large models
- **Mixed Precision**: Use torch.cuda.amp for faster training
- **Batch Size**: Adjust based on GPU memory

### Speed Optimization

- **Data Loading**: Use multiple workers and pin_memory
- **Model**: Use torch.jit.script for inference
- **Hardware**: Use GPU with sufficient VRAM

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Segmentation**: Point-wise accuracy, IoU per class
- **Training**: Loss curves, learning rate scheduling

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or max_points
2. **Slow Training**: Check data loading, use GPU
3. **Poor Performance**: Adjust learning rate, check data normalization

### Debugging

```python
# Check data format
xyz, features, labels = next(iter(train_loader))
print(f"xyz shape: {xyz.shape}")
print(f"features shape: {features.shape}")
print(f"labels shape: {labels.shape}")

# Check model output
model = create_pointnet_plus_plus_model(...)
classification_output, segmentation_output = model(xyz, features)
print(f"classification output: {classification_output.shape}")
print(f"segmentation output: {segmentation_output.shape}")
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{qi2017pointnet++,
  title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## License

This implementation is provided for research purposes. Please ensure compliance with the original PointNet++ paper and your data usage agreements.

## Contributing

Feel free to submit issues and enhancement requests!
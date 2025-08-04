#!/usr/bin/env python3
"""
Setup script for PointNet2_PyTorch feature extraction
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Success: {result.stdout}")
    return True

def setup_pointnet2():
    """Setup PointNet2_PyTorch repository and dependencies"""
    
    # Clone PointNet2_PyTorch if not exists
    pointnet2_path = Path("PointNet2_PyTorch")
    if not pointnet2_path.exists():
        print("Cloning PointNet2_PyTorch repository...")
        if not run_command("git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git"):
            print("Failed to clone PointNet2_PyTorch repository")
            return False
    
    # Install dependencies
    print("Installing Python dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print("Failed to install Python dependencies")
        return False
    
    # Install PointNet2_PyTorch
    print("Installing PointNet2_PyTorch...")
    if not run_command("pip install -e PointNet2_PyTorch"):
        print("Failed to install PointNet2_PyTorch")
        return False
    
    # Test import
    print("Testing PointNet2 import...")
    try:
        import torch
        sys.path.append(str(pointnet2_path))
        from pointnet2_ops import pointnet2_utils
        from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule
        print("✓ PointNet2_PyTorch successfully imported!")
        return True
    except ImportError as e:
        print(f"✗ Failed to import PointNet2: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    import numpy as np
    import h5py
    
    # Create sample directory
    sample_dir = Path("SampleData")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample point cloud data
    num_points = 1000
    points = np.random.randn(num_points, 9)  # 9 features: x, y, z, vr, rcs, etc.
    points[:, :3] *= 10  # Scale spatial coordinates
    
    # Save sample data
    sample_file = sample_dir / "sample_data.h5"
    with h5py.File(sample_file, 'w') as f:
        f.create_dataset('fused_detections', data=points)
    
    print(f"Created sample data: {sample_file}")
    return sample_file

def main():
    print("Setting up PointNet2_PyTorch feature extraction...")
    
    # Setup PointNet2
    if not setup_pointnet2():
        print("Setup failed!")
        return False
    
    # Create sample data
    sample_file = create_sample_data()
    
    print("\nSetup complete!")
    print(f"Sample data created: {sample_file}")
    print("\nTo run feature extraction:")
    print("python pointnet2_feature_extractor.py --input_dir SampleData --output_dir ExtractedFeatures")
    
    return True

if __name__ == "__main__":
    main()
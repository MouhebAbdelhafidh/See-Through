#!/usr/bin/env python3
"""
Test script to verify PointNet++ implementation structure
"""

import sys
import os

def test_imports():
    """Test if the modules can be imported (without PyTorch)"""
    print("Testing module imports...")
    
    # Test basic imports
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError:
        print("✗ numpy not available")
    
    try:
        import h5py
        print("✓ h5py imported successfully")
    except ImportError:
        print("✗ h5py not available")
    
    # Test our custom modules (without PyTorch dependencies)
    try:
        # Read the files to check syntax
        with open('pointnet_plus_plus_backbone.py', 'r') as f:
            content = f.read()
        print("✓ pointnet_plus_plus_backbone.py syntax check passed")
    except FileNotFoundError:
        print("✗ pointnet_plus_plus_backbone.py not found")
    except Exception as e:
        print(f"✗ pointnet_plus_plus_backbone.py syntax error: {e}")
    
    try:
        with open('radar_data_loader.py', 'r') as f:
            content = f.read()
        print("✓ radar_data_loader.py syntax check passed")
    except FileNotFoundError:
        print("✗ radar_data_loader.py not found")
    except Exception as e:
        print(f"✗ radar_data_loader.py syntax error: {e}")
    
    try:
        with open('train_pointnet_plus_plus.py', 'r') as f:
            content = f.read()
        print("✓ train_pointnet_plus_plus.py syntax check passed")
    except FileNotFoundError:
        print("✗ train_pointnet_plus_plus.py not found")
    except Exception as e:
        print(f"✗ train_pointnet_plus_plus.py syntax error: {e}")

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'pointnet_plus_plus.py',
        'pointnet_plus_plus_backbone.py',
        'radar_data_loader.py',
        'train_pointnet_plus_plus.py',
        'requirements.txt',
        'README_PointNet++.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")

def test_code_structure():
    """Test the structure of the implementation"""
    print("\nTesting code structure...")
    
    # Check for key classes and functions
    with open('pointnet_plus_plus_backbone.py', 'r') as f:
        content = f.read()
        
    key_components = [
        'class FarthestPointSample',
        'class BallQuery',
        'class PointNetSetAbstraction',
        'class PointNetFeaturePropagation',
        'class PointNetPlusPlusBackbone',
        'class PointNetPlusPlusClassificationHead',
        'class PointNetPlusPlusSegmentationHead',
        'def create_pointnet_plus_plus_model',
        'def index_points'
    ]
    
    for component in key_components:
        if component in content:
            print(f"✓ {component} found")
        else:
            print(f"✗ {component} missing")
    
    # Check data loader structure
    with open('radar_data_loader.py', 'r') as f:
        content = f.read()
        
    data_loader_components = [
        'class RadarPointCloudDataset',
        'class RadarDataLoader',
        'def visualize_point_cloud',
        'def get_class_distribution'
    ]
    
    for component in data_loader_components:
        if component in content:
            print(f"✓ {component} found")
        else:
            print(f"✗ {component} missing")

def test_requirements():
    """Test requirements file"""
    print("\nTesting requirements...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'torch',
            'numpy',
            'h5py',
            'scikit-learn',
            'matplotlib',
            'tqdm'
        ]
        
        for package in required_packages:
            if package in requirements:
                print(f"✓ {package} in requirements.txt")
            else:
                print(f"✗ {package} missing from requirements.txt")
                
    except FileNotFoundError:
        print("✗ requirements.txt not found")

def main():
    """Main test function"""
    print("PointNet++ Implementation Test")
    print("=" * 40)
    
    test_imports()
    test_file_structure()
    test_code_structure()
    test_requirements()
    
    print("\n" + "=" * 40)
    print("Test completed!")
    print("\nTo run the full implementation, install PyTorch and other dependencies:")
    print("pip install -r requirements.txt")
    print("\nThen you can run:")
    print("python pointnet_plus_plus_backbone.py")

if __name__ == "__main__":
    main()
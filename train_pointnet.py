#!/usr/bin/env python3
"""
Training script for PointNet++ on radar point cloud data.

This script demonstrates how to train the PointNet++ backbone for:
1. Feature extraction (backbone only)
2. Classification (with classification head)
3. Feature extraction for fusion with VoteNet or segmentation heads

Usage:
    python train_pointnet.py --data_path /path/to/radar/data --mode classification
    python train_pointnet.py --data_path /path/to/radar/data --mode feature_extraction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import os
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Import our modules
from pointnet_backbone import RadarPointNetPlusPlus, get_radar_pointnet_model
from radar_dataset import create_radar_dataloader, RadarPointCloudDataset


class PointNetTrainer:
    """
    Trainer class for PointNet++ on radar data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,
            gamma=0.7
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        
        # Create log file
        self.log_file = self.save_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log(f"Trainer initialized. Device: {device}")
        self.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def log(self, message: str):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Extract features
            global_features = self.model(batch)
            
            # If model has classifier, compute classification loss
            if hasattr(self.model, 'classifier') and self.model.classifier is not None:
                # Use global label (majority vote or first label)
                labels = batch['label_id'][:, 0]  # Use first point's label as global label
                predictions = self.model.classifier(global_features)
                loss = self.criterion(predictions, labels)
            else:
                # For feature extraction mode, use a dummy loss (e.g., reconstruction)
                # Here we'll use a simple contrastive-like loss
                loss = self._compute_feature_loss(global_features, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                self.log(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def _compute_feature_loss(self, features: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute a simple feature learning loss for unsupervised pre-training.
        This is a placeholder - you might want to implement proper self-supervised losses.
        """
        # Simple reconstruction loss using features to predict point statistics
        batch_size = features.shape[0]
        
        # Predict some statistics about the input points
        target_stats = torch.stack([
            torch.mean(batch['rcs'], dim=1),
            torch.std(batch['rcs'], dim=1),
            torch.mean(batch['vr'], dim=1),
            torch.std(batch['vr'], dim=1)
        ], dim=1)  # [B, 4]
        
        # Simple MLP to predict stats from features
        predictor = nn.Linear(features.shape[1], 4).to(self.device)
        predicted_stats = predictor(features)
        
        loss = nn.MSELoss()(predicted_stats, target_stats)
        return loss
    
    def validate(self) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                # Forward pass
                global_features = self.model(batch)
                
                if hasattr(self.model, 'classifier') and self.model.classifier is not None:
                    # Classification
                    labels = batch['label_id'][:, 0]  # Global label
                    predictions = self.model.classifier(global_features)
                    loss = self.criterion(predictions, labels)
                    
                    # Accuracy
                    _, predicted = torch.max(predictions.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                else:
                    # Feature extraction loss
                    loss = self._compute_feature_loss(global_features, batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            self.log(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        start_epoch = checkpoint['epoch'] + 1
        self.log(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch
    
    def train(self, num_epochs: int, save_interval: int = 5):
        """Main training loop."""
        self.log(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            # Log epoch results
            epoch_time = time.time() - start_time
            self.log(f"Epoch {epoch+1}/{num_epochs}")
            self.log(f"  Train Loss: {train_loss:.4f}")
            self.log(f"  Val Loss: {val_loss:.4f}")
            self.log(f"  Val Acc: {val_acc:.4f}")
            self.log(f"  Best Val Acc: {self.best_val_acc:.4f}")
            self.log(f"  Time: {epoch_time:.2f}s")
            self.log("-" * 50)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        self.log("Training completed!")
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log("Training curves saved to training_curves.png")


def create_synthetic_data(data_dir: Path, num_files: int = 50, points_per_file: int = 2048):
    """Create synthetic radar data for testing."""
    import h5py
    
    data_dir.mkdir(exist_ok=True)
    
    # Create train/val split
    train_files = int(num_files * 0.8)
    val_files = num_files - train_files
    
    def create_file(filename: str, num_points: int):
        with h5py.File(filename, 'w') as f:
            # Coordinates (2D radar data)
            f.create_dataset('x_cc', data=np.random.uniform(-100, 100, num_points))
            f.create_dataset('y_cc', data=np.random.uniform(-100, 100, num_points))
            
            # Features
            f.create_dataset('rcs', data=np.random.exponential(1.0, num_points))
            f.create_dataset('vr', data=np.random.normal(0, 5, num_points))
            f.create_dataset('vr_compensated', data=np.random.normal(0, 5, num_points))
            f.create_dataset('num_merged', data=np.random.randint(1, 50, num_points))
            f.create_dataset('sensor_id', data=np.random.randint(0, 4, num_points))
            
            # Labels (synthetic classes)
            f.create_dataset('label_id', data=np.random.randint(0, 5, num_points))
            f.create_dataset('track_id', data=np.random.randint(-1, 1000, num_points))
    
    print(f"Creating {train_files} training files...")
    for i in range(train_files):
        create_file(data_dir / f'train_sample_{i:04d}.h5', points_per_file)
    
    print(f"Creating {val_files} validation files...")
    for i in range(val_files):
        create_file(data_dir / f'val_sample_{i:04d}.h5', points_per_file)
    
    print(f"Synthetic data created in {data_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train PointNet++ on radar data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to radar data directory')
    parser.add_argument('--mode', type=str, default='classification', 
                       choices=['classification', 'feature_extraction'],
                       help='Training mode')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for classification')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_points', type=int, default=2048, help='Maximum points per sample')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--create_synthetic', action='store_true', help='Create synthetic data for testing')
    
    args = parser.parse_args()
    
    # Create synthetic data if requested
    data_path = Path(args.data_path)
    if args.create_synthetic:
        print("Creating synthetic radar data...")
        create_synthetic_data(data_path)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    if args.mode == 'classification':
        model = RadarPointNetPlusPlus(num_classes=args.num_classes)
        print(f"Created classification model with {args.num_classes} classes")
    else:
        model = RadarPointNetPlusPlus(num_classes=None)  # Feature extraction only
        print("Created feature extraction model")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_radar_dataloader(
        data_path=data_path,
        subset='train',
        batch_size=args.batch_size,
        max_points=args.max_points,
        normalize_coords=True,
        augment=True,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = create_radar_dataloader(
        data_path=data_path,
        subset='val',
        batch_size=args.batch_size,
        max_points=args.max_points,
        normalize_coords=True,
        augment=False,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create trainer
    trainer = PointNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    # Example of using the trained model for feature extraction
    print("\nExample: Using trained model for feature extraction...")
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            
            # Extract features (for VoteNet/segmentation fusion)
            global_features, intermediate_features = model(batch, return_intermediate=True)
            
            print(f"Global features shape: {global_features.shape}")
            print("Intermediate features available for fusion:")
            for key, value in intermediate_features.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape}")
            break


if __name__ == "__main__":
    main()
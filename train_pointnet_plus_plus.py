import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from datetime import datetime

# Import our custom modules
from pointnet_plus_plus_backbone import create_pointnet_plus_plus_model
from radar_data_loader import RadarDataLoader, RadarPointCloudDataset


class PointNetPlusPlusTrainer:
    """
    Trainer class for PointNet++ model
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device='cuda', learning_rate=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.7)
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.segmentation_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_segmentation_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (xyz, features, labels) in enumerate(pbar):
            xyz = xyz.to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'backbone'):
                # Using backbone + heads
                point_features, global_features, _ = self.model.backbone(xyz, features)
                
                # Classification
                classification_output = self.model.classification_head(xyz, features)
                
                # Segmentation
                segmentation_output = self.model.segmentation_head(xyz, features)
            else:
                # Using complete model
                classification_output, segmentation_output = self.model(xyz, features)
            
            # Calculate losses
            # For classification, use the most common label in the point cloud
            classification_loss = torch.tensor(0.0).to(self.device)
            batch_correct = 0
            batch_total = 0
            
            for i in range(xyz.shape[0]):
                valid_labels = labels[i][labels[i] != -1]
                if len(valid_labels) > 0:
                    unique_labels, counts = torch.unique(valid_labels, return_counts=True)
                    most_common_label = unique_labels[torch.argmax(counts)]
                    
                    loss = self.classification_criterion(
                        classification_output[i:i+1], 
                        most_common_label.unsqueeze(0)
                    )
                    classification_loss += loss
                    
                    # Accuracy
                    pred = torch.argmax(classification_output[i])
                    if pred == most_common_label:
                        batch_correct += 1
                    batch_total += 1
            
            classification_loss = classification_loss / xyz.shape[0]
            
            # For segmentation, use all valid labels
            segmentation_loss = torch.tensor(0.0).to(self.device)
            valid_mask = labels != -1
            if valid_mask.sum() > 0:
                segmentation_loss = self.segmentation_criterion(
                    segmentation_output[:, :, valid_mask], 
                    labels[valid_mask]
                )
            
            # Total loss
            total_batch_loss = classification_loss + segmentation_loss
            
            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            total_classification_loss += classification_loss.item()
            total_segmentation_loss += segmentation_loss.item()
            correct_predictions += batch_correct
            total_predictions += batch_total
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Cls_Loss': f'{classification_loss.item():.4f}',
                'Seg_Loss': f'{segmentation_loss.item():.4f}',
                'Acc': f'{batch_correct/max(batch_total, 1):.3f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_classification_loss = total_classification_loss / len(self.train_loader)
        avg_segmentation_loss = total_segmentation_loss / len(self.train_loader)
        accuracy = correct_predictions / max(total_predictions, 1)
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy, avg_classification_loss, avg_segmentation_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_segmentation_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (xyz, features, labels) in enumerate(self.val_loader):
                xyz = xyz.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'backbone'):
                    classification_output = self.model.classification_head(xyz, features)
                    segmentation_output = self.model.segmentation_head(xyz, features)
                else:
                    classification_output, segmentation_output = self.model(xyz, features)
                
                # Calculate losses
                classification_loss = torch.tensor(0.0).to(self.device)
                batch_correct = 0
                batch_total = 0
                
                for i in range(xyz.shape[0]):
                    valid_labels = labels[i][labels[i] != -1]
                    if len(valid_labels) > 0:
                        unique_labels, counts = torch.unique(valid_labels, return_counts=True)
                        most_common_label = unique_labels[torch.argmax(counts)]
                        
                        loss = self.classification_criterion(
                            classification_output[i:i+1], 
                            most_common_label.unsqueeze(0)
                        )
                        classification_loss += loss
                        
                        # Accuracy
                        pred = torch.argmax(classification_output[i])
                        if pred == most_common_label:
                            batch_correct += 1
                        batch_total += 1
                        
                        # Store predictions for metrics
                        all_predictions.append(pred.item())
                        all_labels.append(most_common_label.item())
                
                classification_loss = classification_loss / xyz.shape[0]
                
                # Segmentation loss
                segmentation_loss = torch.tensor(0.0).to(self.device)
                valid_mask = labels != -1
                if valid_mask.sum() > 0:
                    segmentation_loss = self.segmentation_criterion(
                        segmentation_output[:, :, valid_mask], 
                        labels[valid_mask]
                    )
                
                # Total loss
                total_batch_loss = classification_loss + segmentation_loss
                
                # Update metrics
                total_loss += total_batch_loss.item()
                total_classification_loss += classification_loss.item()
                total_segmentation_loss += segmentation_loss.item()
                correct_predictions += batch_correct
                total_predictions += batch_total
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_classification_loss = total_classification_loss / len(self.val_loader)
        avg_segmentation_loss = total_segmentation_loss / len(self.val_loader)
        accuracy = correct_predictions / max(total_predictions, 1)
        
        # Calculate additional metrics
        if len(all_predictions) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
        else:
            precision = recall = f1 = 0.0
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, avg_classification_loss, avg_segmentation_loss, precision, recall, f1
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc, train_cls_loss, train_seg_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_cls_loss, val_seg_loss, precision, recall, f1 = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  Saved best model (val_loss: {val_loss:.4f})")
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                }, os.path.join(save_dir, 'best_accuracy_model.pth'))
                print(f"  Saved best accuracy model (val_acc: {val_acc:.4f})")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        print(f"\nTraining completed! Best val_loss: {best_val_loss:.4f}, Best val_acc: {best_val_accuracy:.4f}")
    
    def plot_training_curves(self, save_dir):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train PointNet++ on radar data')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing H5 files')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_points', type=int, default=1024, help='Maximum points per frame')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--num_classes', type=int, default=13, help='Number of classes')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--task', type=str, default='both', choices=['classification', 'segmentation', 'both'],
                       help='Task to perform')
    
    args = parser.parse_args()
    
    # Create data loader
    print("Creating data loader...")
    data_loader = RadarDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_points=args.max_points,
        features=['rcs', 'vr', 'vr_compensated', 'num_merged', 'sensor_id'],
        normalize_features=True,
        normalize_xyz=True
    )
    
    # Create datasets
    data_loader.create_datasets(train_split=0.7, val_split=0.15, test_split=0.15)
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    if train_loader is None:
        print("No training data found!")
        return
    
    # Create model
    print("Creating PointNet++ model...")
    backbone, classification_head, segmentation_head = create_pointnet_plus_plus_model(
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        spatial_dims=2,  # x_cc, y_cc
        feature_dims=5,  # rcs, vr, vr_compensated, num_merged, sensor_id
        task=args.task
    )
    
    # Create complete model
    if args.task == 'classification':
        model = classification_head
    elif args.task == 'segmentation':
        model = segmentation_head
    else:  # both
        # Create a wrapper that returns both outputs
        class CombinedModel(nn.Module):
            def __init__(self, backbone, classification_head, segmentation_head):
                super().__init__()
                self.backbone = backbone
                self.classification_head = classification_head
                self.segmentation_head = segmentation_head
            
            def forward(self, xyz, features):
                return self.classification_head(xyz, features), self.segmentation_head(xyz, features)
        
        model = CombinedModel(backbone, classification_head, segmentation_head)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = PointNetPlusPlusTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        learning_rate=args.learning_rate
    )
    
    # Train the model
    trainer.train(num_epochs=args.num_epochs, save_dir=args.save_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
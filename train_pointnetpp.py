# train_pointnetpp.py
import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from extract_features import PointNet2Backbone  

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = "FusedData/sequence_1_fused.h5"
FEATURE_FIELDS = ['x_cc','y_cc','vr','vr_compensated','rcs']
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
SAVE_PATH = "pointnetpp_backbone_new.pth"

# -------------------------
# Load dataset as one point cloud
# -------------------------
def load_h5_single(file_path, selected_fields=FEATURE_FIELDS):
    with h5py.File(file_path, 'r') as f:
        radar_data = f['fused_detections']
        points = np.array([[rec[field] for field in selected_fields] for rec in radar_data], dtype=np.float32)
        xyz = points[:, :2]  # (N,2)
        features = points[:, 2:]  # (N,3)
        # Add batch dim
        xyz = torch.tensor(xyz, dtype=torch.float32).unsqueeze(0).to(DEVICE)       # (1, N, 2)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, N, 3)
        return xyz, features

# -------------------------
# Dummy target for self-supervised training
# -------------------------
# Since you only want to train the backbone to produce features, we can do a simple reconstruction target or
# contrastive-style learning. Here we use a simple "autoencoder-like" target (identity) for demonstration.
# -------------------------
class DummyHead(nn.Module):
    def forward(self, backbone_features):
        # Just map backbone features back to some dummy target size
        return backbone_features  # identity

# -------------------------
# Load data
# -------------------------
xyz, features = load_h5_single(DATA_FILE)

# -------------------------
# Initialize model
# -------------------------
backbone = PointNet2Backbone(feature_channels=3).to(DEVICE)
head = DummyHead().to(DEVICE)

optimizer = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=LEARNING_RATE)
criterion = nn.MSELoss()  # dummy loss for self-supervised training

# -------------------------
# Training loop
# -------------------------
backbone.train()
head.train()

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    backbone_features, _, _ = backbone(xyz, features)  # (B, out_channels)
    output = head(backbone_features)
    # Dummy target: just reconstruct features to match backbone size (self-supervised)
    target = backbone_features.detach()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {loss.item():.6f}")

# -------------------------
# Save backbone weights
# -------------------------
torch.save(backbone.state_dict(), SAVE_PATH)
print(f"Backbone weights saved to {SAVE_PATH}")

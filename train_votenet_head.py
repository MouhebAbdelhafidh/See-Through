# train_votenet_head.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ---------------- Config ----------------
DATA_PATH = "precomputed_data.npz"
BATCH_SIZE = 32
EPOCHS = 0
LR = 1e-4
NUM_CLASSES = 11       # Semantic classes
NUM_SIZE_CLUSTER = 8   # Size bins
NUM_HEADING_BIN = 12   # Heading bins
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Dataset ----------------
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------- Model ----------------
class VoteNetHead(nn.Module):
    def __init__(self, in_dim=256):
        super(VoteNetHead, self).__init__()

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        # Objectness (bg / object)
        self.objectness = nn.Linear(256, 2)

        # Semantic classification
        self.semantic = nn.Linear(256, NUM_CLASSES)

        # BBox center
        self.center = nn.Linear(256, 3)

        # Size prediction
        self.size_scores = nn.Linear(256, NUM_SIZE_CLUSTER)
        self.size_residuals = nn.Linear(256, NUM_SIZE_CLUSTER * 3)

        # Heading prediction
        self.heading_scores = nn.Linear(256, NUM_HEADING_BIN)
        self.heading_residuals = nn.Linear(256, NUM_HEADING_BIN)

        # Velocity prediction
        self.velocity = nn.Linear(256, 2)

    def forward(self, x):
        x = self.mlp(x)

        return {
            "classification": {
                "objectness": self.objectness(x),
                "semantic": self.semantic(x)
            },
            "bbox": {
                "center": self.center(x),
                "size_scores": self.size_scores(x),
                "size_residuals": self.size_residuals(x).view(-1, NUM_SIZE_CLUSTER, 3),
                "heading_scores": self.heading_scores(x),
                "heading_residuals": self.heading_residuals(x)
            },
            "velocity": self.velocity(x)
        }


# ---------------- Load Data ----------------
print(f"Loading features from: {DATA_PATH}")
data = np.load(DATA_PATH)

features = data["features"]  # shape (N, in_dim)
labels = data["labels"]      # semantic labels

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

train_dataset = FeatureDataset(X_train, y_train)
val_dataset = FeatureDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ---------------- Training ----------------
model = VoteNetHead(in_dim=features.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion_ce = nn.CrossEntropyLoss()
criterion_l1 = nn.SmoothL1Loss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for feats, labels in train_loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        outputs = model(feats)

        # Loss: semantic classification only (you can extend to bbox, velocity if labels available)
        loss_sem = criterion_ce(outputs["classification"]["semantic"], labels)

        # Example: bbox + velocity regularization (dummy zeros if no gt)
        loss_bbox = criterion_l1(outputs["bbox"]["center"], torch.zeros_like(outputs["bbox"]["center"]))
        loss_vel = criterion_l1(outputs["velocity"], torch.zeros_like(outputs["velocity"]))

        loss = loss_sem + 0.1 * loss_bbox + 0.1 * loss_vel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            outputs = model(feats)
            loss_sem = criterion_ce(outputs["classification"]["semantic"], labels)
            loss_bbox = criterion_l1(outputs["bbox"]["center"], torch.zeros_like(outputs["bbox"]["center"]))
            loss_vel = criterion_l1(outputs["velocity"], torch.zeros_like(outputs["velocity"]))
            loss = loss_sem + 0.1 * loss_bbox + 0.1 * loss_vel

            val_loss += loss.item()
            preds = outputs["classification"]["semantic"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"[Epoch {epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val Acc: {correct/total:.4f}")

# ---------------- Save ----------------
torch.save(model.state_dict(), "checkpoints/votenet_head.pth")
print("Model saved to votenet_head.pth")

# test_votenet_head.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ---------------- Config ----------------
DATA_PATH = "precomputed_data.npz"
MODEL_PATH = "checkpoints/votenet_head.pth"
BATCH_SIZE = 32
NUM_CLASSES = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset ----------------
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ---------------- Model ----------------
class VoteNetHead(nn.Module):
    def __init__(self, in_dim=256):
        super(VoteNetHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.objectness = nn.Linear(256, 2)
        self.semantic = nn.Linear(256, NUM_CLASSES)
        self.center = nn.Linear(256, 3)
        self.size_scores = nn.Linear(256, 8)
        self.size_residuals = nn.Linear(256, 8 * 3)
        self.heading_scores = nn.Linear(256, 12)
        self.heading_residuals = nn.Linear(256, 12)
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
                "size_residuals": self.size_residuals(x).view(-1, 8, 3),
                "heading_scores": self.heading_scores(x),
                "heading_residuals": self.heading_residuals(x)
            },
            "velocity": self.velocity(x)
        }

# ---------------- Load Data ----------------
print(f"Loading features from: {DATA_PATH}")
data = np.load(DATA_PATH)
features = data["features"]
labels = data["labels"]

dataset = FeatureDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Load Model ----------------
model = VoteNetHead(in_dim=features.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- Evaluation ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for feats, labs in dataloader:
        feats, labs = feats.to(DEVICE), labs.to(DEVICE)
        outputs = model(feats)
        preds = torch.argmax(outputs["classification"]["semantic"], dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())

# ---------------- Metrics ----------------
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
report = classification_report(all_labels, all_preds, zero_division=0)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print(f"Weighted F1 Score: {f1:.4f}\n")
print("Classification Report:\n")
print(report)

# evaluate_votenet.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from train_votenet_head import VoteNetHead  # only the head class

# ---------------- Config ----------------
DATA_PATH = "precomputed_data.npz"
MODEL_PATH = "votenet_head_finetuned.pth"
BATCH_SIZE = 32
NUM_CLASSES = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset ----------------
class FeatureDataset(Dataset):
    def __init__(self, feats, labs):
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.labs = torch.tensor(labs, dtype=torch.long)

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, idx):
        return self.feats[idx], self.labs[idx]

# ---------------- Load Data ----------------
print("Loading precomputed features...")
data = np.load(DATA_PATH, allow_pickle=True)
features = data["features"]
labels = data["labels"]

dataset = FeatureDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Load Model ----------------
print("Loading model...")
model = VoteNetHead(in_dim=features.shape[1], num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- Evaluation ----------------
print("Evaluating...")
all_preds = []
all_labels = []

with torch.no_grad():
    for feats, labs in dataloader:
        feats, labs = feats.to(DEVICE), labs.to(DEVICE)
        outputs = model(feats)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labs.cpu().numpy())

# ---------------- Metrics ----------------
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, zero_division=0)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(report)

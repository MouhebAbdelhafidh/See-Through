import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------------- Config ----------------
DATA_PATH = "precomputed_data.npz"
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
NUM_CLASSES = 11  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------

# Load precomputed features
print("Loading precomputed features...")
data = np.load(DATA_PATH, allow_pickle=True)
features = data["features"]    # shape: (N, 1024)
labels = data["labels"]        # shape: (N,)

# Train/val split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42, shuffle=True
)

# Dataset wrapper
class FeatureDataset(Dataset):
    def __init__(self, feats, labs):
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.labs = torch.tensor(labs, dtype=torch.long)

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, idx):
        return self.feats[idx], self.labs[idx]

train_dataset = FeatureDataset(X_train, y_train)
val_dataset = FeatureDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class VoteNetHead(nn.Module):
    def __init__(self, in_dim=1024, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = VoteNetHead(in_dim=features.shape[1], num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for feats, labs in train_loader:
        feats, labs = feats.to(DEVICE), labs.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labs).sum().item()
        total += labs.size(0)

    train_acc = correct / total
    avg_loss = total_loss / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for feats, labs in val_loader:
            feats, labs = feats.to(DEVICE), labs.to(DEVICE)
            outputs = model(feats)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labs).sum().item()
            val_total += labs.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {avg_loss:.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"Val Acc: {val_acc:.4f}")

# Save model
torch.save(model.state_dict(), "votenet_head.pth")
print("Model saved as votenet_head.pth")

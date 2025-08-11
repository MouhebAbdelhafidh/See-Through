import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Config ---
DATA_PATH = "precomputed_data.npz"
MODEL_PATH = "votenet_head_finetuned.pth"  
BATCH_SIZE = 32
ADDITIONAL_EPOCHS = 20
LR = 1e-5
NUM_CLASSES = 11
DEVICE = torch.device("cpu")  # force CPU for local
print("Using device:", DEVICE)

# --- Load data ---
print("Loading precomputed features...")
data = np.load(DATA_PATH, allow_pickle=True)
features = data["features"]
labels = data["labels"]

X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42, shuffle=True
)

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

# --- Model definition ---
class VoteNetHead(nn.Module):
    def __init__(self, in_dim=1024, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(in_dim, 512),  # ⚠ this should probably be 1024 → 512, not in_dim → 512
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = VoteNetHead(in_dim=features.shape[1], num_classes=NUM_CLASSES).to(DEVICE)

# # --- Load saved weights on CPU ---
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# print("Loaded saved model weights.")

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# # --- Continue training ---
# for epoch in range(ADDITIONAL_EPOCHS):
#     model.train()
#     total_loss, correct, total = 0, 0, 0
#     for feats, labs in train_loader:
#         feats, labs = feats.to(DEVICE), labs.to(DEVICE)
#         optimizer.zero_grad()
#         outputs = model(feats)
#         loss = criterion(outputs, labs)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * feats.size(0)
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == labs).sum().item()
#         total += labs.size(0)

#     train_acc = correct / total
#     avg_loss = total_loss / total

#     # Validation
#     model.eval()
#     val_correct, val_total = 0, 0
#     with torch.no_grad():
#         for feats, labs in val_loader:
#             feats, labs = feats.to(DEVICE), labs.to(DEVICE)
#             outputs = model(feats)
#             _, predicted = torch.max(outputs, 1)
#             val_correct += (predicted == labs).sum().item()
#             val_total += labs.size(0)

#     val_acc = val_correct / val_total
#     print(f"Epoch [{epoch+1}/{ADDITIONAL_EPOCHS}] "
#           f"Loss: {avg_loss:.4f} "
#           f"Train Acc: {train_acc:.4f} "
#           f"Val Acc: {val_acc:.4f}")

# # --- Save the updated model ---
# torch.save(model.state_dict(), "votenet_head_finetuned1.pth")
# print("Model saved as votenet_head_finetuned1.pth")


# --- Load model ---
# model = VoteNetHead(in_dim=1024, num_classes=NUM_CLASSES).to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.eval()
# print("Model loaded successfully.")

# # --- Load data ---
# data = np.load(DATA_PATH, allow_pickle=True)
# features = data["features"]  # shape: (N, 1024)
# labels = data["labels"]

# # --- Pick a random sample ---
# idx = np.random.randint(len(features))
# sample_feat = torch.tensor(features[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)  # shape: (1, 1024)
# true_label = labels[idx]

# # --- Run inference ---
# with torch.no_grad():
#     outputs = model(sample_feat)
#     probs = torch.softmax(outputs, dim=1)
#     pred_class = torch.argmax(probs, dim=1).item()

# print(f"Sample index: {idx}")
# print(f"True label: {true_label}")
# print(f"Predicted class: {pred_class}")
# print(f"Probabilities: {probs.cpu().numpy()}")






#-----------------------------------------------------------#


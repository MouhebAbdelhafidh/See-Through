# train_on_features.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import random

# ------------------ Config ------------------
PRECOMP_PATH = "precomputed_data.npz"
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_BEST = "head_best.pt"
# -------------------------------------------

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=512, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

def main():
    if not os.path.exists(PRECOMP_PATH):
        raise RuntimeError(f"Precomputed file not found: {PRECOMP_PATH}. Run precompute_features.py first.")

    data = np.load(PRECOMP_PATH, allow_pickle=True)
    features = data['features']   # (N, D)
    labels = data['labels']       # (N,)

    N = features.shape[0]
    print(f"Loaded precomputed features: N={N}, dim={features.shape[1]}")

    # shuffle and split
    idxs = np.arange(N)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(idxs)
    n_train = int(N * TRAIN_RATIO)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:]

    train_feats = features[train_idx]
    train_labels = labels[train_idx]
    val_feats = features[val_idx]
    val_labels = labels[val_idx]

    train_ds = FeatureDataset(train_feats, train_labels)
    val_ds = FeatureDataset(val_feats, val_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    in_dim = features.shape[1]
    num_classes = len(np.unique(labels))
    print(f"Num classes = {num_classes}")

    model = MLPHead(in_dim, hidden=512, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        total_acc = 0.0
        for feats_batch, labs_batch in train_loader:
            feats_batch = feats_batch.to(DEVICE)
            labs_batch = labs_batch.to(DEVICE)
            logits = model(feats_batch)
            loss = criterion(logits, labs_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats_batch.size(0)
            total_acc += accuracy(logits.detach(), labs_batch) * feats_batch.size(0)
            total_samples += feats_batch.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_acc / total_samples

        # validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_samples = 0
        with torch.no_grad():
            for feats_batch, labs_batch in val_loader:
                feats_batch = feats_batch.to(DEVICE)
                labs_batch = labs_batch.to(DEVICE)
                logits = model(feats_batch)
                loss = criterion(logits, labs_batch)
                val_loss += loss.item() * feats_batch.size(0)
                val_acc += accuracy(logits, labs_batch) * feats_batch.size(0)
                val_samples += feats_batch.size(0)
        val_loss /= max(1, val_samples)
        val_acc /= max(1, val_samples)

        print(f"Epoch {epoch}/{EPOCHS} — train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc
            }, SAVE_BEST)
            print(f"Saved best model to {SAVE_BEST} (val_acc={val_acc:.4f})")

    print("Training finished. Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()

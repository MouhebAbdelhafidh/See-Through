# train_feature_extractor.py

import os
import glob
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from extract_features import PointNet2Backbone  # your backbone model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RadarPointCloudDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels_set = set()
        
        files = glob.glob(os.path.join(data_dir, "*.h5"))
        for file in files:
            with h5py.File(file, 'r') as f:
                if 'data' not in f:
                    continue
                data = f['data'][:]  # [N, 9] or more

                if data.shape[1] < 6:
                    continue  # skip invalid data

                x_cc = data[:, 0]
                y_cc = data[:, 1]
                vr = data[:, 2]
                vr_comp = data[:, 3]
                rcs = data[:, 4]
                label_id = data[:, 5]

                xyz = torch.tensor(np.stack([x_cc, y_cc], axis=-1), dtype=torch.float32)       # [N, 2]
                features = torch.tensor(np.stack([vr, vr_comp, rcs], axis=-1), dtype=torch.float32)  # [N, 3]
                label = int(label_id[0])  # assuming single label per sample

                self.samples.append((xyz, features, label))
                self.labels_set.add(label)

        self.num_classes = len(self.labels_set)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xyz, features, label = self.samples[idx]
        return xyz, features, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    xyzs, features, labels = zip(*batch)
    xyzs = torch.nn.utils.rnn.pad_sequence(xyzs, batch_first=True)
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    return xyzs, features, torch.stack(labels)


def train():
    dataset = RadarPointCloudDataset("FusedData")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = PointNet2Backbone().to(device)
    classifier = nn.Sequential(
        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Linear(128, dataset.num_classes)
    ).to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        total_loss = 0

        for xyz, features, labels in dataloader:
            xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)

            optimizer.zero_grad()
            feat, _, _ = model(xyz, features)  # [B, 1024]
            pred = classifier(feat)            # [B, num_classes]
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save the trained feature extractor
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pointnetpp_backbone.pth")
    print("✅ Feature extractor saved to checkpoints/pointnetpp_backbone.pth")


if __name__ == "__main__":
    train()

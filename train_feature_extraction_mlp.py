import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from feature_extraction_mlp import RadarMLP  # import your MLP

# ---------------- Config ----------------
DATA_DIR = "SplitBySensor"
BATCH_SIZE = 1024
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_FIELDS = ['x_cc', 'y_cc', 'vr', 'vr_compensated', 'rcs']
OUTPUT_DIR = "SensorMLPs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------

def train_mlp_for_sensor(sensor_folder):
    h5_files = glob.glob(os.path.join(sensor_folder, "*.h5"))
    if not h5_files:
        print(f"No h5 files found in {sensor_folder}")
        return

    # Load all records for this sensor
    all_inputs = []
    all_targets = []

    for fpath in h5_files:
        with h5py.File(fpath, 'r') as f:
            if 'detections' not in f:
                continue
            ds = f['detections']
            for rec in ds:
                # Input features
                inp = np.array([rec[field] for field in FEATURE_FIELDS], dtype=np.float32)
                all_inputs.append(inp)
                # Target: 1024-d backbone features
                if 'backbone_feat' in ds.dtype.names:
                    target = np.array(rec['backbone_feat'], dtype=np.float32)
                else:
                    # If you don't have precomputed backbone features, you can
                    # initially use zeros or random; for real training, you need backbone output
                    target = np.zeros(1024, dtype=np.float32)
                all_targets.append(target)

    if not all_inputs:
        print(f"No data found for {sensor_folder}")
        return

    X = torch.tensor(np.stack(all_inputs), dtype=torch.float32)
    y = torch.tensor(np.stack(all_targets), dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize MLP
    model = RadarMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"Sensor {os.path.basename(sensor_folder)} | Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(dataset):.6f}")

    # Save the trained model
    save_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(sensor_folder)}_mlp.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model for {os.path.basename(sensor_folder)} -> {save_path}")


def main():
    sensor_folders = glob.glob(os.path.join(DATA_DIR, "sensor_*"))
    sensor_folders.sort()
    for folder in sensor_folders:
        print(f"Training MLP for {folder} ...")
        train_mlp_for_sensor(folder)


if __name__ == "__main__":
    main()

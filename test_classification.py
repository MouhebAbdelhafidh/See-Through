import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from extract_features import PointNet2Backbone  # your backbone import

# Config
H5_FILE = "RadarScenes/data/sequence_148/radar_data.h5"
BACKBONE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_DEVICE = BACKBONE_DEVICE
BATCH_SIZE = 256
NUM_CLASSES = 11
FPS = 100

CLASS_NAMES = [
    "Car", "Large Vehicle", "Truck", "Bus", "Train",
    "Bicycle", "Motorcycle", "Pedestrian", "Pedestrian Group",
    "Animal", "Other Dynamic"
]

CLASS_COLORS = np.array([
    [255, 0, 0],
    [255, 165, 0],
    [139, 0, 139],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 192, 203],
    [165, 42, 42],
    [0, 128, 0],
    [128, 128, 128]
]) / 255.0

CLASSIFIER_PATH = "votenet_head_finetuned.pth"

class VoteNetHead(nn.Module):
    def __init__(self, in_dim=1024, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0),
            nn.Linear(1024, 512),
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

def main():
    print(f"Using device: {BACKBONE_DEVICE}")
    print(f"Loading data from {H5_FILE} ...")

    with h5py.File(H5_FILE, 'r') as f:
        radar_data = f['radar_data'][:]

    # Group indices by unique timestamps
    timestamps, inverse_indices = np.unique(radar_data['timestamp'], return_inverse=True)
    num_frames = len(timestamps)
    print(f"Number of frames (unique timestamps): {num_frames}")

    backbone = PointNet2Backbone(feature_channels=3).to(BACKBONE_DEVICE).eval()
    classifier = VoteNetHead(in_dim=1024, num_classes=NUM_CLASSES).to(CLASSIFIER_DEVICE).eval()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=CLASSIFIER_DEVICE))

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter([], [], s=15)
    texts_pred = []
    texts_true = []

    ax.set_xlim(-50, 50)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("x_cc")
    ax.set_ylabel("y_cc")
    ax.grid(True)
    ax.set_title("Radar Detection Classification Over Time")

    # Legend for classes
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=CLASS_COLORS[i], markersize=10)
               for i in range(NUM_CLASSES)]
    ax.legend(handles, CLASS_NAMES, loc='upper right', bbox_to_anchor=(1.25, 1))

    def update(frame_idx):
        nonlocal texts_pred, texts_true
        # Clear old text annotations
        for t in texts_pred + texts_true:
            t.remove()
        texts_pred = []
        texts_true = []

        idxs = np.where(inverse_indices == frame_idx)[0]
        frame_data = radar_data[idxs]
        if len(frame_data) == 0:
            scatter.set_offsets(np.array([]))
            return scatter,

        xyz = np.stack([frame_data['x_cc'], frame_data['y_cc']], axis=1).astype(np.float32)
        feats = np.stack([frame_data['vr'], frame_data['vr_compensated'], frame_data['rcs']], axis=1).astype(np.float32)

        xyz_tensor = torch.from_numpy(xyz).unsqueeze(1).to(BACKBONE_DEVICE)
        feats_tensor = torch.from_numpy(feats).unsqueeze(1).to(BACKBONE_DEVICE)

        with torch.no_grad():
            l3_out, _, _ = backbone(xyz_tensor, feats_tensor)
            features = l3_out.squeeze(1)
            outputs = classifier(features)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        scatter.set_offsets(xyz)
        scatter.set_color(CLASS_COLORS[preds])

        # Add text: predicted and true labels next to each point
        for i, (x, y) in enumerate(xyz):
            pred_label = CLASS_NAMES[preds[i]]
            true_label_id = frame_data['label_id'][i]
            true_label = CLASS_NAMES[true_label_id] if true_label_id < NUM_CLASSES else "Unknown"
            txt_pred = ax.text(x, y, f"P:{pred_label}", color='black', fontsize=7, weight='bold', va='bottom')
            txt_true = ax.text(x, y, f"T:{true_label}", color='white', fontsize=7, va='top')
            texts_pred.append(txt_pred)
            texts_true.append(txt_true)

        ax.set_title(f"Frame {frame_idx + 1} / {num_frames} - Timestamp: {timestamps[frame_idx]}")

        return scatter, *texts_pred, *texts_true

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 // FPS, blit=False, repeat=True)
    plt.show()

if __name__ == "__main__":
    main()

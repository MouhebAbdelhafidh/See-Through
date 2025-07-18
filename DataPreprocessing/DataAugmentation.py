import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset wrapper ---
class RadarDataset(Dataset):
    def __init__(self, data_array):
        self.data = data_array.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- Generator ---
class Generator(nn.Module):
    def __init__(self, noise_dim=10, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
        )
    def forward(self, z):
        return self.net(z)
# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

# --- Load data with label_id=1 ---
def load_label1_points(h5_path):
    with h5py.File(h5_path, "r") as f:
        detections = f["detections"][:]
    label1 = detections[detections["label_id"] == 8]
    if len(label1) == 0:
        return np.array([]) 
    data = np.stack([
        label1["x_cc"],
        label1["y_cc"],
        label1["rcs"],
        label1["vr"],
        label1["vr_compensated"]
    ], axis=1)
    return data

def train_gan(dataloader, epochs=10, noise_dim=10):
    G = Generator(noise_dim=noise_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)

    criterion = nn.BCELoss()
    optim_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    optim_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for real_points in dataloader:
            real_points = real_points.to(DEVICE)
            batch_size = real_points.size(0)

            # --- Train Discriminator ---
            D.zero_grad()
            labels_real = torch.ones(batch_size, 1, device=DEVICE)
            output_real = D(real_points)
            loss_real = criterion(output_real, labels_real)

            noise = torch.randn(batch_size, noise_dim, device=DEVICE)
            fake_points = G(noise)
            labels_fake = torch.zeros(batch_size, 1, device=DEVICE)
            output_fake = D(fake_points.detach())
            loss_fake = criterion(output_fake, labels_fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            # --- Train Generator twice ---
            for _ in range(2):
                G.zero_grad()
                noise = torch.randn(batch_size, noise_dim, device=DEVICE)
                fake_points = G(noise)
                labels_gen = torch.ones(batch_size, 1, device=DEVICE)
                output_gen = D(fake_points)
                loss_G = criterion(output_gen, labels_gen)
                loss_G.backward()
                optim_G.step()

        print(f"  Epoch [{epoch+1}/{epochs}] Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")

    return G


def generate_points(G, num_points=1000, noise_dim=10):
    G.eval()
    with torch.no_grad():
        noise = torch.randn(num_points, noise_dim, device=DEVICE)
        return G(noise).cpu().numpy()

def save_generated_to_h5(fake_points, output_path):
    dtype = np.dtype([
        ("x_cc", "f4"), ("y_cc", "f4"), ("label_id", "u1"),
        ("track_id", "i4"), ("sensor_id", "u1"),
        ("rcs", "f4"), ("vr", "f4"), ("vr_compensated", "f4"),
    ])
    n = fake_points.shape[0]
    data = np.zeros(n, dtype=dtype)
    data["x_cc"] = fake_points[:, 0]
    data["y_cc"] = fake_points[:, 1]
    data["rcs"] = fake_points[:, 2]
    data["vr"] = fake_points[:, 3]
    data["vr_compensated"] = fake_points[:, 4]
    data["label_id"] = 8
    data["track_id"] = -1
    data["sensor_id"] = 0

    frame_dtype = np.dtype([
        ('odometry_index', '<i4'), ('timestamp', '<i8'),
        ('ego_velocity', '<f4'), ('ego_yaw_rate', '<f4'),
        ('detection_start_idx', '<i4'), ('detection_end_idx', '<i4')
    ])
    frames = np.array([(0, 0, 0.0, 0.0, 0, n)], dtype=frame_dtype)

    with h5py.File(output_path, "w") as f:
        f.attrs["sequence_name"] = "generated_label1"
        f.create_dataset("detections", data=data)
        f.create_dataset("frames", data=frames)

    print(f"✅ Saved: {output_path}")


def main():
    input_folder = "../NormlizedData"
    output_folder = "FakeData"
    os.makedirs(output_folder, exist_ok=True)

    h5_files = [f for f in os.listdir(input_folder) if f.endswith(".h5")]

    for h5_file in h5_files:
        input_path = os.path.join(input_folder, h5_file)
        print(f"📂 Processing: {input_path}")

        data = load_label1_points(input_path)
        if data.shape[0] < 50:
            print("⚠️ Skipping (not enough label 8 points)")
            continue

        dataset = RadarDataset(data)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        generator = train_gan(dataloader, epochs=10)
        fake = generate_points(generator, num_points=1000)

        output_file = os.path.splitext(h5_file)[0] + "_fake_label8.h5"
        output_path = os.path.join(output_folder, output_file)
        save_generated_to_h5(fake, output_path)

if __name__ == "__main__":
    main()

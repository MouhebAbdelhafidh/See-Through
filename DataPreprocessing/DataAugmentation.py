import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cpu")

# --- Dataset wrapper ---
class RadarDataset(Dataset):
    def __init__(self, data_array):
        self.data = data_array.astype(np.float32)  # shape (N, 5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- Generator ---
class Generator(nn.Module):
    def __init__(self, noise_dim=10, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
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

# --- Load and prepare data ---
def load_label0_points(h5_path):
    with h5py.File(h5_path, "r") as f:
        detections = f["detections"][:]
    label0 = detections[detections["label_id"] == 0]
    print(f"Found {len(label0)} points with label_id=0")  # Debug print
    if len(label0) == 0:
        raise ValueError("No points with label_id=0 found in this file.")
    data = np.stack([
        label0["x_cc"],
        label0["y_cc"],
        label0["rcs"],
        label0["vr"],
        label0["vr_compensated"]
    ], axis=1)
    return data


# --- Train GAN ---
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

            # Train Discriminator: real
            D.zero_grad()
            labels_real = torch.ones(batch_size, 1, device=DEVICE)
            output_real = D(real_points)
            loss_real = criterion(output_real, labels_real)

            # Train Discriminator: fake
            noise = torch.randn(batch_size, noise_dim, device=DEVICE)
            fake_points = G(noise)
            labels_fake = torch.zeros(batch_size, 1, device=DEVICE)
            output_fake = D(fake_points.detach())
            loss_fake = criterion(output_fake, labels_fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            # Train Generator
            G.zero_grad()
            labels_gen = torch.ones(batch_size, 1, device=DEVICE)  # try to fool D
            output_gen = D(fake_points)
            loss_G = criterion(output_gen, labels_gen)
            loss_G.backward()
            optim_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")

    return G

# --- Generate new points ---
def generate_points(G, num_points=1000, noise_dim=10):
    G.eval()
    with torch.no_grad():
        noise = torch.randn(num_points, noise_dim, device=DEVICE)
        fake_points = G(noise).cpu().numpy()
    return fake_points

# --- Save generated points in .h5 ---
def save_generated_to_h5(fake_points, output_path):
    # Create structured dtype as original detections
    dtype = np.dtype([
        ("x_cc", "f4"),
        ("y_cc", "f4"),
        ("label_id", "u1"),
        ("track_id", "i4"),
        ("sensor_id", "u1"),
        ("rcs", "f4"),
        ("vr", "f4"),
        ("vr_compensated", "f4"),
    ])

    n = fake_points.shape[0]
    data = np.zeros(n, dtype=dtype)

    data["x_cc"] = fake_points[:, 0]
    data["y_cc"] = fake_points[:, 1]
    data["rcs"] = fake_points[:, 2]
    data["vr"] = fake_points[:, 3]
    data["vr_compensated"] = fake_points[:, 4]

    # Fixed fields
    data["label_id"] = 0  # label_id of interest
    data["track_id"] = -1  # unknown tracks
    data["sensor_id"] = 0  # dummy sensor id, you may change if you want

    # Create dummy frames dataset for completeness
    frame_dtype = np.dtype([
        ('odometry_index', '<i4'),
        ('timestamp', '<i8'),
        ('ego_velocity', '<f4'),
        ('ego_yaw_rate', '<f4'),
        ('detection_start_idx', '<i4'),
        ('detection_end_idx', '<i4')
    ])
    frames = np.array([(0, 0, 0.0, 0.0, 0, n)], dtype=frame_dtype)

    with h5py.File(output_path, "w") as f:
        f.attrs["sequence_name"] = "generated_label1_sequence"
        f.create_dataset("detections", data=data)
        f.create_dataset("frames", data=frames)

    print(f"✅ Saved generated data to {output_path}")

# --- Main ---
if __name__ == "__main__":
    input_h5_path = "../NormlizedData/sequence_99.h5"
    output_h5_path = "generated_label0_data.h5"

    data = load_label0_points(input_h5_path)
    if len(data) == 0:
        print("No data for label_id=0 found. Exiting.")
        exit(1)

    dataset = RadarDataset(data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    generator_model = train_gan(dataloader, epochs=10)
    fake_points = generate_points(generator_model, num_points=1000)

    save_generated_to_h5(fake_points, output_h5_path)

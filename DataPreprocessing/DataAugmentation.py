import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RadarDataset(Dataset):
    def __init__(self, data_array):
        self.data = data_array.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Generator(nn.Module):
    def __init__(self, noise_dim=10, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, device=DEVICE)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)

    grad_outputs = torch.ones_like(d_interpolates)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(real_samples.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

def load_label_points(h5_path, label_id):
    with h5py.File(h5_path, "r") as f:
        detections = f["detections"][:]
    label_data = detections[detections["label_id"] == label_id]
    if len(label_data) == 0:
        return np.array([]) 
    data = np.stack([
        label_data["x_cc"],
        label_data["y_cc"],
        label_data["rcs"],
        label_data["vr"],
        label_data["vr_compensated"]
    ], axis=1)
    return data

def train_gan(dataloader, epochs=100, noise_dim=10, lambda_gp=5):
    G = Generator(noise_dim=noise_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)

    optim_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optim_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(epochs):
        for real_points in dataloader:
            real_points = real_points.to(DEVICE)
            batch_size = real_points.size(0)

            for _ in range(10): 
                noise = torch.randn(batch_size, noise_dim, device=DEVICE)
                fake_points = G(noise).detach()

                D_real = D(real_points).mean()
                D_fake = D(fake_points).mean()

                gp = compute_gradient_penalty(D, real_points, fake_points)
                loss_D = D_fake - D_real + lambda_gp * gp

                D.zero_grad()
                loss_D.backward()
                optim_D.step()

            noise = torch.randn(batch_size, noise_dim, device=DEVICE)
            fake_points = G(noise)
            loss_G = -D(fake_points).mean()

            G.zero_grad()
            loss_G.backward()
            optim_G.step()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {loss_D.item():.4f}  Loss G: {loss_G.item():.4f}")

    return G

def generate_points(G, num_points=1000, noise_dim=10):
    G.eval()
    with torch.no_grad():
        noise = torch.randn(num_points, noise_dim, device=DEVICE)
        return G(noise).cpu().numpy()

# --- Save to HDF5 ---
def save_generated_to_h5(fake_points, output_path, label_id):
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
    data["label_id"] = label_id
    data["track_id"] = -1
    data["sensor_id"] = 0

    frame_dtype = np.dtype([
        ('odometry_index', '<i4'), ('timestamp', '<i8'),
        ('ego_velocity', '<f4'), ('ego_yaw_rate', '<f4'),
        ('detection_start_idx', '<i4'), ('detection_end_idx', '<i4')
    ])
    frames = np.array([(0, 0, 0.0, 0.0, 0, n)], dtype=frame_dtype)

    with h5py.File(output_path, "w") as f:
        f.attrs["sequence_name"] = f"generated_label_{label_id}"
        f.create_dataset("detections", data=data)
        f.create_dataset("frames", data=frames)

    print(f"✅ Saved: {output_path}")

def main():
    input_folder = "../NormlizedData"
    base_output_folder = "FakeData"
    os.makedirs(base_output_folder, exist_ok=True)

    h5_files = [f for f in os.listdir(input_folder) if f.endswith(".h5")]

    for label_id in range(7, 11):  
        print(f"\n🔍 Processing label_id: {label_id}")
        label_output_folder = os.path.join(base_output_folder, f"label_{label_id}")
        os.makedirs(label_output_folder, exist_ok=True)

        for h5_file in h5_files:
            input_path = os.path.join(input_folder, h5_file)
            output_file = os.path.splitext(h5_file)[0] + f"_fake_label{label_id}.h5"
            output_path = os.path.join(label_output_folder, output_file)

            if os.path.exists(output_path):
                print(f"✅ Already exists, skipping: {output_path}")
                continue

            print(f"📂 File: {input_path}")
            data = load_label_points(input_path, label_id)

            if data.shape[0] < 50:
                print("⚠️ Skipping (not enough points for this label)")
                continue

            dataset = RadarDataset(data)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            generator = train_gan(dataloader, epochs=100)
            fake = generate_points(generator, num_points=100000)

            save_generated_to_h5(fake, output_path, label_id)


if __name__ == "__main__":
    main()
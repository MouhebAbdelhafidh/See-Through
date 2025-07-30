import os
import glob
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- PointNet++ helpers and classes ---

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, C = xyz.shape
    S = new_xyz.shape[1]
    group_idx = torch.arange(N, dtype=torch.long).to(xyz.device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, C = xyz.shape

        if self.npoint is not None:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
        else:
            new_xyz = xyz.mean(dim=1, keepdim=True)

        if self.radius is not None:
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.view(B, -1, 1, C)

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            grouped_xyz = xyz.view(B, 1, N, C)
            if points is not None:
                grouped_points = points.unsqueeze(2)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 1, 2)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, 3)[0]

        return new_xyz, new_points

class PointNet2Classifier(nn.Module):
    def __init__(self, num_classes=10):  # num_classes won't be used here, just dummy
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 0, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256, [256, 512, 1024])

    def forward(self, xyz):
        B, N, _ = xyz.shape
        points = None
        new_xyz, new_points = self.sa1(xyz, points)
        new_xyz, new_points = self.sa2(new_xyz, new_points)

        if self.sa3.npoint is None:
            x = torch.max(new_points, 2)[0]
            for conv, bn in zip(self.sa3.mlp_convs, self.sa3.mlp_bns):
                x = F.relu(bn(conv(x.unsqueeze(-1))))
            x = x.squeeze(-1)
        else:
            _, x = self.sa3(new_xyz, new_points)

        return new_points, x

# --- Dataset that just loads points from each .h5 file ---

class FusedRadarDataset(Dataset):
    def __init__(self, folder_path, num_points=1024):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.h5")))
        self.num_points = num_points
        if len(self.files) == 0:
            raise RuntimeError(f"No .h5 files found in {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, 'r') as f:
            data = f['fused_detections'][:]
            points = np.stack([data['x_cc'], data['y_cc'], data['vr']], axis=-1).astype(np.float32)

        # Normalize coordinates and velocity (adjust these if needed)
        points[:, :2] /= 100.0
        points[:, 2] /= 50.0

        # Sample fixed num_points with replacement if needed
        if points.shape[0] < self.num_points:
            pad_idx = np.random.choice(points.shape[0], self.num_points - points.shape[0], replace=True)
            points = np.vstack([points, points[pad_idx]])
        else:
            sample_idx = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[sample_idx]

        return torch.from_numpy(points)

def main():
    data_folder = r"C:\Users\asus\Desktop\See-Through\FusedData"  # <-- change this to your folder
    num_points = 1024
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FusedRadarDataset(data_folder, num_points=num_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = PointNet2Classifier().to(device)
    model.eval()

    all_local_feats = []
    all_global_feats = []

    with torch.no_grad():
        for points in dataloader:
            points = points.to(device)  # (B, N, 3)
            local_feat, global_feat = model(points)  # local_feat: (B, 128, 256), global_feat: (B, 1024)

            all_local_feats.append(local_feat.cpu())
            all_global_feats.append(global_feat.cpu())

    local_feats = torch.cat(all_local_feats, dim=0)
    global_feats = torch.cat(all_global_feats, dim=0)

    print(f"Extracted local features shape: {local_feats.shape}")
    print(f"Extracted global features shape: {global_feats.shape}")

    # Save features and model
    torch.save({
        "local_features": local_feats,
        "global_features": global_feats,
    }, "radar_features.pth")

    torch.save(model.state_dict(), "pointnet2_radar_model.pth")

    print("Feature extraction and model saving done.")

if __name__ == "__main__":
    main()

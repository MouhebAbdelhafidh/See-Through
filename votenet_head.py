import torch
import torch.nn as nn


class VoteNetHead(nn.Module):
    def __init__(self,
                 in_dim=1024,
                 num_classes=11,
                 hidden_dim=256,
                 num_heading_bins=12,
                 num_size_clusters=8):
        """
        Args:
            in_dim: dimension of backbone features (e.g. 1024)
            num_classes: semantic classes count
            hidden_dim: shared feature dim after shared MLP
            num_heading_bins: number of discrete heading bins (for classification + residual)
            num_size_clusters: number of size clusters (each has 3 residuals: w, l, h)
        """
        super().__init__()
        # Shared MLP
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # --------------------
        # Classification branch
        # --------------------
        self.objectness = nn.Linear(hidden_dim, 2)       # background / object
        self.semantic = nn.Linear(hidden_dim, num_classes)

        # --------------------
        # Bounding box branch
        # --------------------
        # center regression (x, y, z)
        self.center_reg = nn.Linear(hidden_dim, 3)

        # size: classification over clusters + residuals per cluster
        self.num_size_clusters = num_size_clusters
        self.size_scores = nn.Linear(hidden_dim, num_size_clusters)
        self.size_residuals = nn.Linear(hidden_dim, num_size_clusters * 3)

        # heading: classification + residual per bin
        self.num_heading_bins = num_heading_bins
        self.heading_scores = nn.Linear(hidden_dim, num_heading_bins)
        self.heading_residuals = nn.Linear(hidden_dim, num_heading_bins)

        # --------------------
        # Radar-specific motion
        # --------------------
        self.velocity = nn.Linear(hidden_dim, 2)  # vx, vy

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: (B, in_dim) feature vectors

        Returns:
            dict with:
                classification: {objectness, semantic}
                bbox: {center, size_scores, size_residuals, heading_scores, heading_residuals}
                velocity
        """
        feat = self.shared(x)

        # Classification outputs
        classification = {
            "objectness": self.objectness(feat),  # (B, 2)
            "semantic": self.semantic(feat)       # (B, num_classes)
        }

        # Bounding box outputs
        size_res = self.size_residuals(feat).view(-1, self.num_size_clusters, 3)
        bbox = {
            "center": self.center_reg(feat),              # (B, 3)
            "size_scores": self.size_scores(feat),        # (B, num_size_clusters)
            "size_residuals": size_res,                   # (B, num_size_clusters, 3)
            "heading_scores": self.heading_scores(feat),  # (B, num_heading_bins)
            "heading_residuals": self.heading_residuals(feat)  # (B, num_heading_bins)
        }

        # Velocity
        velocity = self.velocity(feat)  # (B, 2)

        return {
            "classification": classification,
            "bbox": bbox,
            "velocity": velocity
        }

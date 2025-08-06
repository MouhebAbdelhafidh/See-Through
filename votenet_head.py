# votenet_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Helpers (index, FPS, ball query)
# -----------------------
def index_points(points, idx):
    """
    Gather points/features using indices.

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K) indices
    Returns:
        new_points: (B, S, C) or (B, S, K, C)
    """
    B = points.shape[0]
    # view_shape = [B] + [1]*(len(idx.shape)-1)
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """
    Basic FPS implementation.

    Args:
        xyz: (B, N, D)
        npoint: int
    Returns:
        centroids: (B, npoint) indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].view(B, 1, -1)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def ball_query(radius, nsample, xyz, new_xyz):
    """
    For each center in new_xyz, find up to nsample neighbors in xyz within 'radius'.
    Args:
        radius: float
        nsample: int
        xyz: (B, N, D)
        new_xyz: (B, S, D)
    Returns:
        group_idx: (B, S, nsample) indices into xyz
    """
    device = xyz.device
    B, S, _ = new_xyz.shape
    N = xyz.shape[1]

    # pairwise distance: (B, S, N)
    dist = torch.cdist(new_xyz, xyz)  # Euclidean distances

    # initial indices (0..N-1) broadcasted -> (B, S, N)
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).expand(B, S, N)

    # mask out points outside radius
    mask = dist > radius
    group_first = group_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, N)
    group_idx = torch.where(mask, group_first, group_idx)

    # sort by distance and select first nsample
    _, sort_idx = torch.sort(dist, dim=-1)
    group_idx = torch.gather(group_idx, -1, sort_idx)[:, :, :nsample]  # (B, S, nsample)

    return group_idx


# -----------------------
# Voting Module
# -----------------------
class VotingModule(nn.Module):
    """
    Each seed predicts one or more votes: (offset -> vote coords) and vote features.
    Works with D=2 coordinates.
    """
    def __init__(self, in_channels, vote_factor=1, vote_feat_dim=128):
        """
        Args:
            in_channels: seed feature dim (C)
            vote_factor: number of votes per seed (usually 1)
            vote_feat_dim: dimensionality of produced vote features (F)
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.vote_feat_dim = vote_feat_dim

        # Offset predictor -> outputs 2 * vote_factor channels (2D offsets)
        self.offset_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, 2 * vote_factor, 1)
        )

        # Feature predictor -> outputs vote_feat_dim * vote_factor channels
        self.feature_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, vote_feat_dim * vote_factor, 1)
        )

    def forward(self, seed_xyz, seed_features):
        """
        Args:
            seed_xyz: (B, M, 2)
            seed_features: (B, M, C)
        Returns:
            votes_xyz: (B, M * vote_factor, 2)
            votes_feats: (B, M * vote_factor, vote_feat_dim)
        """
        B, M, C = seed_features.shape
        feat = seed_features.permute(0, 2, 1)  # (B, C, M)

        offsets = self.offset_conv(feat)      # (B, 2*vf, M)
        vote_feats = self.feature_conv(feat)  # (B, vf*F, M)

        # reshape offsets -> (B, M*vf, 2)
        offsets = offsets.view(B, self.vote_factor, 2, M).permute(0, 3, 1, 2).contiguous()
        offsets = offsets.view(B, M * self.vote_factor, 2)

        # reshape vote_feats -> (B, M*vf, F)
        vote_feats = vote_feats.view(B, self.vote_factor, self.vote_feat_dim, M).permute(0, 3, 1, 2).contiguous()
        vote_feats = vote_feats.view(B, M * self.vote_factor, self.vote_feat_dim)

        # expand seed coords to match vote_factor and add offsets
        seed_xyz_exp = seed_xyz.unsqueeze(2).expand(-1, -1, self.vote_factor, -1).contiguous()
        seed_xyz_exp = seed_xyz_exp.view(B, M * self.vote_factor, 2)

        votes_xyz = seed_xyz_exp + offsets  # (B, M*vf, 2)
        return votes_xyz, vote_feats


# -----------------------
# Vote Aggregation (sample proposals and aggregate)
# -----------------------
class VoteAggregation(nn.Module):
    """
    Sample votes to get proposal centers and aggregate nearby votes to produce proposal features.
    """
    def __init__(self, nproposals=128, radius=0.3, nsample=16, in_feat_dim=128, out_feat_dim=256):
        """
        Args:
            nproposals: number of proposals (P)
            radius: grouping radius
            nsample: number of votes per group (K)
            in_feat_dim: input vote feature dim (F)
            out_feat_dim: output aggregated feature dim (C_out)
        """
        super().__init__()
        self.nproposals = nproposals
        self.radius = radius
        self.nsample = nsample

        # small MLP (Conv2d) that expects input channels = (2 + F)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_feat_dim + 2, out_feat_dim, 1),
            nn.BatchNorm2d(out_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feat_dim, out_feat_dim, 1),
            nn.BatchNorm2d(out_feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, votes_xyz, votes_feats):
        """
        Args:
            votes_xyz: (B, V, 2)
            votes_feats: (B, V, F)
        Returns:
            new_xyz: (B, P, 2) proposal centers
            pooled_feats: (B, P, out_feat_dim)
        """
        B, V, _ = votes_xyz.shape

        # 1) FPS on votes to choose proposal centers
        fps_idx = farthest_point_sample(votes_xyz, self.nproposals)  # (B, P)
        new_xyz = index_points(votes_xyz, fps_idx)                   # (B, P, 2)

        # 2) group votes around each proposal center
        group_idx = ball_query(self.radius, self.nsample, votes_xyz, new_xyz)  # (B, P, K)
        grouped_votes = index_points(votes_xyz, group_idx)       # (B, P, K, 2)
        grouped_feats = index_points(votes_feats, group_idx)     # (B, P, K, F)

        # 3) relative coords
        grouped_xyz_norm = grouped_votes - new_xyz.unsqueeze(2)  # (B, P, K, 2)

        # 4) concat coords & feats -> (B, P, K, 2+F)
        combined = torch.cat([grouped_xyz_norm, grouped_feats], dim=-1)
        combined = combined.permute(0, 3, 1, 2)  # (B, 2+F, P, K)

        # 5) MLP + max-pool over K
        processed = self.mlp(combined)               # (B, out_feat_dim, P, K)
        pooled = torch.max(processed, dim=-1)[0]    # (B, out_feat_dim, P)

        return new_xyz, pooled.permute(0, 2, 1)      # (B, P, out_feat_dim)


# -----------------------
# Proposal Head (objectness + class logits)
# -----------------------
class ProposalHead(nn.Module):
    """
    Classification head per proposal: predicts objectness logits and class logits.
    """
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.cls_layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels // 2, 1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.objectness_fc = nn.Conv1d(in_channels // 2, 1, 1)     # scalar objectness
        self.class_fc = nn.Conv1d(in_channels // 2, num_classes, 1) # class logits

    def forward(self, aggregated_feats):
        """
        Args:
            aggregated_feats: (B, P, C)
        Returns:
            objectness: (B, P) logits
            class_logits: (B, P, num_classes)
        """
        x = aggregated_feats.permute(0, 2, 1)  # (B, C, P)
        x = self.cls_layers(x)                 # (B, C', P)
        obj = self.objectness_fc(x).squeeze(1) # (B, P)
        cls = self.class_fc(x).permute(0, 2, 1) # (B, P, num_classes)
        return obj, cls


# -----------------------
# Full VoteNet Head
# -----------------------
class VoteNetHead(nn.Module):
    """
    VoteNet Head wrapper:
      seeds -> voting -> votes -> aggregation -> proposals -> classification
    """
    def __init__(self,
                 seed_feat_dim,
                 vote_feat_dim=128,
                 vote_factor=1,
                 nproposals=128,
                 agg_feat_dim=256,
                 num_classes=10,
                 radius=0.3,
                 nsample=16):
        """
        Args:
            seed_feat_dim: feature dim of seeds (C)
            vote_feat_dim: per-vote feature dim (F)
            vote_factor: votes per seed
            nproposals: number of proposals (P)
            agg_feat_dim: aggregated proposal feature dim
            num_classes: number of classes to predict
            radius: grouping radius for aggregation
            nsample: samples per group
        """
        super().__init__()
        self.voting = VotingModule(in_channels=seed_feat_dim, vote_factor=vote_factor, vote_feat_dim=vote_feat_dim)
        self.agg = VoteAggregation(nproposals=nproposals, radius=radius, nsample=nsample,
                                   in_feat_dim=vote_feat_dim, out_feat_dim=agg_feat_dim)
        self.proposal = ProposalHead(in_channels=agg_feat_dim, num_classes=num_classes)

    def forward(self, seed_xyz, seed_feats):
        """
        Args:
            seed_xyz: (B, M, 2)
            seed_feats: (B, M, C)
        Returns:
            proposals_xyz: (B, P, 2)
            objectness_logits: (B, P)
            class_logits: (B, P, num_classes)
        """
        votes_xyz, votes_feats = self.voting(seed_xyz, seed_feats)     # (B, V, 2), (B, V, F)
        proposals_xyz, agg_feats = self.agg(votes_xyz, votes_feats)    # (B, P, 2), (B, P, out_dim)
        obj_logits, class_logits = self.proposal(agg_feats)            # (B, P), (B, P, num_classes)
        return proposals_xyz, obj_logits, class_logits


# If run as script, a small smoke-test
if __name__ == "__main__":
    # smoke test with dummy data
    B, M, C = 1, 512, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_xyz = torch.rand(B, M, 2).to(device)
    seed_feats = torch.rand(B, M, C).to(device)
    head = VoteNetHead(seed_feat_dim=C, vote_feat_dim=128, vote_factor=1,
                       nproposals=64, agg_feat_dim=256, num_classes=10).to(device)
    proposals_xyz, obj_logits, class_logits = head(seed_xyz, seed_feats)
    print("proposals_xyz", proposals_xyz.shape)
    print("obj_logits", obj_logits.shape)
    print("class_logits", class_logits.shape)

# final_model.py
import torch
import torch.nn as nn
import numpy as np
import os

# adjust these imports depending on how you saved the backbone/head files
# If your backbone is in this same file, import accordingly.
from votenet_head import VoteNetHead
from extract_features import PointNet2Backbone


class FinalModel(nn.Module):
    """
    Connects a PointNet2 backbone to a VoteNet-style head.
    - backbone.forward(xyz, features) -> (B, 1024) global features
    - head takes (B, 1024) and returns dict of outputs
    """
    def __init__(self,
                 backbone_feature_channels=5,
                 head_in_dim=1024,
                 num_classes=11,
                 freeze_backbone=False,
                 **head_kwargs):
        super().__init__()
        # instantiate backbone and head
        self.backbone = PointNet2Backbone(feature_channels=backbone_feature_channels)
        self.head = VoteNetHead(in_dim=head_in_dim, num_classes=num_classes, **head_kwargs)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, xyz, point_features):
        """
        Args:
            xyz: (B, N, 2) or (B, N, 3) tensor of point coordinates (your backbone expects 2 here)
            point_features: (B, N, C) per-point features (e.g., vr, rcs, etc.)
        Returns:
            head_outputs: dict from VoteNetHead with keys like objectness, center_reg, ...
            aux: dict with backbone intermediate outputs (optional)
        """
        # Backbone: returns (B, out_dim) global features and optionally aux data
        # The backbone in your file returns (l3_features.squeeze(1), l1_features, aux_dict)
        global_feat, l1_feat, aux = self.backbone(xyz, point_features)

        # Ensure shape (B, in_dim) for head
        # Some backbones might return (B, out_dim, 1) so check:
        if global_feat.dim() == 3 and global_feat.size(2) == 1:
            global_feat = global_feat.squeeze(2)
        # If global_feat has shape (B, out_dim, 1) or (B, 1, out_dim) handle above.

        head_out = self.head(global_feat)

        return head_out, aux

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None, strict=True):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, strict=strict)


if __name__ == "__main__":
    # ---------- quick tests ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example 1: dummy input (random)
    B = 2
    N = 512              # number of points (consistent with backbone SA1 nsample / npoint choices)
    coord_dim = 2        # your backbone uses 2D coords (x_cc, y_cc)
    feat_dim = 5         # number of per-point features used by backbone (set to same as declared in backbone)

    dummy_xyz = torch.randn(B, N, coord_dim).to(device)
    dummy_feats = torch.randn(B, N, feat_dim).to(device)

    model = FinalModel(backbone_feature_channels=feat_dim, head_in_dim=1024, num_classes=11).to(device)
    model.eval()
    with torch.no_grad():
        head_out, aux = model(dummy_xyz, dummy_feats)

    print("Head output keys:", list(head_out.keys()))
    for k, v in head_out.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("Aux keys (backbone):", list(aux.keys()) if isinstance(aux, dict) else type(aux))

    # Example 2: run on precomputed feature vectors instead of full backbone
    # If you have precomputed_data.npz containing 'features' shape (N_total, 1024),
    # you can feed them directly to the head (skips backbone)
    NPZ_PATH = "precomputed_data.npz"
    if os.path.exists(NPZ_PATH):
        data = np.load(NPZ_PATH, allow_pickle=True)
        features = data["features"][:8].astype(np.float32)  # take first 8
        x = torch.from_numpy(features).to(device)
        model_head_only = model.head  # head expects (B, 1024)
        model_head_only.eval()
        with torch.no_grad():
            out = model_head_only(x)
        print("\nRan head-only on precomputed features:")
        for k, v in out.items():
            print(f"  {k}: {tuple(v.shape)}")

    print("\nDone.")

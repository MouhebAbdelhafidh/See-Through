# final_model.py
import torch
import torch.nn as nn
import numpy as np
import os

from votenet_head import VoteNetHead
from extract_features import PointNet2Backbone


class FinalModel(nn.Module):
    """
    Connects a PointNet2 backbone to a VoteNet-style head.
    - backbone.forward(xyz, features) -> (B, 1024) global features
    - head takes (B, 1024) and returns dict of outputs (classification, bbox, velocity)
    """
    def __init__(self,
                 backbone_feature_channels=5,
                 head_in_dim=1024,
                 num_classes=11,
                 freeze_backbone=True,
                 **head_kwargs):
        super().__init__()
        self.backbone = PointNet2Backbone(feature_channels=backbone_feature_channels)
        self.head = VoteNetHead(in_dim=head_in_dim, num_classes=num_classes, **head_kwargs)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, xyz, point_features):
        """
        Args:
            xyz: (B, N, coord_dim) tensor of point coordinates
            point_features: (B, N, C) per-point features
        Returns:
            head_outputs: dict from VoteNetHead (nested dicts)
            aux: dict with backbone intermediate outputs (optional)
        """
        global_feat, l1_feat, aux = self.backbone(xyz, point_features)

        # Squeeze if needed (PointNet2 can output (B, 1024, 1))
        if global_feat.dim() == 3 and global_feat.size(-1) == 1:
            global_feat = global_feat.squeeze(-1)

        head_out = self.head(global_feat)
        return head_out, aux

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None, strict=True):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, strict=strict)


def _print_nested_shapes(d, indent=0):
    """Helper to nicely print nested dict shapes."""
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + f"{k}:")
            _print_nested_shapes(v, indent + 1)
        else:
            print("  " * indent + f"{k}: {tuple(v.shape)}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build model ---
    model = FinalModel(
        backbone_feature_channels=5,  # match your radar features
        head_in_dim=1024,
        num_classes=11
    ).to(device)

    # --- Load pretrained backbone & head ---
    print("Loading pretrained weights...")
    model.backbone.load_state_dict(
        torch.load("checkpoints/pointnetpp_backbone.pth", map_location=device)
    )
    # Load head weights with key remapping
    state_dict = torch.load("checkpoints/votenet_head.pth", map_location=device)
    key_map = {
        "mlp.0.weight": "shared.0.weight",
        "mlp.0.bias": "shared.0.bias",
        "mlp.2.weight": "shared.1.weight",
        "mlp.2.bias": "shared.1.bias",
        "mlp.2.running_mean": "shared.1.running_mean",
        "mlp.2.running_var": "shared.1.running_var",
        "mlp.3.weight": "shared.3.weight",
        "mlp.3.bias": "shared.3.bias",
        "mlp.5.weight": "shared.4.weight",
        "mlp.5.bias": "shared.4.bias",
        "mlp.5.running_mean": "shared.4.running_mean",
        "mlp.5.running_var": "shared.4.running_var",
        "center.weight": "center_reg.weight",
        "center.bias": "center_reg.bias",
    }

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = key_map.get(k, k)
        new_state_dict[new_k] = v

    model.head.load_state_dict(new_state_dict, strict=False)   

    model.eval()

    # --- Dummy test ---
    B, N, coord_dim, feat_dim = 2, 512, 2, 5
    dummy_xyz = torch.randn(B, N, coord_dim).to(device)
    dummy_feats = torch.randn(B, N, feat_dim).to(device)

    with torch.no_grad():
        head_out, aux = model(dummy_xyz, dummy_feats)

    print("Head outputs (nested shapes):")
    _print_nested_shapes(head_out)
    print("Aux keys (backbone):", list(aux.keys()) if isinstance(aux, dict) else type(aux))

    # --- Head-only test with precomputed features ---
    NPZ_PATH = "precomputed_data.npz"
    if os.path.exists(NPZ_PATH):
        data = np.load(NPZ_PATH, allow_pickle=True)
        features = data["features"][:8].astype(np.float32)
        x = torch.from_numpy(features).to(device)

        model_head_only = model.head
        model_head_only.eval()
        with torch.no_grad():
            out = model_head_only(x)

        print("\nHead-only on precomputed features:")
        _print_nested_shapes(out)

    print("\nDone.")

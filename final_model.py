import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import json
from votenet_head import VoteNetHead
from extract_features import PointNet2Backbone


class FinalRadarModel(nn.Module):
    """Pipeline: PointNet2 -> VoteNetHead"""
    def __init__(self,
                 backbone_feature_channels=5,  
                 head_in_dim=1024,
                 num_classes=11,
                 freeze_backbone=True):
        super().__init__()
        self.backbone = PointNet2Backbone(feature_channels=backbone_feature_channels)
        self.head = VoteNetHead(in_dim=head_in_dim, num_classes=num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, xyz, point_features):
        # Extract global features
        global_feat, _, aux = self.backbone(xyz, point_features)

        if global_feat.dim() == 3 and global_feat.size(-1) == 1:
            global_feat = global_feat.squeeze(-1)

        # VoteNet predictions
        head_out = self.head(global_feat)
        return head_out, aux

    def load_pretrained(self, backbone_path, head_path, device):
        # Load backbone
        self.backbone.load_state_dict(torch.load(backbone_path, map_location=device))

        # Load head with remapping
        state_dict = torch.load(head_path, map_location=device)
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
        new_state_dict = {key_map.get(k, k): v for k, v in state_dict.items()}
        self.head.load_state_dict(new_state_dict, strict=False)


def load_mmwave_file(file_path, selected_fields=None):
    """Return xyz and features tensors from HDF5 mmWave file"""
    if selected_fields is None:
        selected_fields = ['x_cc','y_cc','vr','vr_compensated','rcs']

    with h5py.File(file_path, 'r') as f:
        radar_data = f['radar_data']
        points = []
        for i in range(len(radar_data)):
            record = radar_data[i]
            point = [record[field] for field in selected_fields]
            points.append(point)
    
    points = np.array(points, dtype=np.float32)
    xyz = points[:, :2]           # (N, 2)
    features = points[:, 2:]      # (N, 5)
    
    # Add batch dim
    xyz = torch.from_numpy(xyz).unsqueeze(0)        # (1, N, 2)
    features = torch.from_numpy(features).unsqueeze(0)  # (1, N, 5)
    return xyz, features

# --- Load first record only ---
def load_first_record(file_path, selected_fields=None):
    if selected_fields is None:
        selected_fields = ['x_cc','y_cc','vr','vr_compensated','rcs']
    
    with h5py.File(file_path, 'r') as f:
        radar_data = f['radar_data']
        record = radar_data[0]  # first record
        points = [record[field] for field in selected_fields]
        points = np.array(points, dtype=np.float32)
        xyz = points[:2].reshape(1, 1, 2)  # (1, 1, 2)
        features = points[2:].reshape(1, 1, -1)  # (1, 1, 5)
        
        # Ground truth
        gt_label = int(record['label_id']) if 'label_id' in record.dtype.names else None
        gt_center = record['center'] if 'center' in record.dtype.names else None
        gt_velocity = record['velocity'] if 'velocity' in record.dtype.names else None
        
    return torch.from_numpy(xyz), torch.from_numpy(features), gt_label, gt_center, gt_velocity

# if __name__ == "__main__":
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = FinalRadarModel(backbone_feature_channels=5, num_classes=11).to(DEVICE)
#     model.load_pretrained(
#         backbone_path="checkpoints/pointnetpp_backbone.pth",
#         head_path="checkpoints/votenet_head.pth",
#         device=DEVICE
#     )
#     model.eval()

#     file_path = "RadarScenes/data/sequence_4/radar_data.h5"
#     xyz, features, gt_label, gt_center, gt_velocity = load_first_record(file_path)
#     xyz, features = xyz.to(DEVICE), features.to(DEVICE)

#     with torch.no_grad():
#         predictions, aux = model(xyz, features)

#     # --- Predicted labels ---
#     pred_obj = int(np.argmax(predictions["classification"]["objectness"][0]))
#     pred_semantic = int(np.argmax(predictions["classification"]["semantic"][0]))
#     pred_center = predictions["bbox"]["center"][0].tolist()
#     size_class = int(np.argmax(predictions["bbox"]["size_scores"][0]))
#     pred_size = predictions["bbox"]["size_residuals"][0][size_class].tolist()
#     heading_class = int(np.argmax(predictions["bbox"]["heading_scores"][0]))
#     pred_heading = float(predictions["bbox"]["heading_residuals"][0][heading_class])
#     pred_velocity = predictions["velocity"][0].tolist()

#     # --- Print comparison ---
#     comparison = {
#         "semantic_label": {"predicted": pred_semantic, "actual": gt_label},
#         "center": {"predicted": pred_center, "actual": gt_center.tolist() if gt_center is not None else None},
#         "size_residual": {"predicted": pred_size},
#         "heading": {"predicted": pred_heading},
#         "velocity": {"predicted": pred_velocity, "actual": gt_velocity.tolist() if gt_velocity is not None else None},
#         "objectness": {"predicted": pred_obj}
#     }

#     print(json.dumps(comparison, indent=2))

def run_model_on_file(file_path, model, device, selected_fields=None, max_records=50):
    if selected_fields is None:
        selected_fields = ['x_cc','y_cc','vr','vr_compensated','rcs']

    results = []

    with h5py.File(file_path, 'r') as f:
        radar_data = f['radar_data']
        num_records = min(len(radar_data), max_records)  # limit to first max_records
        for i in range(num_records):
            record = radar_data[i]

            # Prepare point cloud
            points = [record[field] for field in selected_fields]
            points = np.array(points, dtype=np.float32)
            xyz = torch.from_numpy(points[:2]).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 2)
            features = torch.from_numpy(points[2:]).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 5)

            # Run model
            with torch.no_grad():
                predictions, aux = model(xyz, features)

            # Parse predictions
            pred_obj = int(np.argmax(predictions["classification"]["objectness"][0]))
            pred_semantic = int(np.argmax(predictions["classification"]["semantic"][0]))
            pred_center = predictions["bbox"]["center"][0].tolist()
            size_class = int(np.argmax(predictions["bbox"]["size_scores"][0]))
            pred_size = predictions["bbox"]["size_residuals"][0][size_class].tolist()
            heading_class = int(np.argmax(predictions["bbox"]["heading_scores"][0]))
            pred_heading = float(predictions["bbox"]["heading_residuals"][0][heading_class])
            pred_velocity = predictions["velocity"][0].tolist()

            # Ground truth
            gt_label = int(record['label_id']) if 'label_id' in record.dtype.names else None
            gt_center = record['center'] if 'center' in record.dtype.names else None
            gt_velocity = record['velocity'] if 'velocity' in record.dtype.names else None

            results.append({
                "semantic_label": {"predicted": pred_semantic, "actual": gt_label},
                "center": {"predicted": pred_center, "actual": gt_center.tolist() if gt_center is not None else None},
                "size_residual": {"predicted": pred_size},
                "heading": {"predicted": pred_heading},
                "velocity": {"predicted": pred_velocity, "actual": gt_velocity.tolist() if gt_velocity is not None else None},
                "objectness": {"predicted": pred_obj}
            })

    return results


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalRadarModel(backbone_feature_channels=3, num_classes=11).to(DEVICE)

    # Clean load, handles remapping internally
    model.load_pretrained(
        backbone_path="checkpoints/pointnetpp_backbone.pth",
        head_path="checkpoints/votenet_head.pth",
        device=DEVICE
    )

    model.eval()


    file_path = "RadarScenes/data/sequence_148/radar_data.h5"
    all_results = run_model_on_file(file_path, model, DEVICE, max_records=50)
    output_file = "first_50_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_file}")

    print(json.dumps(all_results, indent=2))
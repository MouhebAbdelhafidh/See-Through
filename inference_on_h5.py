import h5py
import torch
import numpy as np
from final_model import FinalModel

def load_first_frame_from_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        radar_data = f['radar_data']
        timestamps = radar_data['timestamp']
        first_ts = timestamps[0]
        mask = timestamps == first_ts
        frame_points = radar_data[mask]

        xyz = np.stack([frame_points['x_cc'], frame_points['y_cc']], axis=1)
        xyz = np.expand_dims(xyz, axis=0)

        feats_list = ['vr', 'rcs', 'vr_compensated', 'range_sc', 'azimuth_sc']
        features = np.stack([frame_points[f] for f in feats_list], axis=1)
        features = np.expand_dims(features, axis=0)

        actual_labels = frame_points['label_id']

    return xyz.astype(np.float32), features.astype(np.float32), actual_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FinalModel(backbone_feature_channels=5, head_in_dim=1024, num_classes=11).to(device)

    print("Loading backbone weights...")
    model.backbone.load_state_dict(torch.load("checkpoints/pointnetpp_backbone.pth", map_location=device))
    print("Loading head weights...")
    model.head.load_state_dict(torch.load("checkpoints/votenet_head.pth", map_location=device), strict=False)

    model.eval()

    h5_path = "RadarScenes/data/sequence_1/radar_data.h5"
    xyz, features, actual_labels = load_first_frame_from_h5(h5_path)
    xyz_t = torch.from_numpy(xyz).to(device)
    features_t = torch.from_numpy(features).to(device)

    with torch.no_grad():
        outputs, aux = model(xyz_t, features_t)

    # Predicted semantic class ID (argmax of semantic logits)
    pred_semantic_logits = outputs["classification"]["semantic"].cpu().numpy()  # shape (1, num_classes)
    pred_label_id = np.argmax(pred_semantic_logits[0])

    # Predicted velocity (vx, vy)
    pred_velocity = outputs["velocity"].cpu().numpy()[0]

    # Print predicted label and actual label IDs (unique)
    print(f"Predicted label_id: {pred_label_id}")
    print(f"Actual label_ids (unique in frame): {np.unique(actual_labels)}")

    # Predicted velocity
    print(f"Predicted velocity (vx, vy): {pred_velocity}")

    # Estimate actual velocity (vx, vy) from features (vr and azimuth_sc)
    actual_vr = features[0, :, 0]  # 'vr' 
    actual_azimuth = features[0, :, 4]  # 'azimuth_sc' 
    actual_vx = actual_vr * np.cos(actual_azimuth)
    actual_vy = actual_vr * np.sin(actual_azimuth)
    avg_actual_velocity = (np.mean(actual_vx), np.mean(actual_vy))
    print(f"Estimated average actual velocity (vx, vy): {avg_actual_velocity}")

    print("\nSample of actual labels and velocities for first 10 points:")
    for i in range(min(10, len(actual_labels))):
        print(f"Point {i}: Actual label={actual_labels[i]}, vr={actual_vr[i]:.2f}, azimuth={actual_azimuth[i]:.2f}")

if __name__ == "__main__":
    main()

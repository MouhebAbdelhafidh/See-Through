# clissify.py
import os
import csv
import torch
import numpy as np
import h5py

# Import backbone and head (adjust import if your module path is different)
from extract_features import PointNet2Backbone
from votenet_head import VoteNetHead

# --- Config ---
DATA_PATH = "FusedData/sequence_2_fused.h5"  # path to your HDF5
possible_label_fields = ['label_id', 'label', 'class_id', 'class', 'gt_label']

# Map model class index -> dataset label id (edit to match your dataset IDs)
# Example for 10 classes (replace RHS values with your dataset's real label IDs if different)
index_to_label_id = {i: i for i in range(10)}
num_classes = len(index_to_label_id)

# Save CSV of results? Set to True to save predictions
SAVE_CSV = True
CSV_OUT = "predictions_vs_actual.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_full_fused_h5(path):
    """
    Load the entire fused_detections dataset and return:
      - xyz: (1, N, 2)
      - feats: (1, N, 3)
      - labels: (N,) or None
      - names: original record field names
      - dtype_names: dataset dtype names
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with h5py.File(path, 'r') as f:
        if 'fused_detections' not in f:
            raise RuntimeError("No 'fused_detections' dataset found in HDF5.")
        fused = f['fused_detections']
        N = len(fused)
        # detect label field if any
        label_field = None
        for lf in possible_label_fields:
            if lf in fused.dtype.names:
                label_field = lf
                break

        pts = np.zeros((N, 2), dtype=np.float32)
        feats = np.zeros((N, 3), dtype=np.float32)
        labels = None if label_field is None else np.zeros((N,), dtype=np.int64)

        for i in range(N):
            rec = fused[i]
            try:
                x = float(rec['x_cc']); y = float(rec['y_cc'])
            except Exception as e:
                raise RuntimeError("Records do not contain 'x_cc'/'y_cc' fields or values cannot be read.") from e
            # optional fields: vr, vr_compensated, rcs
            vr = float(rec['vr']) if 'vr' in fused.dtype.names else 0.0
            vrc = float(rec['vr_compensated']) if 'vr_compensated' in fused.dtype.names else (vr if vr is not None else 0.0)
            rcs = float(rec['rcs']) if 'rcs' in fused.dtype.names else 0.0

            pts[i, 0] = x
            pts[i, 1] = y
            feats[i, :] = [vr, vrc, rcs]
            if label_field:
                labels[i] = int(rec[label_field])

        # add batch dim
        xyz = pts[None, :, :]     # (1, N, 2)
        features = feats[None, :, :]  # (1, N, 3)
        return xyz, features, labels, label_field


def main():
    # 1) Load data (full file -> one big point cloud)
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}. Exiting.")
        return

    print(f"Loading HDF5: {DATA_PATH}")
    xyz_np, feat_np, labels_np, label_field = load_full_fused_h5(DATA_PATH)
    has_gt = labels_np is not None
    if has_gt:
        print(f"Found per-detection GT label field: '{label_field}'")

    # 2) convert to tensors and to device
    xyz = torch.from_numpy(xyz_np).to(device)          # (1, N, 2)
    features = torch.from_numpy(feat_np).to(device)    # (1, N, 3)
    if has_gt:
        labels = torch.from_numpy(labels_np).to(device)  # (N,)
    else:
        labels = None

    # 3) instantiate backbone and head
    backbone = PointNet2Backbone(feature_channels=features.shape[-1]).to(device)
    backbone.eval()

    # we will re-create head after we know seed feature dim
    # create a temporary head with seed_feat_dim=128; will be replaced after backbone run if needed
    tmp_seed_dim = 128
    head = VoteNetHead(seed_feat_dim=tmp_seed_dim, vote_feat_dim=128, vote_factor=1,
                       nproposals=64, agg_feat_dim=256, num_classes=num_classes).to(device)
    head.eval()

    # 4) forward pass backbone once over the entire point cloud
    with torch.no_grad():
        l3_feats, l1_feats, aux = backbone(xyz, features)

    l1_xyz = aux['l1_xyz'].to(device)           # (1, M, 2)
    l1_features = aux['l1_features'].to(device) # (1, M, C)

    # rebuild head if seed_feat_dim mismatch
    seed_feat_dim = l1_features.shape[-1]
    if seed_feat_dim != tmp_seed_dim:
        head = VoteNetHead(seed_feat_dim=seed_feat_dim, vote_feat_dim=128, vote_factor=1,
                           nproposals=64, agg_feat_dim=256, num_classes=num_classes).to(device)
        head.eval()

    # 5) forward pass VoteNet head
    with torch.no_grad():
        proposals_xyz, obj_logits, class_logits = head(l1_xyz, l1_features)

    # post-process predictions
    obj_probs = torch.sigmoid(obj_logits).cpu().numpy()[0]           # (P,)
    class_probs = torch.softmax(class_logits, dim=-1).cpu().numpy()[0]  # (P, num_classes)
    pred_class_indices = np.argmax(class_probs, axis=-1)               # (P,)
    # map indices -> label ids
    mapped_label_ids = np.array([index_to_label_id.get(int(i), -1) for i in pred_class_indices])  # (P,)

    proposals = proposals_xyz.cpu().numpy()[0]  # (P, 2)
    orig_pts = xyz.cpu().numpy()[0]             # (N, 2)
    N = orig_pts.shape[0]
    P = proposals.shape[0]

    # 6) For each original record, find nearest proposal and assign predicted label
    # compute pairwise distances: (N, P)
    # To save memory, compute in chunks if N large
    CHUNK = 4096
    predicted_for_record = np.zeros((N,), dtype=np.int64)
    predicted_prob_for_record = np.zeros((N,), dtype=float)

    # precompute proposal array for efficient broadcasting
    proposals_tensor = torch.from_numpy(proposals).to(device)  # (P,2)
    orig_tensor = torch.from_numpy(orig_pts).to(device)        # (N,2)

    # compute distances (N,P) via batching over orig points
    start = 0
    while start < N:
        end = min(N, start + CHUNK)
        chunk_pts = orig_tensor[start:end].unsqueeze(1)  # (chunk,1,2)
        # cdist between chunk and proposals: (chunk, P)
        dists = torch.cdist(chunk_pts, proposals_tensor.unsqueeze(0)).squeeze(1).cpu().numpy()  # (chunk,P)
        nearest_idx = np.argmin(dists, axis=1)  # (chunk,)
        # assign predicted label id and objectness prob
        for i, prop_idx in enumerate(nearest_idx):
            predicted_for_record[start + i] = int(mapped_label_ids[prop_idx])
            predicted_prob_for_record[start + i] = float(obj_probs[prop_idx])
        start = end

    # 7) Print and optionally save results
    print(f"\nTotal records (N): {N}, proposals (P): {P}\n")
    header = ["record_index", "x_cc", "y_cc", "pred_label_id", "pred_obj_prob"]
    if has_gt:
        header.append("actual_label_id")
    rows = []

    # reload H5 to read per-record fields for printing (to avoid re-parsing above)
    with h5py.File(DATA_PATH, 'r') as f:
        fused = f['fused_detections']
        for i in range(N):
            rec = fused[i]
            x = float(rec['x_cc']); y = float(rec['y_cc'])
            pred_label = int(predicted_for_record[i])
            pred_prob = float(predicted_prob_for_record[i])
            if has_gt:
                actual = int(rec[label_field])
                print(f"Record {i:06d}: (x={x:.3f}, y={y:.3f}) -> predicted_label_id={pred_label} (obj_prob={pred_prob:.3f}), actual_label_id={actual}")
                rows.append([i, x, y, pred_label, pred_prob, actual])
            else:
                print(f"Record {i:06d}: (x={x:.3f}, y={y:.3f}) -> predicted_label_id={pred_label} (obj_prob={pred_prob:.3f})")
                rows.append([i, x, y, pred_label, pred_prob])

    if SAVE_CSV:
        print(f"\nSaving results to {CSV_OUT}")
        with open(CSV_OUT, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
        print("Saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()

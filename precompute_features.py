# precompute_features.py
import os
import glob
import h5py
import numpy as np
import torch
from tqdm import tqdm

# Import your backbone
from extract_features import PointNet2Backbone

# ------------------ Config ------------------
DATA_DIR = "FusedData"
H5_PATTERN = os.path.join(DATA_DIR, "*_fused.h5")
OUT_PATH = "precomputed_data.npz"
LABEL_FIELD_NAMES = ['label_id', 'label', 'class_id', 'class', 'gt_label']
SELECTED_FEAT_FIELDS = ['vr', 'vr_compensated', 'rcs']  # used if present
BATCH_SIZE = 256   # how many records to run at once through backbone
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------

def find_h5_files(pattern):
    files = glob.glob(pattern)
    files.sort()
    return files

def load_records_from_file(path):
    """
    Return list of (xyz(2,), feats(3,), label_id, (file, idx)) for the given h5 file.
    Only returns records that have a label field.
    """
    recs = []
    with h5py.File(path, 'r') as f:
        if 'fused_detections' not in f:
            return recs
        ds = f['fused_detections']
        N = len(ds)
        # find label field
        label_field = None
        for lf in LABEL_FIELD_NAMES:
            if lf in ds.dtype.names:
                label_field = lf
                break
        if label_field is None:
            # no labels in this file -> skip
            return recs

        for i in range(N):
            rec = ds[i]
            try:
                x = float(rec['x_cc']); y = float(rec['y_cc'])
            except Exception:
                continue
            # get features, handle missing fields
            vr = float(rec['vr']) if 'vr' in ds.dtype.names else 0.0
            vrc = float(rec['vr_compensated']) if 'vr_compensated' in ds.dtype.names else vr
            rcs = float(rec['rcs']) if 'rcs' in ds.dtype.names else 0.0
            label = int(rec[label_field])
            xyz = np.array([x, y], dtype=np.float32)
            feats = np.array([vr, vrc, rcs], dtype=np.float32)
            recs.append((xyz, feats, label, path, i))
    return recs

def main():
    files = find_h5_files(H5_PATTERN)
    if not files:
        raise RuntimeError(f"No files found with pattern {H5_PATTERN}")

    # collect all records metadata first (to know total count)
    all_records = []
    print("Scanning files for labeled records...")
    for fp in files:
        recs = load_records_from_file(fp)
        if recs:
            all_records.extend(recs)
    N = len(all_records)
    print(f"Total labeled records found: {N}")

    if N == 0:
        raise RuntimeError("No labeled records found in any H5 files. Check LABEL_FIELD_NAMES and your data.")

    # Create arrays to store features and labels
    # We'll extract backbone features in batches.
    # Initialize model
    backbone = PointNet2Backbone(feature_channels=3).to(DEVICE)  # features length matches (vr, vrc, rcs)
    backbone.eval()

    # Determine feature dimension by running a dummy pass
    with torch.no_grad():
        xyz_dummy = torch.zeros(1, 1, 2).to(DEVICE)
        feats_dummy = torch.zeros(1, 1, 3).to(DEVICE)
        l3, l1, aux = backbone(xyz_dummy, feats_dummy)
    feat_dim = l3.squeeze(1).shape[-1]
    print("Backbone feature dim:", feat_dim)

    features_arr = np.zeros((N, feat_dim), dtype=np.float32)
    labels_arr = np.zeros((N,), dtype=np.int64)
    meta = []  # keep (file_path, record_index) tuples

    # Process in batches
    idx = 0
    pbar = tqdm(total=N, desc="Extracting features")
    while idx < N:
        b = min(BATCH_SIZE, N - idx)
        # prepare batch tensors shaped (B, N_points=1, D) and (B, 1, feat_dim)
        xyz_batch = np.zeros((b, 1, 2), dtype=np.float32)
        feats_batch = np.zeros((b, 1, 3), dtype=np.float32)
        labels_batch = np.zeros((b,), dtype=np.int64)
        batch_meta = []
        for j in range(b):
            xyz_j, feats_j, label_j, file_j, rec_j = all_records[idx + j]
            xyz_batch[j, 0, :] = xyz_j
            feats_batch[j, 0, :] = feats_j
            labels_batch[j] = label_j
            batch_meta.append((file_j, rec_j))

        # to tensors
        t_xyz = torch.from_numpy(xyz_batch).to(DEVICE)      # (B,1,2)
        t_feats = torch.from_numpy(feats_batch).to(DEVICE)  # (B,1,3)

        with torch.no_grad():
            l3_out, l1_out, aux = backbone(t_xyz, t_feats)   # l3_out: (B,1,feat_dim)
            global_feats = l3_out.squeeze(1).cpu().numpy()   # (B, feat_dim)

        features_arr[idx:idx+b, :] = global_feats
        labels_arr[idx:idx+b] = labels_batch
        meta.extend(batch_meta)

        idx += b
        pbar.update(b)
    pbar.close()

    # Save to .npz
    np.savez_compressed(OUT_PATH, features=features_arr, labels=labels_arr, meta=np.array(meta, dtype=object))
    print(f"Saved precomputed features and labels to {OUT_PATH}. N={N}, feat_dim={feat_dim}")

if __name__ == "__main__":
    main()

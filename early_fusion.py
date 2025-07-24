#!/usr/bin/env python
import argparse
import h5py
import numpy as np

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[WARN] scipy not found; falling back to slower O(n^2) clustering.")

############################
# -------- Utils ----------
############################

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def majority_vote(vals):
    if len(vals) == 0:
        return -1
    uniq, counts = np.unique(vals, return_counts=True)
    return uniq[np.argmax(counts)]

def build_weights(sensor_ids, user_weights=None):
    """
    Returns an array of weight per detection (same length as sensor_ids).
    user_weights is a dict {sensor_id:int -> weight:float}
    If not provided, uniform weights = 1.0
    """
    if user_weights is None:
        return np.ones_like(sensor_ids, dtype=float)
    w = np.ones_like(sensor_ids, dtype=float)
    for sid, weight in user_weights.items():
        w[sensor_ids == sid] = weight
    return w

def cluster_points_xy(points_xy, radius):
    """
    Cluster 2D points with a union-find on pairs within radius.
    Returns an array 'cluster_id' with same length, cluster ids in [0..num_clusters-1].
    """
    n = points_xy.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    uf = UnionFind(n)
    if HAVE_SCIPY:
        tree = cKDTree(points_xy)
        pairs = tree.query_pairs(r=radius)
        for i, j in pairs:
            uf.union(i, j)
    else:
        # O(n^2) fallback
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(points_xy[i] - points_xy[j]) <= radius:
                    uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(n)])
    # reindex to 0..k-1
    _, remapped = np.unique(roots, return_inverse=True)
    return remapped

def fuse_cluster(indices, det, weights):
    """
    Fuse a single cluster (indices of detections).
    Weighted average for continuous fields, majority vote for categorical.
    """
    idx = indices
    w = weights[idx]
    wsum = np.sum(w)

    # Weighted average x, y, vr, vr_compensated, rcs if available
    def wavg(field):
        return np.sum(det[field][idx] * w) / wsum

    fused = {}
    fused["x_cc"] = wavg("x_cc")
    fused["y_cc"] = wavg("y_cc")

    if "vr" in det.dtype.names:
        fused["vr"] = wavg("vr")
    if "vr_compensated" in det.dtype.names:
        fused["vr_compensated"] = wavg("vr_compensated")
    if "rcs" in det.dtype.names:
        fused["rcs"] = wavg("rcs")

    # label_id -> majority vote (if exists)
    if "label_id" in det.dtype.names:
        fused["label_id"] = majority_vote(det["label_id"][idx])
    # track_id -> you can also majority vote or set -1
    if "track_id" in det.dtype.names:
        fused["track_id"] = majority_vote(det["track_id"][idx])

    # Keep frame_id / timestamp if they exist (all should match in a cluster)
    for meta in ["frame_id", "timestamp"]:
        if meta in det.dtype.names:
            fused[meta] = det[meta][idx][0]

    # Optionally annotate how many raw points merged and which sensors
    fused["num_merged"] = len(idx)
    if "sensor_id" in det.dtype.names:
        fused["sensor_id"] = 255  # 255 == "fused" (change as you like)

    return fused

def fuse_frame(det_frame, radius, sensor_weights_map=None):
    """Fuse detections within a single frame."""
    if len(det_frame) == 0:
        return det_frame

    pts = np.stack([det_frame["x_cc"], det_frame["y_cc"]], axis=1)
    weights = build_weights(det_frame["sensor_id"], sensor_weights_map) \
              if "sensor_id" in det_frame.dtype.names else np.ones(len(det_frame))

    cluster_ids = cluster_points_xy(pts, radius)

    fused_rows = []
    for c in np.unique(cluster_ids):
        idx = np.where(cluster_ids == c)[0]
        fused_rows.append(fuse_cluster(idx, det_frame, weights))

    # Build a new structured dtype for fused output: keep only what we produced
    fields = [("x_cc", np.float32), ("y_cc", np.float32)]
    if "vr" in det_frame.dtype.names: fields.append(("vr", np.float32))
    if "vr_compensated" in det_frame.dtype.names: fields.append(("vr_compensated", np.float32))
    if "rcs" in det_frame.dtype.names: fields.append(("rcs", np.float32))
    if "label_id" in det_frame.dtype.names: fields.append(("label_id", np.uint8))
    if "track_id" in det_frame.dtype.names: fields.append(("track_id", np.int32))
    if "frame_id" in det_frame.dtype.names: fields.append(("frame_id", np.int32))
    if "timestamp" in det_frame.dtype.names: fields.append(("timestamp", np.float64))
    fields.append(("num_merged", np.int32))
    if "sensor_id" in det_frame.dtype.names: fields.append(("sensor_id", np.uint8))

    out = np.zeros(len(fused_rows), dtype=fields)
    for i, row in enumerate(fused_rows):
        for k in row:
            out[k][i] = row[k]
    return out

def detect_frame_key(detections):
    """Try to auto-detect a frame key."""
    for key in ("frame_id", "timestamp"):
        if key in detections.dtype.names:
            return key
    return None

def group_by_frame(detections, frame_key):
    """Yield (frame_value, indices)."""
    if frame_key is None:
        yield None, np.arange(len(detections))
        return
    values = detections[frame_key]
    uniq = np.unique(values)
    for v in uniq:
        idx = np.where(values == v)[0]
        yield v, idx

############################
# -------- Main -----------
############################

def main():
    parser = argparse.ArgumentParser(description="Early fuse multi-sensor detections from an HDF5 file.")
    parser.add_argument("--in", dest="in_file", required=True, help="Input .h5 file")
    parser.add_argument("--out", dest="out_file", required=True, help="Output .h5 file")
    parser.add_argument("--dataset", default="detections", help="Dataset name inside the .h5 (default: detections)")
    parser.add_argument("--radius", type=float, default=0.5, help="Clustering radius in meters (default: 0.5)")
    parser.add_argument("--weights", type=str, default="", 
                        help='Per-sensor weights, e.g. "0:1.0,1:0.8,2:0.5" (default: uniform)')
    args = parser.parse_args()

    # Parse weights map
    sensor_weights_map = None
    if args.weights:
        sensor_weights_map = {}
        for pair in args.weights.split(","):
            sid, w = pair.split(":")
            sensor_weights_map[int(sid)] = float(w)

    with h5py.File(args.in_file, "r") as f:
        det = f[args.dataset][:]
        print(f"Loaded {len(det)} detections from {args.dataset}")

    frame_key = detect_frame_key(det)
    if frame_key:
        print(f"Detected frame key: {frame_key}")
    else:
        print("No frame key found (frame_id/timestamp). Treating all detections as one frame.")

    fused_all = []
    for frame_val, idx in group_by_frame(det, frame_key):
        det_frame = det[idx]
        fused = fuse_frame(det_frame, radius=args.radius, sensor_weights_map=sensor_weights_map)
        fused_all.append(fused)

    fused_all = np.concatenate(fused_all) if fused_all else np.empty(0, dtype=det.dtype)
    print(f"Fused detections: {len(fused_all)} (down from {len(det)})")

    with h5py.File(args.out_file, "w") as f:
        f.create_dataset("fused_detections", data=fused_all, compression="gzip")
    print(f"Saved to {args.out_file}:/fused_detections")

if __name__ == "__main__":
    main()

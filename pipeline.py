#!/usr/bin/env python
import argparse
import h5py
import numpy as np
import torch
import cv2
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import torch.nn as nn
import torch.nn.functional as F
from final_model import FinalRadarModel

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[WARNING] scipy not found - falling back to slower clustering method")

# ------------------ Constants ------------------ #
SCALE_M_TO_PX = 20.0
CENTER_PX = np.array([400.0, 400.0], dtype=np.float32)
MAX_SKIPPED_FRAMES = 5
DIST_THRESHOLD_PX = 60.0
VR_MOVING_THRESH = 0.5
NUM_CLASSES = 11  # Number of classes (0-10)

CLASS_COLORS = [
    (255, 0, 0),    # 0: red
    (255, 165, 0),  # 1: orange
    (139, 0, 139),  # 2: dark magenta
    (0, 0, 255),    # 3: blue
    (0, 255, 255),  # 4: cyan
    (0, 255, 0),    # 5: green
    (255, 255, 0),  # 6: yellow
    (255, 192, 203),# 7: pink
    (165, 42, 42),  # 8: brown
    (0, 128, 0),    # 9: dark green
    (128, 128, 128),# 10: gray
    (255, 255, 255) # 11: white (default)
]

# ------------------ Fusion Functions ------------------ #
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
        if ra == rb: 
            return
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
    if user_weights is None:
        return np.ones_like(sensor_ids, dtype=float)
    w = np.ones_like(sensor_ids, dtype=float)
    for sid, weight in user_weights.items():
        w[sensor_ids == sid] = weight
    return w

def cluster_points_xy(points_xy, radius):
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
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(points_xy[i] - points_xy[j]) <= radius:
                    uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(n)])
    _, remapped = np.unique(roots, return_inverse=True)
    return remapped

def fuse_cluster(indices, det, weights):
    idx = indices
    w = weights[idx]
    wsum = np.sum(w)

    def wavg(field):
        return np.sum(det[field][idx] * w) / wsum

    fused = {
        "x_cc": wavg("x_cc"),
        "y_cc": wavg("y_cc"),
        "num_merged": len(idx),
        "sensor_id": 255  # Mark as fused
    }

    for field in ["vr", "vr_compensated", "rcs"]:
        if field in det.dtype.names:
            fused[field] = wavg(field)

    for field in ["label_id", "track_id"]:
        if field in det.dtype.names:
            fused[field] = majority_vote(det[field][idx])

    for meta in ["frame_id", "timestamp"]:
        if meta in det.dtype.names:
            fused[meta] = det[meta][idx][0]

    return fused

def fuse_frame(det_frame, radius, sensor_weights_map=None):
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

    # Build output dtype
    fields = [
        ("x_cc", np.float32), ("y_cc", np.float32),
        ("num_merged", np.int32), ("sensor_id", np.uint8)
    ]
    for field in ["vr", "vr_compensated", "rcs"]:
        if field in det_frame.dtype.names:
            fields.append((field, np.float32))
    for field in ["label_id", "track_id"]:
        if field in det_frame.dtype.names:
            fields.append((field, np.uint8 if field == "label_id" else np.int32))
    for field in ["frame_id", "timestamp"]:
        if field in det_frame.dtype.names:
            fields.append((field, np.int32 if field == "frame_id" else np.float64))

    out = np.zeros(len(fused_rows), dtype=fields)
    for i, row in enumerate(fused_rows):
        for k in row:
            out[k][i] = row[k]
    return out

def detect_frame_key(detections):
    for key in ("frame_id", "timestamp"):
        if key in detections.dtype.names:
            return key
    return None

def group_by_frame(detections, frame_key):
    if frame_key is None:
        yield None, np.arange(len(detections))
        return
    values = detections[frame_key]
    uniq = np.unique(values)
    for v in uniq:
        idx = np.where(values == v)[0]
        yield v, idx

# ------------------ Tracking ------------------ #
class Track:
    def __init__(self, detection_xy_px, track_id, class_id=None):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]], dtype=np.float32)
        self.kf.R *= 1.0
        self.kf.P *= 50.0
        self.kf.Q *= 0.01

        self.kf.x[:2] = np.asarray(detection_xy_px, dtype=np.float32).reshape(2, 1)
        self.track_id = track_id
        self.class_id = class_id if class_id is not None else -1
        self.skipped_frames = 0
        self.hits = 1
        self.age = 0

    @property
    def pos(self):
        return self.kf.x[:2].reshape(-1)

    def predict(self):
        self.kf.predict()
        self.age += 1
        return self.pos

    def update(self, detection_xy_px, class_id=None):
        self.kf.update(np.asarray(detection_xy_px, dtype=np.float32))
        self.skipped_frames = 0
        self.hits += 1
        if class_id is not None:
            self.class_id = int(class_id)

class Tracker:
    def __init__(self, max_skipped_frames=MAX_SKIPPED_FRAMES, dist_threshold_px=DIST_THRESHOLD_PX):
        self.max_skipped_frames = max_skipped_frames
        self.dist_threshold_px = dist_threshold_px
        self.tracks = []
        self._next_id = 0

    def _make_cost(self, detections_px):
        N, M = len(self.tracks), len(detections_px)
        cost = np.zeros((N, M), dtype=np.float32)
        for i, trk in enumerate(self.tracks):
            trk_xy = trk.pos
            cost[i, :] = np.linalg.norm(detections_px - trk_xy, axis=1)
        return cost

    def update(self, detections_px, pred_labels):
        # Predict existing tracks
        for trk in self.tracks:
            trk.predict()

        if len(detections_px) == 0:
            self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped_frames]
            return

        if len(self.tracks) == 0:
            for det, p in zip(detections_px, pred_labels):
                self.tracks.append(Track(det, self._next_id, p))
                self._next_id += 1
            return

        cost = self._make_cost(detections_px)
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.dist_threshold_px:
                self.tracks[r].update(detections_px[c], pred_labels[c])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Handle unassigned tracks
        for i, trk in enumerate(self.tracks):
            if i not in assigned_tracks:
                trk.skipped_frames += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped_frames]

        # Create new tracks for unassigned detections
        for j, det in enumerate(detections_px):
            if j not in assigned_dets:
                class_id = pred_labels[j] if j < len(pred_labels) else -1
                self.tracks.append(Track(det, self._next_id, class_id))
                self._next_id += 1

# ------------------ Utility Functions ------------------ #
def world_to_image(xy_m):
    return (xy_m.astype(np.float32) * SCALE_M_TO_PX) + CENTER_PX

def compute_features(fused_data):
    """Compute features from fused detections"""
    xyz = np.stack([fused_data['x_cc'], fused_data['y_cc'], np.zeros(len(fused_data))], axis=1)
    ranges = np.sqrt(fused_data['x_cc']**2 + fused_data['y_cc']**2)
    azimuths = np.arctan2(fused_data['y_cc'], fused_data['x_cc'])
    
    features = np.stack([
        fused_data['vr_compensated'],
        fused_data['rcs'],
        ranges,
        azimuths
    ], axis=1)
    
    return xyz, features

def predict_labels_for_frame(frame_data, model, device):
    """
    frame_data: structured numpy array containing x_cc, y_cc, vr_compensated, rcs, etc.
    Returns: pred_labels (numpy array, N)
    """
    if len(frame_data) == 0:
        return np.array([], dtype=np.int32)

    # Convert to tensors
    xyz = np.stack([frame_data['x_cc'], frame_data['y_cc']], axis=1)  # (N,2)
    features = np.stack([
        frame_data['vr_compensated'] if 'vr_compensated' in frame_data.dtype.names else np.zeros(len(frame_data)),
        frame_data['rcs'] if 'rcs' in frame_data.dtype.names else np.zeros(len(frame_data)),
        np.sqrt(frame_data['x_cc']**2 + frame_data['y_cc']**2),
        np.arctan2(frame_data['y_cc'], frame_data['x_cc']),
        np.zeros(len(frame_data))  # placeholder for 5th channel if needed
    ], axis=1)

    xyz_tensor = torch.from_numpy(xyz).float().unsqueeze(0).to(device)       # [1, N, 2]
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)  # [1, N, 5]

    with torch.no_grad():
        outputs, _ = model(xyz_tensor, features_tensor)
    
    # Choose semantic labels as predicted label_id
    semantic_scores = outputs["classification"]["semantic"][0]  # [N, num_classes]
    pred_labels = torch.argmax(semantic_scores, dim=-1).cpu().numpy()  # [N]

    return pred_labels

# ------------------ Main Pipeline ------------------ #
import argparse
import h5py
import cv2
import numpy as np
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Processed H5 file with radar data")
    parser.add_argument("--output_video", type=str, default="tracking_output.mp4", help="Output MP4 file")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--size", type=int, default=800, help="Video size (square)")
    args = parser.parse_args()

    # Load H5 radar data
    with h5py.File(args.input, "r") as f:
        if "radar_data" in f:
            data = f["radar_data"][:]
        else:
            raise ValueError("H5 file must contain 'radar_data' dataset.")

    # Determine frame grouping key
    frame_key = "frame_id" if "frame_id" in data.dtype.names else "timestamp"
    tracker = Tracker()

    # Setup video writer
    video_out = cv2.VideoWriter(
        args.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.size, args.size),
    )

    # Load model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalRadarModel().to(DEVICE)
    model.load_pretrained(
        backbone_path="checkpoints/pointnetpp_backbone.pth",
        head_path="checkpoints/votenet_head.pth",
        device=DEVICE
    )
    model.eval()

    # Iterate over frames
    for frame_val, idxs in group_by_frame(data, frame_key):
        frame_data = data[idxs]
        if len(frame_data) == 0:
            continue

        # Convert world coordinates to pixels
        xy_m = np.stack([frame_data["x_cc"], frame_data["y_cc"]], axis=1)
        xy_px = world_to_image(xy_m)

        # Compute 3D coordinates and features
        xyz, features = compute_features(frame_data)
        xyz_tensor = torch.from_numpy(xyz).float().unsqueeze(0).to(DEVICE)       # [1, N, 3]
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)  # [1, N, 4]

        # Model inference
        with torch.no_grad():
            outputs, _ = model(xyz_tensor, features_tensor)

        # Semantic label prediction
        semantic_scores = outputs["classification"]["semantic"][0]  # [N, num_classes]
        pred_labels = torch.argmax(semantic_scores, dim=-1).cpu().numpy()

        # Velocities for visualization
        velocities = frame_data["vr_compensated"] if "vr_compensated" in frame_data.dtype.names else np.zeros(len(xy_px))

        # Update tracker
        tracker.update(xy_px, pred_labels)

        # Draw frame
        frame_img = np.zeros((args.size, args.size, 3), dtype=np.uint8)
        for trk in tracker.tracks:
            cx, cy = int(trk.pos[0]), int(trk.pos[1])
            tid = trk.track_id
            cls = trk.class_id
            # Closest detection to show velocity
            dists = np.linalg.norm(xy_px - trk.pos, axis=1)
            v = velocities[np.argmin(dists)] if len(dists) > 0 else 0.0

            # Draw circle and labels
            color = CLASS_COLORS[cls] if 0 <= cls < len(CLASS_COLORS) else (255, 255, 255)
            cv2.circle(frame_img, (cx, cy), 5, color, -1)
            cv2.putText(frame_img, f"ID:{tid} L:{cls} v:{v:.1f}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        video_out.write(frame_img)

    video_out.release()
    print(f"[INFO] Tracking video saved to {args.output_video}")


if __name__ == "__main__":
    main()


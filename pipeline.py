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

# ------------------ Model Definition ------------------ #
class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 64)  # 3 (xyz) + 4 (features)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        
    def forward(self, xyz, features):
        x = torch.cat([xyz, features], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        global_feat = torch.max(x, dim=1)[0]  # Global max pooling
        return global_feat, None, None

class SimpleHead(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        return {
            "classification": {
                "semantic": self.fc(x).unsqueeze(1)  # [B, 1, num_classes]
            }
        }

class FinalRadarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.head = SimpleHead()

    def forward(self, xyz, point_features):
        global_feat, _, _ = self.backbone(xyz, point_features)
        return self.head(global_feat), None

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

# ------------------ Main Pipeline ------------------ #
import argparse
import h5py
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Processed H5 file with tracking results")
    parser.add_argument("--output_video", type=str, default="tracking_output.mp4", help="Output MP4 file")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--size", type=int, default=800, help="Video size (square)")
    args = parser.parse_args()

    # open file
    with h5py.File(args.input, "r") as f:
        if "radar_data" in f:
            data = f["radar_data"][:]
        else:
            raise ValueError("H5 file must contain 'radar_data' dataset.")

        # extract fields
        x = data["x_cc"]
        y = data["y_cc"]
        track_ids = data["track_id"] if "track_id" in data.dtype.names else np.full(len(x), -1)
        velocities = data["vr_compensated"] if "vr_compensated" in data.dtype.names else None
        bboxes = None  # RadarScenes doesn't have bboxes by default

    # setup video writer
    video_out = cv2.VideoWriter(
        args.output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.size, args.size),
    )

    # normalize coordinates to fit video size
    x_norm = np.interp(x, (np.min(x), np.max(x)), (0, args.size - 1)).astype(int)
    y_norm = np.interp(y, (np.min(y), np.max(y)), (0, args.size - 1)).astype(int)

    # iterate per frame
    num_frames = len(x_norm)
    for idx in range(num_frames):
        frame_img = np.zeros((args.size, args.size, 3), dtype=np.uint8)

        cx, cy = x_norm[idx], y_norm[idx]
        tid = track_ids[idx]

        # draw detection
        cv2.circle(frame_img, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame_img, str(tid), (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # optional: draw velocity as text
        if velocities is not None:
            v = velocities[idx]
            cv2.putText(frame_img, f"v:{v:.1f}", (cx + 5, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        video_out.write(frame_img)

    video_out.release()
    print(f"[INFO] Tracking video saved → {args.output_video}")


if __name__ == "__main__":
    main()

# tracker.py
import os
import numpy as np
import h5py
import cv2
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# ---------------- Config ---------------- #
H5_PATH = r"ProcessedData/sequence_1.h5"
VIDEO_OUT = "tracking_result.mp4"
FPS = 10
FRAME_SIZE = (800, 800)
SCALE_M_TO_PX = 20.0
CENTER_PX = np.array([400.0, 400.0], dtype=np.float32)
DIST_THRESHOLD_PX = 60.0
MAX_SKIPPED_FRAMES = 5
VR_MOVING_THRESH = 0.5

CLASS_COLORS = [
    (255, 0, 0), (255, 165, 0), (139, 0, 139), (0, 0, 255),
    (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 192, 203),
    (165, 42, 42), (0, 128, 0), (128, 128, 128), (255, 255, 255)
]

# ---------------- Simple Track + Tracker ---------------- #
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
        self.class_id = class_id
        self.actual_label = -1
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
            diffs = detections_px - trk_xy[None, :]
            cost[i, :] = np.sqrt(np.sum(diffs * diffs, axis=1))
        return cost

    def update(self, detections_px, pred_labels, actual_labels=None):
        if actual_labels is None:
            actual_labels = [-1] * len(detections_px)

        # Predict existing tracks
        for trk in self.tracks:
            trk.predict()

        if len(detections_px) == 0:
            for trk in self.tracks:
                trk.skipped_frames += 1
            self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped_frames]
            return

        if len(self.tracks) == 0:
            for det, p, a in zip(detections_px, pred_labels, actual_labels):
                trk = Track(det, self._next_id, p)
                trk.actual_label = a
                self.tracks.append(trk)
                self._next_id += 1
            return

        # Assignment
        cost = self._make_cost(detections_px)
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.dist_threshold_px:
                self.tracks[r].update(detections_px[c], pred_labels[c])
                self.tracks[r].actual_label = actual_labels[c]
                assigned_tracks.add(r)
                assigned_dets.add(c)

        for i, trk in enumerate(self.tracks):
            if i not in assigned_tracks:
                trk.skipped_frames += 1

        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped_frames]

        for j, det in enumerate(detections_px):
            if j not in assigned_dets:
                trk = Track(det, self._next_id, pred_labels[j])
                trk.actual_label = actual_labels[j]
                self.tracks.append(trk)
                self._next_id += 1

# ---------------- Visualization ---------------- #
def world_to_image(xy_m):
    return (xy_m.astype(np.float32) * SCALE_M_TO_PX) + CENTER_PX

def draw_tracks(frame, tracks):
    for trk in tracks:
        x, y = trk.pos.astype(np.int32)
        pred_label = int(trk.class_id) if trk.class_id is not None else -1
        actual_label = getattr(trk, "actual_label", -1)
        color = CLASS_COLORS[pred_label if 0 <= pred_label < len(CLASS_COLORS) else -1]
        cv2.circle(frame, (x, y), 5, color, -1)
        text = f"P:{pred_label} C:{actual_label}"
        cv2.putText(frame, text, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame

# ---------------- Main ---------------- #
def main():
    print(f"Opening data file: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        detections = f["detections"]
        frames = f["frames"]
        frames_sorted = sorted(frames, key=lambda r: int(r["timestamp"]))

        width, height = FRAME_SIZE
        out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))

        tracker = Tracker(max_skipped_frames=MAX_SKIPPED_FRAMES, dist_threshold_px=DIST_THRESHOLD_PX)

        for frame_info in frames_sorted: 
            ts = int(frame_info["timestamp"])
            s = int(frame_info["detection_start_idx"])
            e = int(frame_info["detection_end_idx"])
            dets = detections[s:e]

            frame = np.zeros((height, width, 3), dtype=np.uint8)

            if dets.size > 0:
                if "vr_compensated" in dets.dtype.names:
                    vr_used = dets["vr_compensated"]
                elif "vr" in dets.dtype.names:
                    vr_used = dets["vr"]
                else:
                    vr_used = np.ones(len(dets), dtype=np.float32) * VR_MOVING_THRESH
                moving_mask = np.abs(vr_used) >= VR_MOVING_THRESH
                dets = dets[moving_mask]

            if dets.size == 0:
                dets_px = np.empty((0, 2), dtype=np.float32)
                pred_labels = []
                actual_labels = []
            else:
                xy_m = np.stack([dets["x_cc"], dets["y_cc"]], axis=1).astype(np.float32)
                dets_px = world_to_image(xy_m)
                if "predicted_id" in dets.dtype.names:
                    pred_labels = dets["predicted_id"].astype(int).tolist()
                else:
                    pred_labels = dets["label_id"].astype(int).tolist()
                actual_labels = dets["label_id"].astype(int).tolist()

            tracker.update(dets_px, pred_labels, actual_labels)
            draw_tracks(frame, tracker.tracks)

            cv2.putText(frame, f"Timestamp: {ts}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Detections: {len(dets_px)}  Tracks: {len(tracker.tracks)}",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            out.write(frame)
            cv2.imshow("Radar Tracking (moving objects)", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()
        print(f"Saved: {VIDEO_OUT}")

if __name__ == "__main__":
    main()

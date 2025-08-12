import numpy as np
import h5py
import torch
import torch.nn as nn
import cv2
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# ---------------- Your VoteNetHead class ---------------- #
NUM_CLASSES = 11

class VoteNetHead(nn.Module):
    def __init__(self, in_dim=1024, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- Tracker and Track classes ---------------- #
class Track:
    def __init__(self, detection, track_id):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.R *= 0.01
        self.kf.P *= 10.
        self.kf.Q *= 0.01

        self.kf.x[:2] = np.array(detection).reshape(2, 1)

        self.track_id = track_id
        self.skipped_frames = 0
        self.prediction = detection
        self.label_id = None
        self.predicted_id = None

    def update(self, detection):
        self.skipped_frames = 0
        self.kf.update(detection)
        self.prediction = self.kf.x[:2].reshape(1, -1)[0]

    def predict(self):
        self.kf.predict()
        self.prediction = self.kf.x[:2].reshape(1, -1)[0]
        return self.prediction


class Tracker:
    def __init__(self, max_skipped_frames=5, dist_threshold=3.0):
        self.max_skipped_frames = max_skipped_frames
        self.dist_threshold = dist_threshold
        self.tracks = []
        self.track_id_count = 0

    def update(self, detections, labels, preds):
        if len(self.tracks) == 0:
            for i, det in enumerate(detections):
                trk = Track(det, self.track_id_count)
                trk.label_id = labels[i]
                trk.predicted_id = preds[i]
                self.track_id_count += 1
                self.tracks.append(trk)
            return

        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros((N, M))

        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost[i][j] = np.linalg.norm(trk.prediction - det)

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r][c] > self.dist_threshold:
                continue
            self.tracks[r].update(detections[c])
            self.tracks[r].label_id = labels[c]
            self.tracks[r].predicted_id = preds[c]
            assigned_tracks.add(r)
            assigned_dets.add(c)

        for i, trk in enumerate(self.tracks):
            if i not in assigned_tracks:
                trk.skipped_frames += 1
                trk.predict()

        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped_frames]

        for j, det in enumerate(detections):
            if j not in assigned_dets:
                trk = Track(det, self.track_id_count)
                trk.label_id = labels[j]
                trk.predicted_id = preds[j]
                self.track_id_count += 1
                self.tracks.append(trk)


# ---------------- Visualization ---------------- #
def draw_tracks(frame, tracks):
    for trk in tracks:
        x, y = int(trk.prediction[0]), int(trk.prediction[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        text = f"ID:{trk.track_id} L:{trk.label_id} P:{trk.predicted_id}"
        cv2.putText(frame, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
    return frame


# ---------------- Main ---------------- #
def main():
    h5_path = "ProcessedData/sequence_1.h5"
    classifier_path = "checkpoints/votenet_head_finetuned.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classifier only (assuming features pre-extracted or dummy features here)
    print("Loading classifier...")
    classifier = VoteNetHead(in_dim=1024, num_classes=NUM_CLASSES).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()

    tracker = Tracker(max_skipped_frames=5, dist_threshold=3.0)

    print(f"Opening data file: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        radar_data = f['radar_data']  # Structured array with fields including x_cc, y_cc, label_id, timestamp

        # Group points by timestamp for frame-wise processing
        timestamps = np.unique(radar_data['timestamp'])
        print(f"Total frames (unique timestamps): {len(timestamps)}")

        # Open video writer
        out = cv2.VideoWriter("tracking_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (800, 800))

        for ts in timestamps:
            # Select points for this timestamp
            mask = radar_data['timestamp'] == ts
            points = radar_data[mask]

            frame = np.zeros((800, 800, 3), dtype=np.uint8)

            if len(points) == 0:
                detections = np.array([])
                preds = []
                labels = []
            else:
                # Extract x_cc, y_cc for detection positions (center coordinates)
                pts_xy = np.vstack([points['x_cc'], points['y_cc']]).T

                # Scale and shift points to image coordinates
                pts_img = pts_xy * 20 + 400  # Adjust scaling to fit your frame size

                detections = pts_img.astype(np.float32)

                # Dummy features (since backbone not loaded here, just use zeros or random)
                # In your pipeline, extract features properly!
                features = torch.zeros((len(detections), 1024), device=device)  # Replace with real features!

                with torch.no_grad():
                    outputs = classifier(features)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()

                labels = points['label_id'].tolist()

            tracker.update(detections, labels, preds)
            frame = draw_tracks(frame, tracker.tracks)

            # Show timestamp on frame
            cv2.putText(frame, f"Timestamp: {ts}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            out.write(frame)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

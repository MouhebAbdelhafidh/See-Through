import torch
import h5py
import numpy as np
import cv2
from Product.Tracking.tracker import Tracker, world_to_image, draw_tracks
from Product.final_model import FinalRadarModel, load_mmwave_file  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H5_PATH = "RadarScenes/data/sequence_148/radar_data.h5"
VIDEO_OUT = "tracking_result_with_preds.mp4"
FPS = 10
FRAME_SIZE = (800, 800)
SCALE_M_TO_PX = 20.0
CENTER_PX = np.array([400.0, 400.0], dtype=np.float32)
VR_MOVING_THRESH = 0.5

# --- Load your trained model ---
model = FinalRadarModel(backbone_feature_channels=3, num_classes=11).to(DEVICE)
model.load_pretrained(
    backbone_path="checkpoints/pointnetpp_backbone.pth",
    head_path="checkpoints/votenet_head.pth",
    device=DEVICE
)
model.eval()
def load_mmwave_file_from_record(record, selected_fields=None):
    if selected_fields is None:
        selected_fields = ['x_cc','y_cc','vr','vr_compensated','rcs']

    points = np.array([record[field] for field in selected_fields], dtype=np.float32)
    xyz = torch.from_numpy(points[:2]).unsqueeze(0).unsqueeze(0)      # (1, 1, 2)
    features = torch.from_numpy(points[2:]).unsqueeze(0).unsqueeze(0) # (1, 1, F)
    return xyz, features
# --- Tracker ---
tracker = Tracker(max_skipped_frames=5, dist_threshold_px=60.0)

with h5py.File(H5_PATH, "r") as f:
    radar_data = f["radar_data"]
    width, height = FRAME_SIZE
    out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    
    tracker = Tracker(max_skipped_frames=5, dist_threshold_px=60.0)
    
    for i, record in enumerate(radar_data):
        # Prepare point cloud
        xyz, features = load_mmwave_file_from_record(record)
        xyz, features = xyz.to(DEVICE), features.to(DEVICE)
        
        # Model predictions
        with torch.no_grad():
            preds, _ = model(xyz, features)

        pred_label = int(np.argmax(preds["classification"]["semantic"][0].cpu().numpy()))
        pred_velocity = preds["velocity"][0].cpu().numpy()  # array, e.g., [vx, vy]

        # Convert point positions to image coordinates
        xy_m = xyz[0, :, :2].cpu().numpy()
        dets_px = world_to_image(xy_m)

        # Update tracker
        tracker.update(dets_px, [pred_label], actual_labels=[-1])

        # Draw frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_tracks(frame, tracker.tracks)

        out.write(frame)  # save frame to video

        # Optional: show progress every 100 frames
        if i % 100 == 0:
            print(f"Processed frame {i}/{len(radar_data)}")

        # Optional: display window (can comment out for speed)
        # cv2.imshow("Radar Tracking with Predictions", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {VIDEO_OUT}")

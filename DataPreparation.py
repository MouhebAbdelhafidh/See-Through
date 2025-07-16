import os
import json
import h5py
import numpy as np
from tqdm import tqdm

# Create output directory
os.makedirs("ProcessedData", exist_ok=True)

# Define dtype for processed detections
detection_dtype = np.dtype([
    ("x_cc", "f4"),      # float32
    ("y_cc", "f4"),
    ("label_id", "u1"),  # uint8 (0-255)
    ("track_id", "i4"),  # int32
    ("sensor_id", "u1"),
    ("rcs", "f4"),
    ("vr", "f4"),
    ("vr_compensated", "f4"),
])
def process_sequence(sequence_dir, output_path):
    """Process a single RadarScenes sequence into the optimized HDF5 format."""
    with open(os.path.join(sequence_dir, "scenes.json"), "r") as f:
        scenes = json.load(f)["scenes"]
    
    with h5py.File(os.path.join(sequence_dir, "radar_data.h5"), "r") as raw_h5:
        radar_data = raw_h5["radar_data"]
        odometry = raw_h5["odometry"]
        
        # Group scenes by odometry_index
        frame_groups = {}
        for scene in scenes.values():
            odom_idx = scene["odometry_index"]
            if odom_idx not in frame_groups:
                ego_data = odometry[odom_idx]
                frame_groups[odom_idx] = {
                    "ego_vx": float(ego_data[4]),
                    "ego_yaw_rate": float(ego_data[5]),
                    "detection_indices": []
                }
            frame_groups[odom_idx]["detection_indices"].append(
                (scene["radar_indices"][0], scene["radar_indices"][1])
            )
        
        # Calculate totals
        total_frames = len(frame_groups)
        total_detections = sum(
            end - start
            for frame in frame_groups.values()
            for start, end in frame["detection_indices"]
        )
        
        with h5py.File(output_path, "w") as out_h5:
            # Create datasets
            frames_ds = out_h5.create_dataset(
                "frames",
                shape=(total_frames,),
                dtype=np.dtype([
                    ("odometry_index", "i4"),
                    ("timestamp", "i8"),
                    ("ego_velocity", "f4"),
                    ("ego_yaw_rate", "f4"),
                    ("detection_start_idx", "i4"),
                    ("detection_end_idx", "i4"),
                ])
            )
            
            detections_ds = out_h5.create_dataset(
                "detections",
                shape=(total_detections,),
                dtype=detection_dtype
            )
            
            # Store metadata
            out_h5.attrs["sequence_name"] = os.path.basename(sequence_dir)
            
            # Process detections with error handling
            detection_idx = 0
            for frame_idx, (odom_idx, frame) in enumerate(frame_groups.items()):
                odom_data = odometry[odom_idx]
                frame_detections = 0
                
                # Get all detection ranges for this frame
                ranges = []
                for start, end in frame["detection_indices"]:
                    ranges.append((start, end))
                    frame_detections += (end - start)
                
                # Write frame metadata
                frames_ds[frame_idx] = (
                    int(odom_idx),
                    int(odom_data[0]),
                    float(frame["ego_vx"]),
                    float(frame["ego_yaw_rate"]),
                    int(detection_idx),
                    int(detection_idx + frame_detections)
                )
                
                # Process and write detections
                for start, end in ranges:
                    det_slice = radar_data[start:end]
                    
                    # Convert track_id from bytes to int safely
                    track_ids = []
                    for tid in det_slice["track_id"]:
                        try:
                            track_ids.append(int(tid) if tid != b'' else -1)
                        except:
                            track_ids.append(-1)
                    
                    # Build detections array
                    detections = np.zeros((end-start,), dtype=detection_dtype)
                    detections["x_cc"] = det_slice["x_cc"]
                    detections["y_cc"] = det_slice["y_cc"]
                    detections["label_id"] = det_slice["label_id"]
                    detections["track_id"] = track_ids
                    detections["sensor_id"] = det_slice["sensor_id"]
                    detections["rcs"] = det_slice["rcs"]
                    detections["vr"] = det_slice["vr"]
                    detections["vr_compensated"] = det_slice["vr_compensated"]
                    
                    # Write to HDF5
                    detections_ds[detection_idx:detection_idx + (end-start)] = detections
                    detection_idx += (end - start)
                    
# Process first 5 sequences
base_dir = "RadarScenes/data"  # Change to your dataset path
sequences = [f"sequence_{i}" for i in range(1, 159)]

for seq in tqdm(sequences, desc="Processing sequences"):
    seq_dir = os.path.join(base_dir, seq)
    output_path = os.path.join("ProcessedData", f"{seq}.h5")
    process_sequence(seq_dir, output_path)

print(f"Successfully processed {len(sequences)} sequences to ProcessedData/")




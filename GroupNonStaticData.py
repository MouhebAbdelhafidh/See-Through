import os
import json
import h5py
import numpy as np
from tqdm import tqdm

# Create output directory for non-static processed data
os.makedirs("NonStaticData", exist_ok=True)

# Define dtype for processed detections
detection_dtype = np.dtype([
    ("x_cc", "f4"),
    ("y_cc", "f4"),
    ("label_id", "u1"),
    ("track_id", "i4"),
    ("sensor_id", "u1"),
    ("rcs", "f4"),
    ("vr", "f4"),
    ("vr_compensated", "f4"),
])

def filter_and_save_non_static(input_path, output_path):
    with h5py.File(input_path, "r") as in_file:
        frames = in_file["frames"][:]
        detections = in_file["detections"][:]

        # Prepare new frame and detection lists
        new_frames = []
        new_detections = []

        current_index = 0
        for frame in frames:
            start, end = frame["detection_start_idx"], frame["detection_end_idx"]
            frame_dets = detections[start:end]
            
            # Remove static points (label_id == 11)
            non_static_dets = frame_dets[frame_dets["label_id"] != 11]
            count = len(non_static_dets)
            
            if count == 0:
                continue  # Skip frame if all points are static

            new_frame = (
                frame["odometry_index"],
                frame["timestamp"],
                frame["ego_velocity"],
                frame["ego_yaw_rate"],
                current_index,
                current_index + count
            )
            new_frames.append(new_frame)
            new_detections.append(non_static_dets)
            current_index += count

        if len(new_frames) == 0:
            print(f"⚠️  No non-static data in {input_path}, skipping.")
            return

        # Save filtered data
        with h5py.File(output_path, "w") as out_file:
            out_file.attrs["sequence_name"] = in_file.attrs["sequence_name"]
            out_file.create_dataset("frames", data=new_frames, dtype=frames.dtype)
            out_file.create_dataset("detections", data=np.concatenate(new_detections), dtype=detection_dtype)

# Paths
input_dir = "ProcessedData"
output_dir = "NonStaticData"
sequences = [f"sequence_{i}.h5" for i in range(1, 159)]

# Process each sequence
for seq in tqdm(sequences, desc="Filtering static points"):
    input_path = os.path.join(input_dir, seq)
    output_path = os.path.join(output_dir, seq)
    
    if not os.path.exists(input_path):
        print(f"Missing: {input_path}")
        continue
    
    filter_and_save_non_static(input_path, output_path)

print(f"\n✅ Done! Cleaned sequences saved to: {output_dir}")

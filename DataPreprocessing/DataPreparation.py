import os
import json
import h5py
import numpy as np
from tqdm import tqdm

# This comes after grouping by odmetery index
class DataPreparation:
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

    def __init__(self, input_dir="MovingObjectsData", output_dir="NormlizedData"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def filter_and_save_non_static(self, input_path, output_path):
        with h5py.File(input_path, "r") as in_file:
            frames = in_file["frames"][:]
            detections = in_file["detections"][:]

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
                    continue  # Skip frames with no non-static points

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
                print(f"  No non-static data in {input_path}, skipping.")
                return

            with h5py.File(output_path, "w") as out_file:
                out_file.attrs["sequence_name"] = in_file.attrs["sequence_name"]
                out_file.create_dataset("frames", data=new_frames, dtype=frames.dtype)
                out_file.create_dataset("detections", data=np.concatenate(new_detections), dtype=self.detection_dtype)

    def filter_by_velocity(self, input_path, output_path, velocity_threshold=0.1):
        """Filter points with absolute compensated velocity below threshold."""
        with h5py.File(input_path, "r") as in_file:
            frames = in_file["frames"][:]
            detections = in_file["detections"][:]

            new_frames = []
            new_detections = []

            current_index = 0
            for frame in frames:
                start, end = frame["detection_start_idx"], frame["detection_end_idx"]
                frame_dets = detections[start:end]

                # Filter points with |vr_compensated| >= velocity_threshold
                moving_points = frame_dets[np.abs(frame_dets["vr_compensated"]) >= velocity_threshold]
                count = len(moving_points)

                if count == 0:
                    continue  # Skip frames with no points above velocity threshold

                new_frame = (
                    frame["odometry_index"],
                    frame["timestamp"],
                    frame["ego_velocity"],
                    frame["ego_yaw_rate"],
                    current_index,
                    current_index + count
                )
                new_frames.append(new_frame)
                new_detections.append(moving_points)
                current_index += count

            if len(new_frames) == 0:
                print(f"  No points above velocity threshold in {input_path}, skipping.")
                return

            with h5py.File(output_path, "w") as out_file:
                out_file.attrs["sequence_name"] = in_file.attrs["sequence_name"]
                out_file.create_dataset("frames", data=new_frames, dtype=frames.dtype)
                out_file.create_dataset("detections", data=np.concatenate(new_detections), dtype=self.detection_dtype)
    
    def NormalizeData(self, input_path, output_path):
        """Normalize point cloud by centering and scaling per frame."""
        with h5py.File(input_path, "r") as in_file:
            frames = in_file["frames"][:]
            detections = in_file["detections"][:]
            sequence_name = in_file.attrs["sequence_name"]  

            normalized_detections = detections.copy()

            for frame in frames:
                start, end = frame["detection_start_idx"], frame["detection_end_idx"]
                frame_dets = normalized_detections[start:end]

                # Centering
                x_mean = np.mean(frame_dets["x_cc"])
                y_mean = np.mean(frame_dets["y_cc"])
                frame_dets["x_cc"] -= x_mean
                frame_dets["y_cc"] -= y_mean

                normalized_detections[start:end] = frame_dets

        with h5py.File(output_path, "w") as out_file:
            out_file.attrs["sequence_name"] = sequence_name
            out_file.create_dataset("frames", data=frames, dtype=frames.dtype)
            out_file.create_dataset("detections", data=normalized_detections, dtype=self.detection_dtype)

        print(f" Normalized data saved to: {output_path}")



    def process_all_sequences(self, sequences=None, velocity_filter=False, velocity_threshold=0.1, normlize_data = False):
        if sequences is None:
            sequences = [f"sequence_{i}.h5" for i in range(1, 159)]

        for seq in tqdm(sequences, desc="Processing sequences"):
            input_path = os.path.join(self.input_dir, seq)
            output_path = os.path.join(self.output_dir, seq)

            if not os.path.exists(input_path):
                print(f"Missing: {input_path}")
                continue

            if velocity_filter:
                self.filter_by_velocity(input_path, output_path, velocity_threshold)

            elif normlize_data:
                self.NormalizeData(input_path, output_path)

            else:
                self.filter_and_save_non_static(input_path, output_path)

        print(f"\n Done! Processed sequences saved to: {self.output_dir}")



prep = DataPreparation()

# Velocity = False => Remove Static data | Velocity = True => Filter by velocity
prep.process_all_sequences(velocity_filter=False, velocity_threshold=0.1, normlize_data=True) 

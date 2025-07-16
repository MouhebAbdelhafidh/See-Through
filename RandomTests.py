import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import open3d as o3d

class Tests:
    def __init__(self, data_folder="NonStaticData"):
        self.data_folder = data_folder
        
        # RadarScenes color mapping (normalized RGB)
        self.class_colors = np.array([
            [255, 0, 0],    # 0: car
            [255, 165, 0],  # 1: large vehicle
            [139, 0, 139],  # 2: truck
            [0, 0, 255],    # 3: bus
            [0, 255, 255],  # 4: train
            [0, 255, 0],    # 5: bicycle
            [255, 255, 0],  # 6: motorcycle
            [255, 192, 203],# 7: pedestrian
            [165, 42, 42],  # 8: pedestrian group
            [0, 128, 0],    # 9: animal
            [128, 128, 128],# 10: other dynamic
            [64, 64, 64]    # 11: static
        ]) / 255.0
        
        self.class_names = [
            "Car", "Large Vehicle", "Truck", "Bus", "Train",
            "Bicycle", "Motorcycle", "Pedestrian", "Pedestrian Group",
            "Animal", "Other Dynamic", "Static"
        ]

    def _load_sequence(self, seq_num):
        """Load frames and detections for a given sequence number."""
        seq_file = os.path.join(self.data_folder, f"sequence_{seq_num}.h5")
        if not os.path.exists(seq_file):
            raise FileNotFoundError(f"Sequence file {seq_file} not found.")
        with h5py.File(seq_file, "r") as f:
            frames = f["frames"][:]
            detections = f["detections"][:]
        return frames, detections

    def show_sensors_per_frame(self, seq_num):
        frames, detections = self._load_sequence(seq_num)
        for i in range(len(frames)):
            frame = frames[i]
            dets = detections[frame["detection_start_idx"]:frame["detection_end_idx"]]
            sensor_ids = np.unique(dets["sensor_id"])
            if len(sensor_ids) > 1:
                print(f"Frame {i} has sensors: {sensor_ids}")

    def get_window_detections(self, seq_num, center_idx, window_size=2, vr_threshold=0.1):
        frames, detections = self._load_sequence(seq_num)
        merged = []
        for offset in range(-window_size, window_size + 1):
            idx = center_idx + offset
            if 0 <= idx < len(frames):
                f = frames[idx]
                points = detections[f["detection_start_idx"]:f["detection_end_idx"]]
                moving_points = points[np.abs(points["vr_compensated"]) > vr_threshold]
                merged.append(moving_points)
        return np.concatenate(merged) if merged else np.array([])

    def plot_frame_2d(self, seq_num, frame_idx, window_size=2):
        merged_points = self.get_window_detections(seq_num, frame_idx, window_size)
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            merged_points["x_cc"],
            merged_points["y_cc"],
            c=merged_points["label_id"],
            cmap=ListedColormap(self.class_colors),
            s=5,
            vmin=0,
            vmax=11
        )
        ax.scatter([0], [0], c='black', marker='s', s=100, label='Ego Vehicle')
        ax.set_title(f"Sliding Window: Sequence {seq_num} Frame {frame_idx} ± {window_size}")
        ax.set_xlabel("X (Longitudinal) [m]")
        ax.set_ylabel("Y (Lateral) [m]")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-25, 25)
        ax.grid(True)
        ax.legend()

        cbar = plt.colorbar(scatter, ticks=range(12))
        cbar.set_label("Semantic Class")
        cbar.set_ticklabels(self.class_names)

        plt.show()

    def visualize_frame_3d(self, seq_num, frame_idx, window_size=2):
        merged_points = self.get_window_detections(seq_num, frame_idx, window_size)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.column_stack([merged_points["x_cc"], merged_points["y_cc"], np.zeros_like(merged_points["x_cc"])])
        )
        colors = self.class_colors[merged_points["label_id"]]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        ego = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        ego.paint_uniform_color([0, 0, 0])
        ego.translate([0, 0, 0])

        o3d.visualization.draw_geometries([pcd, ego], window_name=f"Sequence {seq_num} Frame {frame_idx} ± {window_size}")

    def plot_class_distribution(self):
        class_counts = np.zeros(len(self.class_names), dtype=int)
        for filename in os.listdir(self.data_folder):
            if not filename.endswith(".h5"):
                continue
            filepath = os.path.join(self.data_folder, filename)
            with h5py.File(filepath, "r") as f:
                detections = f["detections"][:]
                labels = detections["label_id"]
                for i in range(len(self.class_names)):
                    class_counts[i] += np.sum(labels == i)

        plt.figure(figsize=(12,6))
        bars = plt.bar(self.class_names, class_counts, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Number of Detections")
        plt.title("Radar Point Cloud Class Distribution (NonStaticData)")
        plt.tight_layout()

        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{count:,}', 
                     ha='center', va='bottom', fontsize=9)

        plt.show()

        
tests = Tests(data_folder="NonStaticData")

# Show sensors for sequence 10
tests.show_sensors_per_frame(seq_num=10)

# Plot 2D sliding window for sequence 10, frame 15
tests.plot_frame_2d(seq_num=10, frame_idx=15, window_size=2)

# Visualize 3D sliding window for sequence 10, frame 15
tests.visualize_frame_3d(seq_num=10, frame_idx=15, window_size=2)

# Plot histogram across all sequences in the folder
tests.plot_class_distribution()

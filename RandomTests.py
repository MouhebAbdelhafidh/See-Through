import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import open3d as o3d

class Tests:
    def __init__(self, data_folder="DataPreprocessing"):
        self.data_folder = data_folder

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

    def extract_h5_summary(self, file_path, max_frames=5):
        with h5py.File(file_path, 'r') as f:
            detections = f['detections'][:]
            frames = f['frames'][:]

            summary = {}

            for i, frame in enumerate(frames[:max_frames]):
                frame_dict = {
                    "timestamp": int(frame['timestamp']),
                    "ego_velocity": frame['ego_velocity'].tolist(),
                    "ego_yaw_rate": float(frame['ego_yaw_rate']),
                    "detections": []
                }

                start_idx = frame['detection_start_idx']
                end_idx = frame['detection_end_idx']
                frame_detections = detections[start_idx:end_idx]

                for det in frame_detections:
                    frame_dict["detections"].append({
                        "x_cc": float(det["x_cc"]),
                        "y_cc": float(det["y_cc"]),
                        "sensor_id": int(det["sensor_id"]),
                        "rcs": float(det["rcs"]),
                        "vr": float(det["vr"]),
                        "vr_compensated": float(det["vr_compensated"]),
                        "label_id": int(det["label_id"]),
                        "track_id": int(det["track_id"]),
                    })

                summary[f"odometry_index_{int(frame['odometry_index'])}"] = frame_dict

        print(json.dumps(summary, indent=4))

    def compare_real_vs_generated(self, real_path, fake_path, label_id=1):
        def load_points(h5_path, label_id):
            with h5py.File(h5_path, 'r') as f:
                detections = f['detections'][:]
                if 'label_id' in detections.dtype.names:
                    detections = detections[detections['label_id'] == label_id]
                x = detections['x_cc']
                y = detections['y_cc']
            return x, y

        x_real, y_real = load_points(real_path, label_id)
        x_fake, y_fake = load_points(fake_path, label_id)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(x_real, y_real, s=5, c='blue', alpha=0.6)
        plt.title('Real Data')
        plt.xlabel('x_cc')
        plt.ylabel('y_cc')
        plt.axis('equal')

        plt.subplot(1, 2, 2)
        plt.scatter(x_fake, y_fake, s=5, c='green', alpha=0.6)
        plt.title('Generated Data (GAN Output)')
        plt.xlabel('x_cc')
        plt.ylabel('y_cc')
        plt.axis('equal')

        plt.tight_layout()
        plt.show()

    def plot_feature_distributions(self, real_path, fake_path, label_id=None, features=None, bins=50):
        if features is None:
            features = ["x_cc", "y_cc", "rcs", "vr", "vr_compensated"]

        def load_selected_features(h5_path, label_id, features):
            with h5py.File(h5_path, 'r') as f:
                detections = f['detections'][:]
                if label_id is not None and 'label_id' in detections.dtype.names:
                    detections = detections[detections['label_id'] == label_id]
                return {feat: detections[feat] for feat in features if feat in detections.dtype.names}

        real_data = load_selected_features(real_path, label_id, features)
        fake_data = load_selected_features(fake_path, label_id, features)

        n = len(features)
        plt.figure(figsize=(5 * n, 5))
        for i, feat in enumerate(features):
            plt.subplot(1, n, i + 1)
            plt.hist(real_data[feat], bins=bins, alpha=0.6, label='Real', color='blue', density=True)
            plt.hist(fake_data[feat], bins=bins, alpha=0.6, label='Generated', color='green', density=True)
            plt.title(f"Distribution of {feat}")
            plt.xlabel(feat)
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    tests = Tests(data_folder="NormlizedData")

    # Test 1: Show frames with multiple sensors
    # tests.show_sensors_per_frame(seq_num=10)

    # Test 2: Plot 2D points 
    # tests.plot_frame_2d(seq_num=125, frame_idx=145, window_size=2)

    # Test 3: Plot class histogram
    # tests.plot_class_distribution()

    # Test 4: Extract summary of H5
    # tests.extract_h5_summary("NormlizedData/sequence_99.h5", max_frames=3)

    # Test 5: GAN vs real (2D plot)
    tests.compare_real_vs_generated(
        real_path="NormlizedData/sequence_125.h5",
        fake_path="DataPreprocessing/FakeData/sequence_125_fake_label8.h5",
        label_id=8
    )

    # Test 6: GAN vs real (Distribution)
    tests.plot_feature_distributions(
    real_path="NormlizedData/sequence_125.h5",
    fake_path="DataPreprocessing/FakeData/sequence_125_fake_label8.h5",
    label_id=8  
    )

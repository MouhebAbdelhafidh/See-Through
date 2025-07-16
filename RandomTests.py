import os
import h5py
import numpy as np

# # =================================================================
# # See The sensors involved in a sequence per frame
# # =================================================================
path = "ProcessedData/sequence_158.h5"
with h5py.File(path, "r") as f:
    frames = f["frames"]
    detections = f["detections"]
    
    for i in range(len(frames)):
        frame = frames[i]
        dets = detections[frame["detection_start_idx"]:frame["detection_end_idx"]]
        sensor_ids = np.unique(dets["sensor_id"])
        if len(sensor_ids) > 1:
            print(f"Frame {i} has sensors: {sensor_ids}")


import h5py
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.colors import ListedColormap

# # =================================================================
# # Plot the point cloud Visualizations 2D + 3D
# # =================================================================

file_path = "ProcessedData/sequence_158.h5"
with h5py.File(file_path, "r") as f:
    frames = f["frames"][:]
    detections = f["detections"][:]
    print(f"Loaded {len(frames)} frames with {len(detections)} total detections")

# RadarScenes color mapping
class_colors = np.array([
    [255, 0, 0],    # 0: car (red)
    [255, 165, 0],  # 1: large vehicle (orange)
    [139, 0, 139],  # 2: truck (purple)
    [0, 0, 255],    # 3: bus (blue)
    [0, 255, 255],  # 4: train (cyan)
    [0, 255, 0],    # 5: bicycle (green)
    [255, 255, 0],  # 6: motorcycle (yellow)
    [255, 192, 203],# 7: pedestrian (pink)
    [165, 42, 42],  # 8: pedestrian group (brown)
    [0, 128, 0],    # 9: animal (dark green)
    [128, 128, 128],# 10: other dynamic (gray)
    [64, 64, 64]    # 11: static (dark gray)
]) / 255.0  # Normalize to [0,1]

# 2D view
def plot_frame_2d(frame_idx):
    """Plot a single frame's detections in 2D"""
    frame = frames[frame_idx]
    start, end = frame["detection_start_idx"], frame["detection_end_idx"]
    points = detections[start:end]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    scatter = ax.scatter(
        points["x_cc"], 
        points["y_cc"], 
        c=points["label_id"],
        cmap=ListedColormap(class_colors),
        s=5,
        vmin=0,
        vmax=11
    )
    
    ax.scatter([0], [0], c='black', marker='s', s=100, label='Ego Vehicle')
    
    # Formatting
    ax.set_title(f"Frame {frame_idx} | Velocity: {frame['ego_velocity']:.1f} m/s")
    ax.set_xlabel("X (Longitudinal) [m]")
    ax.set_ylabel("Y (Lateral) [m]")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-25, 25)
    ax.grid(True)
    ax.legend()
    
    cbar = plt.colorbar(scatter, ticks=range(12))
    cbar.set_label("Semantic Class")
    cbar.set_ticklabels([
        "Car", "LargeVeh", "Truck", "Bus", "Train", 
        "Bicycle", "Motorcycle", "Ped", "PedGroup", 
        "Animal", "Other", "Static"
    ])
    
    plt.show()

# 3D view
def visualize_frame_3d(frame_idx):
    """Interactive 3D visualization with Open3D"""
    frame = frames[frame_idx]
    start, end = frame["detection_start_idx"], frame["detection_end_idx"]
    points = detections[start:end]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.column_stack([points["x_cc"], points["y_cc"], np.zeros_like(points["x_cc"])])
    )
    
    # Color by class
    colors = class_colors[points["label_id"]]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create ego vehicle marker
    ego_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    ego_marker.paint_uniform_color([0, 0, 0])
    ego_marker.translate([0, 0, 0])
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, ego_marker], window_name=f"Frame {frame_idx}")

# Run visualizations
if __name__ == "__main__":
    # Visualize first frame
    # plot_frame_2d(626)
    # visualize_frame_3d(626)
    
    # Uncomment to browse through frames
    for i in range(len(frames)):
        plot_frame_2d(i)
        visualize_frame_3d(i)
        if input("Press q to quit, any key to continue: ") == 'q':
            break
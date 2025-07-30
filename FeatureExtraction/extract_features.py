import os
import h5py
import numpy as np
from chainer_pointnet.models.pointnet2_cls_msg import PointNet2MSG
import chainer
from chainer import Variable

# === CONFIG ===
H5_FILE = "../../RadarScenes/FusedData/sequence_100_fused.h5"  # adjust path if needed
OUTPUT_FEATURES = "features_output.npy"
BATCH_SIZE = 1
NUM_POINTS = 1024  # randomly sample 1024 points per frame
USE_GPU = True

# === Step 1: Load radar data from HDF5 ===
def load_fused_points(h5_path):
    with h5py.File(h5_path, 'r') as f:
        data = f['fused_detections'][:]
        points = np.stack([data['x_cc'], data['y_cc'], data['vr']], axis=-1)
    return points

# === Step 2: Prepare PointNet++ input format ===
def prepare_batches(points, num_points=NUM_POINTS):
    N = len(points)
    if N < num_points:
        pad = points[np.random.choice(N, num_points - N)]
        points = np.concatenate([points, pad], axis=0)
    else:
        indices = np.random.choice(N, num_points, replace=False)
        points = points[indices]
    return np.expand_dims(points.astype(np.float32), axis=0)  # shape: (1, 1024, 3)

# === Step 3: Load pretrained PointNet++ and extract features ===
def extract_features(points_batch):
    model = PointNet2MSG(n_classes=40)
    chainer.serializers.load_npz("../../chainer-pointnet/experiments/modelnet40/pointnet2_cls_msg_best.npz", model)
    if USE_GPU:
        chainer.cuda.get_device_from_id(0).use()
        model.to_gpu()
        points_batch = chainer.backends.cuda.to_gpu(points_batch)
    with chainer.using_config("train", False):
        x = Variable(points_batch.transpose((0, 2, 1)))  # shape: (B, 3, N)
        _, features = model(x, return_feature=True)
    return features.array  # shape: (B, F)

# === Step 4: Full pipeline ===
def main():
    print("Loading radar data...")
    points = load_fused_points(H5_FILE)
    print(f"Total points loaded: {len(points)}")

    print("Preparing PointNet++ input...")
    points_batch = prepare_batches(points)

    print("Extracting features using pretrained PointNet++...")
    features = extract_features(points_batch)
    print(f"Extracted feature shape: {features.shape}")

    np.save(OUTPUT_FEATURES, features)
    print(f"Saved features to {OUTPUT_FEATURES}")

if __name__ == "__main__":
    main()

####################################################################
# File: test_votenet_head.py
# Minimal test script for VoteNetHead alone.
# Run: python test_votenet_head.py
####################################################################

import numpy as np
import torch
from pprint import pprint
from votenet_head import VoteNetHead 
import os

if __name__ == "__main__":
    DATA_PATH = "precomputed_data.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading features from:", DATA_PATH)
    data = np.load(DATA_PATH, allow_pickle=True)
    features = data["features"]  # (N, 1024)

    # Pick a small batch (B=8)
    B = 8
    sample = features[:B].astype(np.float32)
    x = torch.from_numpy(sample).to(device)

    # Create head
    model = VoteNetHead(in_dim=features.shape[1], num_classes=11).to(device)

    # Load saved weights if available
    if os.path.exists("votenet_head.pth"):
        try:
            model.load_state_dict(torch.load("votenet_head.pth", map_location=device))
            print("Loaded votenet_head.pth")
        except Exception as e:
            print("Could not load votenet_head.pth:", e)

    # Inference
    model.eval()
    with torch.no_grad():
        out = model(x)

    # Output structure
    print("\nModel outputs (keys):")
    pprint(list(out.keys()))

    print("\nShapes:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k}: {tuple(sub_v.shape)}")

    # Example semantic prediction
    sem_logits = out["classification"]["semantic"][0]
    sem_prob = torch.softmax(sem_logits, dim=0).cpu().numpy()
    print("\nSemantic probs (first sample, top-5):")
    topk = sem_prob.argsort()[::-1][:5]
    for i in topk:
        print(f"  class {i}: {sem_prob[i]:.4f}")

    # Example size prediction
    size_scores = out["bbox"]["size_scores"][0]
    size_cluster = int(torch.argmax(size_scores).item())
    size_residual = out["bbox"]["size_residuals"][0, size_cluster].cpu().numpy()

    print(f"\nSample 0: chosen size cluster={size_cluster}, residual={size_residual}")

    # Example heading prediction
    heading_bin = int(torch.argmax(out["bbox"]["heading_scores"][0]).item())
    heading_res = float(out["bbox"]["heading_residuals"][0, heading_bin].cpu().item())

    print(f"Sample 0: heading bin={heading_bin}, heading residual={heading_res:.4f}")

    # Example objectness prediction
    obj_logits = out["classification"]["objectness"][0]
    obj_prob = torch.softmax(obj_logits, dim=0).cpu().numpy()
    print(f"Sample 0: objectness prob (bg,obj) = {obj_prob}")

    # Example velocity prediction
    vel = out["velocity"][0].cpu().numpy()
    print(f"Sample 0: velocity (vx,vy) = {vel}")

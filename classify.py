import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from extract_features import PointNet2Backbone
from votenet_classifier import VoteNetClassifier
from train_feature_extractor import RadarPointCloudDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classify():
    # === Load dataset ===
    dataset = RadarPointCloudDataset("FusedData")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # === Load backbone ===
    backbone = PointNet2Backbone().to(device)
    backbone.load_state_dict(torch.load("checkpoints/pointnetpp_backbone.pth", map_location=device))
    backbone.eval()

    # === Initialize VoteNet-style classifier ===
    classifier = VoteNetClassifier(num_classes=dataset.num_classes).to(device)
    # classifier.load_state_dict(torch.load("checkpoints/votenet_classifier.pth", map_location=device))  # Optional
    classifier.eval()

    # === Classify ===
    with torch.no_grad():
        for idx, (xyz, features, label) in enumerate(dataloader):
            xyz, features = xyz.to(device), features.to(device)

            feat, _, _ = backbone(xyz, features)
            logits = classifier(feat)
            pred = torch.argmax(logits, dim=1)

            print(f"Sample {idx}: Predicted = {pred.item()}, Ground Truth = {label.item()}")


if __name__ == "__main__":
    classify()

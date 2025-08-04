import torch
import torch.nn as nn
import torch.nn.functional as F


class VoteNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VoteNetClassifier, self).__init__()

        # VoteNet-style classification head
        self.vote_layer = nn.Sequential(
            nn.Linear(1024, 256),  # Assuming input feature vector is of size 1024
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, global_features):
        logits = self.vote_layer(global_features)
        return logits

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from pytorch_metric_learning import losses
from torchvision import models, transforms
from PIL import Image



device = "cuda"

# MLP for dimensionality reduction
class DiversifyingMLP(nn.Module):
    def __init__(self, input_dim=2048, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[3].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.net(x)

    def fit(self, holdout_features, holdout_labels, epochs=100, batch_size=32, learning_rate=1e-3, margin=0.2):
        """
        Train MLP with triplet loss on holdout dataset.

        Args:
            holdout_features: Tensor of ResNet features [n_samples, feature_dim]
            holdout_labels: Tensor of labels [n_samples]
            epochs: Number of training epochs
            batch_size: Batch size for triplet training
            learning_rate: Learning rate for Adam optimizer
            margin: Margin for triplet loss
        """
        criterion = losses.TripletMarginLoss(margin=margin)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Select semi-hard triplets
        def select_semi_hard_triplets(features, labels, num_triplets=1000):
            triplets = []
            features = features.cpu().numpy() if features.is_cuda else features.numpy()
            labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
            for i in range(len(features)):
                anchor = features[i]
                anchor_label = labels[i]
                pos_indices = np.where(labels == anchor_label)[0]
                neg_indices = np.where(labels != anchor_label)[0]
                pos_dist = np.linalg.norm(features[pos_indices] - anchor, axis=1)
                neg_dist = np.linalg.norm(features[neg_indices] - anchor, axis=1)
                if len(pos_dist) > 0 and len(neg_dist) > 0:
                    valid_pos = pos_indices[pos_dist > np.min(neg_dist) + margin]
                    valid_neg = neg_indices[neg_dist < np.max(pos_dist) - margin]
                    if len(valid_pos) > 0 and len(valid_neg) > 0:
                        hardest_pos = valid_pos[np.argmax(pos_dist[valid_pos])]
                        hardest_neg = valid_neg[np.argmin(neg_dist[valid_neg])]
                        triplets.append((i, hardest_pos, hardest_neg))
            return triplets[:min(num_triplets, len(triplets))]

        triplets = select_semi_hard_triplets(holdout_features, holdout_labels)
        triplet_dataset = TensorDataset(
            torch.tensor([holdout_features[i] for i, _, _ in triplets], dtype=torch.float32),
            torch.tensor([holdout_features[j] for _, j, _ in triplets], dtype=torch.float32),
            torch.tensor([holdout_features[k] for _, _, k in triplets], dtype=torch.float32)
        )
        dataloader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            for anchor, positive, negative in dataloader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_out = self(anchor)
                positive_out = self(positive)
                negative_out = self(negative)
                loss = criterion(anchor_out, positive_out, negative_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def fit_transform(self, cal_feature, test_feature):
        """
        Transform calibration and test features using trained MLP.

        Args:
            cal_feature: Tensor of calibration features [n_cal, feature_dim]
            test_feature: Tensor of test feature [feature_dim]
        Returns:
            cal_reduced: Reduced calibration features [n_cal, output_dim]
            test_reduced: Reduced test feature [output_dim]
        """
        return self(cal_feature), self(test_feature)


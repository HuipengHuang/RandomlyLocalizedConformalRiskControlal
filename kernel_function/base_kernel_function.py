from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import os
from dim_reduction.utils import get_dimension_reduction_tool
import matplotlib.pyplot as plt


class BaseKernelFunction(ABC):
    def __init__(self, args, holdout_feature=None, holdout_target=None, h=None):
        #  h means hyperparamter
        self.args = args
        self.holdout_feature = holdout_feature
        self.h = h
        self.h_lower = None
        self.h_upper = None

        self.dimension_reduction_tool = get_dimension_reduction_tool(args, holdout_feature, holdout_target)

    @abstractmethod
    def sample(self, test_feature):
        """Randomly sample a feature from the distribution of the localized function"""
        pass

    def fit_transform(self, cal_feature, test_feature):
        new_cal_feature, new_test_feature = self.dimension_reduction_tool.fit_transform(cal_feature, test_feature)
        return new_cal_feature, new_test_feature

    def calculate_avg_efficient_sample_size(self, weight):
        """Weight shape:[test_size, cal_size+1]"""

        efficient_size = (torch.sum(weight[:, :-1], dim=-1) ** 2) / torch.sum(weight[:, :-1] ** 2, dim=-1)
        avg_efficient_size = torch.mean(efficient_size)
        assert torch.sum(torch.isnan(avg_efficient_size)) == 0
        return avg_efficient_size

    @abstractmethod
    def calculate_weight(self, cal_feature, test_feature, sampled_features, d):
        pass

    def get_p(self, cal_feature, test_feature, test_score, test_target):
        """
            Args:
                cal_feature shape: [calibration_set_size, feature_dim], test_feature shape: [batch_size, feature_dim]
                """
        #cal_feature, test_feature = cal_feature / torch.norm(cal_feature, dim=-1, keepdim=True), test_feature / torch.norm(test_feature, dim=-1, keepdim=True)
        if self.dimension_reduction_tool is not None:
                cal_feature, test_feature = self.fit_transform(cal_feature, test_feature)

        if self.args.plot == "True" and self.args.current_run == 0:
            plot_feature_distance(self.args, cal_feature, test_feature, test_score, test_target)

        d = test_feature.shape[1]

        if self.args.efficient_calibration_size is None:
            sampled_features = self.sample(test_feature)
            weight = self.calculate_weight(cal_feature, test_feature, sampled_features, d)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)
        elif self.args.adaptive == "False":
            assert self.h is None
            print("Find hyperparamters---------")
            h = self.get_h(cal_feature, test_feature)
            sampled_features = self.sample(test_feature, h)
            print("Finish finding hyperparameter")

            weight = self.calculate_weight(cal_feature, test_feature, sampled_features, h)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)
        else:
            print("Find hyperparamters---------")
            h = torch.zeros(size=(test_feature.shape[0], 1), device="cuda")

            for i in tqdm(range(test_feature.shape[0]), desc=f"Finding Hyperparameter"):
                h[i] = self.get_h(cal_feature, test_feature[i].unsqueeze(0))
            print("Finish finding hyperparameter")

            sampled_features = self.sample(test_feature, h)

            weight = self.calculate_weight(cal_feature, test_feature, sampled_features, h)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)
        return p

    def get_h(self, cal_feature, test_feature):
        if test_feature.shape[0] > 10000:
            sub_test_feature = test_feature[:100]
        else:
            sub_test_feature = test_feature
        efficient_calibration_size = self.args.efficient_calibration_size
        assert efficient_calibration_size <= self.args.calibration_size
        #h_lower = 1e10
        #h_upper = 256.0
        h_lower = self.h_lower if self.h_lower else cal_feature.shape[0] **(-1 / (cal_feature.shape[0]+4)) * torch.std(cal_feature)
        h_upper = self.h_upper if self.h_upper else cal_feature.shape[0] **(-1 / (cal_feature.shape[0]+4)) * torch.std(cal_feature)
        while (True):
            sampled_features = self.sample(sub_test_feature, h_lower)
            weight = self.calculate_weight(cal_feature, sub_test_feature, sampled_features, h_lower)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)

            efficient_size = self.calculate_avg_efficient_sample_size(p)

            if efficient_size <= efficient_calibration_size:
                break
            else:
                h_upper = h_lower.clone().cpu().to("cuda")
                h_lower /= 2

        while (True):
            sampled_features = self.sample(sub_test_feature, h_upper)
            weight = self.calculate_weight(cal_feature, sub_test_feature, sampled_features, h_upper)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)
            efficient_size = self.calculate_avg_efficient_sample_size(p)

            if efficient_size >= efficient_calibration_size:
                break
            else:
                h_upper *= 2

        while(True):
            mid = (h_lower + h_upper) / 2
            sampled_features = self.sample(sub_test_feature, mid)
            weight = self.calculate_weight(cal_feature, sub_test_feature, sampled_features, mid)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)
            efficient_size = self.calculate_avg_efficient_sample_size(p)

            if abs(efficient_size - self.args.efficient_calibration_size) < 0.1 * self.args.efficient_calibration_size:
                break

            if efficient_size >= efficient_calibration_size:
                h_upper = mid
            else:
                h_lower = mid
        self.h_lower = h_lower
        self.h_upper = h_upper

        return mid

    @abstractmethod
    def sample(self, test_feature, h):
        pass


def batched_pairwise_dist(feature, batch_size=1000, device='cuda'):
    """
    Compute pairwise squared distances in batches to avoid OOM errors

    Args:
        feature: Tensor of shape [n, d]
        batch_size: Number of rows to process at once
        device: Device to compute on

    Returns:
        dist_sq: Squared distance matrix of shape [n, n]
    """
    n = feature.shape[0]
    dist_sq = torch.zeros((n, n), device=device)

    # Pre-compute squared norms for all features
    feature_sqnorms = torch.sum(feature ** 2, dim=1)  # Shape [n]

    # Process in batches
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        batch_features = feature[i:batch_end]

        # Compute dot product for current batch
        batch_dot = batch_features @ feature.T  # Shape [batch_size, n]

        # Compute squared distances for current batch
        batch_dist_sq = (feature_sqnorms[i:batch_end, None] -
                         2 * batch_dot +
                         feature_sqnorms[None, :])

        # Store results
        dist_sq[i:batch_end] = (batch_dist_sq + 1e-6) ** 0.5

    # Handle numerical stability
    return dist_sq

def plot_feature_distance(args, cal_feature, test_feature, test_score=None, test_target=None):
    if args.plot_class == "True":
        plot_class_distance(test_feature,test_target)
    if args.plot_similar_threshold == "True":
        plot_similar_threshold_distance(test_feature, test_score)

    d = test_feature.shape[1]
    cal_distance = torch.zeros(size=(test_feature.shape[0], cal_feature.shape[0]), device="cuda")
    for i in range(test_feature.shape[0]):
        cal_distance[i] = torch.sum(((cal_feature - test_feature[i]) / d) ** 2, dim=-1) ** (0.5)
    min_val = torch.min(cal_distance)
    max_val = torch.max(cal_distance)
    normalized_distance = (cal_distance - min_val) / (max_val - min_val + 1e-8)

    plt.hist(normalized_distance.view(-1).cpu().numpy(), bins=100)
    i = 0
    path = f"./plot_results/distance{i}.pdf"

    while os.path.exists(path):
        i += 1
        path = f"./plot_results/distance{i}.pdf"

    plt.savefig(path)
    plt.show()

def plot_similar_threshold_distance(feature, score):
    target = score_to_label(score)
    plot_class_distance(feature, target)


def score_to_label(score):
  
    # Initialize output tensor with default label 2
    labels = torch.full_like(score, fill_value=4, dtype=torch.long)

    # Mask for label 0 (0.95 <= x <= 1.0)
    mask_label0 = (score >= 0.0) & (score <= 0.05)
    labels[mask_label0] = 0

    # Mask for label 1 (0.6 <= x <= 0.7)
    mask_label1 = (score > 0.05) & (score <= 0.1)
    labels[mask_label1] = 1

    mask_label1 = (score > 0.1) & (score <= 0.2)
    labels[mask_label1] = 2

    mask_label1 = (score > 0.2) & (score <= 0.4)
    labels[mask_label1] = 3

    mask_label1 = (score > 0.4) & (score <= 0.7)
    labels[mask_label1] = 3

    return labels

def plot_class_distance(test_feature, test_target):
    #feature = torch.cat((cal_feature, test_feature), dim=0)
    #target = torch.cat((cal_target, test_target), dim=0)
    feature, target = test_feature, test_target

    distances = batched_pairwise_dist(feature)

    distance_of_same_class = []
    distance_of_different_class = []
    unique_classes = torch.unique(target)

    for cls in unique_classes:
        # Create mask for current class
        class_mask = (target == cls)
        class_indices = torch.where(class_mask)[0]

        # Same-class distances (upper triangular to avoid duplicates)
        triu_indices = torch.triu_indices(len(class_indices), len(class_indices), offset=1)
        same_pairs = distances[class_indices[triu_indices[0]], class_indices[triu_indices[1]]]
        distance_of_same_class.append(same_pairs.view(-1))

        # Different-class distances
        other_indices = torch.where(target != cls)[0]
        if len(other_indices) > 0:
            # Get all cross-class pairs
            i, j = torch.meshgrid(class_indices, other_indices)
            diff_pairs = distances[i.flatten(), j.flatten()]
            distance_of_different_class.append(diff_pairs.view(-1))

    distance_of_same_class = torch.cat(distance_of_same_class, dim=0)
    distance_of_different_class = torch.cat(distance_of_different_class, dim=0)

    max_value = max(torch.max(distance_of_same_class).item(), torch.max(distance_of_different_class).item())
    min_value = min(torch.min(distance_of_same_class).item(), torch.min(distance_of_different_class).item())

    normalized_same = (distance_of_same_class - min_value) / (max_value - min_value)
    normalized_diff = (distance_of_different_class - min_value) / (max_value - min_value)

    plt.figure(figsize=(10, 6))

    plt.hist(normalized_diff.cpu().numpy(),
             bins=100,
             alpha=0.6,
             color='red',
             density=True,
             label='Different Class')

    # Plot histograms with different colors and transparency
    plt.hist(normalized_same.cpu().numpy(),
             bins=100,
             alpha=0.6,
             color='blue',
             density=True,
             label='Same Class')

    # Add plot decorations
    plt.xlabel('Normalized Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Normalized Distances by Class Relationship', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    i = 0
    path = f"./plot_results/class_distance{i}.pdf"

    while os.path.exists(path):
        i += 1
        path = f"./plot_results/class_distance{i}.pdf"

    plt.savefig(path)





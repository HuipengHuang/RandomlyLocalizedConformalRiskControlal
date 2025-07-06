from .base_kernel_function import BaseKernelFunction
import torch
from overrides import overrides

class BoxKernel(BaseKernelFunction):
    """Box Kernel: H(x,x′) = 1 if ∥x−x′∥_2 <= h, else 0"""
    def __init__(self, args, holdout_feature=None, holdout_target=None, h=None):
        super().__init__(args, holdout_feature, holdout_target, h)


    def calculate_weight(self, cal_feature, test_feature, sampled_features, h):
        # Ensure h is of shape [batch_size, ]
        if h.dim() == 0:
            h = torch.zeros(size=(test_feature.shape[0], 1), device="cuda") + h
        d = test_feature.shape[1]

        # Compute L2 distances between test features and sampled features
        # Shape: [batch_size, ]
        test_distance = torch.norm(test_feature - sampled_features, p=2, dim=-1) / (d * h)

        # Compute L2 distances between calibration features and sampled features
        # Shape: [batch_size, calibration_set_size]
        cal_distance = torch.zeros(size=(test_feature.shape[0], cal_feature.shape[0]), device="cuda")
        for i in range(test_feature.shape[0]):
            cal_distance[i] = torch.norm(cal_feature - sampled_features[i], p=2, dim=-1) / (d * h[i])

        # Concatenate calibration and test distances
        # Shape: [batch_size, calibration_set_size + 1]
        print(cal_feature.shape, test_distance.shape)
        l2 = torch.cat((cal_distance, test_distance.unsqueeze(dim=1)), dim=1)

        # Box kernel: Indicator function (1 if distance <= 1, else 0)
        weight = (l2 <= 1).float()

        return weight

    def sample(self, test_feature, h):
        """Sample features uniformly within the L2 ball of radius h * sqrt(d) centered at test_feature"""
        # Generate random directions (unit vectors)
        d = test_feature.shape[1]
        random_direction = torch.randn_like(test_feature)
        random_direction = random_direction / torch.norm(random_direction, p=2, dim=-1, keepdim=True)

        # Generate random radii (uniform in volume)
        u = torch.rand(test_feature.shape[0], 1, device="cuda")  # Uniform in [0, 1]
        random_radius = (u ** (1.0 / d)) * h * torch.sqrt(torch.tensor(d, device="cuda"))

        # Scale directions by radii and shift to test_feature
        sampled_features = test_feature + random_direction * random_radius

        return sampled_features

from .base_kernel_function import BaseKernelFunction
import torch
from overrides import overrides

class BoxKernel(BaseKernelFunction):
    """Box Kernel: H(x,x′) = 1 if ∥x−x′∥_2 <= h, else 0"""
    def __init__(self, args, holdout_feature=None, h=None):
        super().__init__(args, holdout_feature, h)


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
        l2 = torch.cat((cal_distance, test_distance.unsqueeze(dim=1)), dim=1)

        self.l2 = l2
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
    @overrides
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

        return mid, sampled_features
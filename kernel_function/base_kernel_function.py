from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import os
from dim_reduction.utils import get_dimension_reduction_tool

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

    def get_p(self, cal_feature, test_feature):
        """
            Args:
                cal_feature shape: [calibration_set_size, feature_dim], test_feature shape: [batch_size, feature_dim]
                """
        if self.dimension_reduction_tool is not None:
                cal_feature, test_feature = self.fit_transform(cal_feature, test_feature)
        print(self.args.current_run)
        print("haha")
        if self.args.current_run == 0:
            self.plot_feature_distance(cal_feature, test_feature)
            print("Plot feature distance")
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

    def plot_feature_distance(self, cal_feature, test_feature):
        d = test_feature.shape[1]
        cal_distance = torch.zeros(size=(test_feature.shape[0], cal_feature.shape[0]), device="cuda")
        for i in range(test_feature.shape[0]):
            cal_distance[i] = torch.sum(((cal_feature - test_feature[i]) / d) ** 2, dim=-1)**(0.5)
        min_val = torch.min(cal_distance)
        max_val = torch.max(cal_distance)
        normalized_distance = (cal_distance - min_val) / (max_val - min_val + 1e-8)
        import matplotlib.pyplot as plt
        plt.hist(normalized_distance.view(-1).cpu().numpy(), bins=100)
        i = 0
        path = f"./plot_results/distance{i}.pdf"

        while os.path.exists(path):
            i += 1
            path = f"./plot_results/distance{i}.pdf"

        plt.savefig(path)
        plt.show()


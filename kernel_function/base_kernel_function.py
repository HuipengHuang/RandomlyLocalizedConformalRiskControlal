from abc import ABC, abstractmethod
from PCA.utils import get_pca
import torch
from VAE.vae import VariationalAutoEncoder

class BaseKernelFunction(ABC):
    def __init__(self, args, holdout_feature=None, h=None):
        #  h means hyperparamter
        self.args = args
        self.holdout_feature = holdout_feature
        self.h = h
        if args.pca:
            self.PCA = get_pca(args)
            self.V = self.PCA.fit(self.holdout_feature)
        else:
            self.PCA = None
            self.V = None

        if args.vae:
            self.VAE = VariationalAutoEncoder(input_dim=holdout_feature.shape[1], latent_dim=args.latent_dim if args.latent_dim else 10).to("cuda")
            self.VAE.train()
            self.VAE.fit(self.holdout_feature)
            self.VAE.eval()
        else:
            self.VAE = None

    @abstractmethod
    def sample(self, test_feature):
        """Randomly sample a feature from the distribution of the localized function"""
        pass

    def fit_transform(self, cal_feature, test_feature):
        if self.holdout_feature is not None:
            if self.PCA is not None:
                new_cal_feature = torch.tensor([], device="cuda")
                new_test_feature = torch.tensor([], device="cuda")
                for i in range(cal_feature.shape[0]):
                    input_feature = torch.cat((self.holdout_feature, cal_feature[i].unsqueeze(0)), dim=0)
                    out_feature = self.PCA.fit_transform(input_feature)[-1].unsqueeze(0)
                    new_cal_feature = torch.cat((new_cal_feature, out_feature), dim=0)

                for i in range(test_feature.shape[0]):
                    input_feature = torch.cat((self.holdout_feature, test_feature[i].unsqueeze(0)), dim=0)
                    out_feature = self.PCA.fit_transform(input_feature)[-1].unsqueeze(0)
                    new_test_feature = torch.cat((new_test_feature, out_feature), dim=0)
            elif self.VAE is not None:
                new_cal_feature, new_test_feature = self.VAE.encode(cal_feature), self.VAE.encode(test_feature)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
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
        if self.PCA is not None or self.VAE is not None:
                cal_feature, test_feature = self.fit_transform(cal_feature, test_feature)

        d = test_feature.shape[1]

        if self.args.efficient_calibration_size is None:
            sampled_features = self.sample(test_feature)
            weight = self.calculate_weight(cal_feature, test_feature, sampled_features, d)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)
        else:
            assert self.h is None
            h, sampled_features = self.get_h(cal_feature, test_feature)
            weight = self.calculate_weight(cal_feature, test_feature, sampled_features, h)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)
        return p

    def get_h(self, cal_feature, test_feature):
        sub_test_feature =  test_feature[:100]
        print("Find hyperparamters---------")
        efficient_calibration_size = self.args.efficient_calibration_size
        assert efficient_calibration_size <= self.args.calibration_size
        #h_lower = 1e10
        #h_upper = 256.0
        h_lower = cal_feature.shape[0] **(-1 / (cal_feature.shape[0]+4)) * torch.std(cal_feature)
        h_upper = cal_feature.shape[0] **(-1 / (cal_feature.shape[0]+4)) * torch.std(cal_feature)
        while (True):
            sampled_features = self.sample(sub_test_feature, h_lower)
            weight = self.calculate_weight(cal_feature, sub_test_feature, sampled_features, h_lower)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)

            efficient_size = self.calculate_avg_efficient_sample_size(p)

            print(f"Finding h_lower. Efficient size {efficient_size} Lower:{h_lower}")
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

            print(f"Finding h_upper. Efficient size {efficient_size} h_upper:{h_upper}")
            if efficient_size >= efficient_calibration_size:
                break
            else:
                h_upper *= 2

        mid = None
        sampled_features = None
        while(True):
            mid = (h_lower + h_upper) / 2
            sampled_features = self.sample(sub_test_feature, mid)
            weight = self.calculate_weight(cal_feature, sub_test_feature, sampled_features, mid)
            p = weight / torch.sum(weight, dim=-1).unsqueeze(-1)
            efficient_size = self.calculate_avg_efficient_sample_size(p)
            print(f"Efficient size {efficient_size} Low : {h_lower} Mid:{mid} High: {h_upper}")

            if abs(efficient_size - self.args.efficient_calibration_size) < 0.1 * self.args.efficient_calibration_size:
                break

            if efficient_size >= efficient_calibration_size:
                h_upper = mid
            else:
                h_lower = mid
        print(f"Finish finding hyperparamters. Efficient sample size: {efficient_size}")
        sampled_features = self.sample(test_feature, mid)
        return mid, sampled_features

    @abstractmethod
    def sample(self, test_feature, h):
        pass


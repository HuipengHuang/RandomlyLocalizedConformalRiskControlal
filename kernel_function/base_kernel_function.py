from abc import ABC, abstractmethod
from PCA.utils import get_pca
import torch


class BaseKernelFunction(ABC):
    def __init__(self, args, holdout_feature=None):
        self.args = args
        self.holdout_feature = holdout_feature
        if args.pca:
            self.PCA = get_pca(args)
            self.V = self.PCA.fit(self.holdout_feature)
        else:
            self.PCA = None
            self.V = None
    @abstractmethod
    def get_weight(self, cal_feature, test_feature):
        """
        Return the normalized weight for calibration data and test data.
        Args:
            cal_feature shape: [calibration_set_size, feature_dim], test_feature shape: [batch_size, feature_dim]
        """
        pass

    def sample(self, test_feature):
        """Randomly sample a feature from the distribution of the localized function"""
        pass

    def fit_transform(self, cal_feature, test_feature):
        if self.holdout_feature is not None:
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
        else:
            raise NotImplementedError
        return new_cal_feature, new_test_feature
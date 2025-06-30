from abc import ABC, abstractmethod


class BaseKernelFunction(ABC):
    def __init__(self):
        pass

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
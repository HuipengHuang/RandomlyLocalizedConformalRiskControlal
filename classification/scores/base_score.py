from abc import ABCMeta, abstractmethod


class BaseScore():
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, prob):
        """Input probability, output scores."""
        raise NotImplementedError

    def compute_target_score(self, prob, target):
        """Input probability, output scores for target label."""
        raise NotImplementedError

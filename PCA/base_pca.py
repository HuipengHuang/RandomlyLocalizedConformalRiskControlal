import numpy as np
from abc import abstractmethod, ABCMeta


class BasePCA:
    __metaclass__ = ABCMeta

    def __init__(self, args):
        """
        Initialize PCA.

        Args:
            n_components (int): Number of principal components to keep.
        """
        self.n_components = args.n_components if args.n_components else None
        self.args = args
        self.mean = None  # shape [feature,]

    @abstractmethod
    def fit(self, X):
        """
        Fit PCA to the data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
        """
        return NotImplementedError

    @abstractmethod
    def transform(self, X, n_components=None):
        """
        Project data onto the principal components.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data (n_samples, n_components).
        """
        raise NotImplementedError

    def fit_transform(self, X):
        """Fit and transform the data."""
        self.fit(X)
        return self.transform(X)
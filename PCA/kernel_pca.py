import numpy as np
from .base_pca import BasePCA
from sklearn.metrics.pairwise import pairwise_kernels


class KernelPCA(BasePCA):
    def __init__(self, args):
        """
        Initialize Kernel PCA.

        Args:
            n_components (int): Number of principal components to keep
            kernel (str): Kernel type ('rbf', 'poly', 'linear')
            gamma (float): Kernel coefficient for 'rbf' and 'poly' kernels
        """
        super().__init__(args)
        self.kernel = args.kernel
        self.gamma = args.gamma
        self.X_fit = None
        self.alphas = None
        self.eg_vectors = None

        self.lambdas = None
        self.eg_values = None

        self.mean_kernel = None

    def _kernel(self, X, Y=None):
        """Compute kernel matrix"""
        return pairwise_kernels(X, Y=Y if Y is not None else X, metric=self.kernel, gamma=self.gamma)

    def fit(self, X):
        """
        Fit the Kernel PCA model to the data.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features)
        """
        self.X_fit = X

        # Compute kernel matrix
        kernel_X = self._kernel(X)

        # Center the kernel matrix
        K_centered = self.center(kernel_X)
        self.mean_kernel = K_centered

        # Compute eigenvalues and eigenvectors
        eg_value, eg_vectors = np.linalg.eigh(K_centered)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eg_value)[::-1]
        eg_value = eg_value[idx]
        eg_vectors = eg_vectors[:, idx]

        # Store first n_components components
        if self.n_components is None:
            self.n_components = X.shape[1]

        self.eg_values = eg_value
        self.eg_vectors = eg_vectors
        self.eg_vectors = self.eg_vectors / np.sqrt(self.eg_values + 1e-6)

    def transform(self, X, n_components=None):
        """
        Project data onto the kernel principal components.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features)
            n_components (int, optional): Number of components to use

        Returns:
            np.ndarray: Transformed data, shape (n_samples, n_components)
        """
        if n_components is None:
            n_components = self.n_components

        # Compute kernel matrix between new data and training data
        K_test = self._kernel(X, self.X_fit)

        K_test_centered = self.center(K_test)

        # Project onto principal components
        X_transformed = np.dot(K_test_centered, self.eg_vectors[:, :n_components])

        return X_transformed

    def center(self, K):
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        return K_centered

    def inverse_transform(self, X_transformed, n_components=None):
        """
        Transform data back to its original space from the reduced representation.

        Args:
            X_transformed (np.ndarray): Reduced data, shape (n_samples, n_components).
            n_components (int, optional): Number of components used in the transformation.
                                         Defaults to self.n_components.

        Returns:
            np.ndarray: Approximate reconstruction in original feature space.
        """
        n = n_components if n_components is not None else self.n_components

        # Compute the pseudo-inverse of the transformed training data
        X_transformed_train = self.eg_vectors[:, :n] * np.sqrt(self.eg_values[:n]+1e-10)

        # Project back to kernel space
        K_approx = np.dot(X_transformed, X_transformed_train.T)

        # Perform kernel ridge regression to approximate the inverse mapping
        K_train = self._kernel(self.X_fit)
        K_train_inv = np.linalg.pinv(K_train)

        # Reconstruct the original data approximation
        X_reconstructed = np.dot(K_approx, np.dot(K_train_inv, self.X_fit))

        return X_reconstructed
import numpy as np
from scipy.linalg import pinvh
from .base_pca import BasePCA


class ProbabilisticPCA(BasePCA):
    def __init__(self, args):
        """
        Initialize Probabilistic PCA with EM algorithm.

        Parameters:
        -----------
        args : object
            Should contain:
            - n_components: number of principal components
            - tol: convergence tolerance (default 1e-6)
            - max_iter: maximum iterations for EM (default 1000)
        """
        super().__init__(args)
        self.tol = args.tol if args.tol else 1e-6
        self.max_iter = args.max_iter if hasattr(args, 'max_iter') else 1000
        self.mean = None
        self.W = None  # Transformation matrix (D x M)
        self.sigma2 = None  # Noise variance
        self.M = None  # Number of components

    def fit(self, X):
        # Input validation
        if not np.all(np.isfinite(X)):
            raise ValueError("Input X contains non-finite values")
        n_samples, n_features = X.shape
        if self.n_components > min(n_samples, n_features):
            raise ValueError("n_components must not exceed min(n_samples, n_features)")

        self.M = self.n_components

        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Initialize parameters
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.W = Vt[:self.M, :].T  # Shape (n_features, self.M)
        self.sigma2 = np.var(X_centered) / 10.0

        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            M = self.W.T @ self.W + self.sigma2 * np.eye(self.M)
            M_inv = np.linalg.solve(M, np.eye(self.M))  # More stable
            XWT = X_centered @ self.W  # Shape (n_samples, self.M)
            Ez = M_inv @ XWT.T  # Shape (self.M, n_samples)
            Ezz = self.sigma2 * M_inv + Ez @ Ez.T  # Shape (self.M, self.M)

            # M-step
            # Compute W = (X^T E[z^T]) (E[zz^T])^-1
            W_T = np.linalg.solve(Ezz, Ez @ X_centered)  # Shape (self.M, n_features)
            self.W = W_T.T  # Shape (n_features, self.M)
            self.sigma2 = np.sum((X_centered - Ez.T @ self.W.T) ** 2) / (n_samples * n_features)

            # Log likelihood
            C = self.W @ self.W.T + self.sigma2 * np.eye(n_features)
            sign, logdet = np.linalg.slogdet(C)
            log_likelihood = -0.5 * (
                    n_samples * n_features * np.log(2 * np.pi)
                    + n_samples * logdet
                    + np.trace(X_centered @ np.linalg.pinv(C) @ X_centered.T)
            )

            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

        # Principal axes
        C = self.W @ self.W.T + self.sigma2 * np.eye(n_features)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        self.V = eigenvectors[:, -self.M:]  # Top M eigenvectors

    def transform(self, X, n_components=None):
        if n_components is None:
            n_components = self.M

        X_centered = X - self.mean
        M_inv = np.linalg.inv(self.W.T @ self.W + self.sigma2 * np.eye(self.M))
        return (M_inv @ self.W.T @ X_centered.T).T[:, :n_components]

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.W.T + self.mean

    def sample(self, n_samples=1):
        """
        Generate samples from the fitted PPCA model.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate

        Returns:
        --------
        samples : array-like, shape (n_samples, n_features)
            Generated samples
        """
        z = np.random.randn(n_samples, self.M)
        noise = np.random.randn(n_samples, self.W.shape[0]) * np.sqrt(self.sigma2)
        return z @ self.W.T + noise + self.mean

    def log_likelihood(self, X):
        """
        Calculate log likelihood of data under the model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to evaluate

        Returns:
        --------
        log_likelihood : float
            Log likelihood of the data
        """
        X_centered = X - self.mean
        n_samples, n_features = X.shape
        C = self.W @ self.W.T + self.sigma2 * np.eye(n_features)
        return -0.5 * (
                n_samples * n_features * np.log(2 * np.pi)
                + n_samples * np.log(np.linalg.det(C))
                + np.trace(X_centered @ pinvh(C) @ X_centered.T)
        )
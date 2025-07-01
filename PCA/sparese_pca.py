import numpy as np
from sklearn.linear_model import ElasticNet
from .base_pca import BasePCA


class SparsePCA(BasePCA):
    def __init__(self, args):
        """
        Initialize Sparse PCA with stable numerical implementation.
        """
        super().__init__(args)
        self.lambda_ = 1.0
        self.lambda1 = 1.0
        self.max_iter = args.max_iter if args.max_iter else 1000
        self.tol = args.tol if args.tol else 1e-6
        self.A = None
        self.B = None
        self.mean = None

    def fit(self, X):
        """Stable implementation of the exact 3-step algorithm"""
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        n_samples, n_features = X_centered.shape
        XTX = X_centered.T @ X_centered

        # Step 1: Initialize A via SVD
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.A = Vt[:self.n_components].T
        self.B = np.zeros_like(self.A)

        for iteration in range(self.max_iter):
            B_prev = self.B.copy()

            for j in range(self.n_components):

                enet = ElasticNet(
                    alpha=self.lambda_ + self._get_lambda1(j),
                    l1_ratio=self._get_lambda1(j) / (self.lambda_ + self._get_lambda1(j)),
                    fit_intercept=False,
                    max_iter=10000,
                    tol=1e-4
                )


                try:
                    L = np.linalg.cholesky(XTX + self.lambda_ * np.eye(n_features))
                    enet.fit(L.T, L @ self.A[:, j])
                    self.B[:, j] = enet.coef_
                except np.linalg.LinAlgError:
                    # Fallback to SVD if not positive definite
                    U, s, Vh = np.linalg.svd(XTX + self.lambda_ * np.eye(n_features))
                    sqrt_mat = U @ np.diag(np.sqrt(s)) @ Vh
                    enet.fit(sqrt_mat, sqrt_mat @ self.A[:, j])
                    self.B[:, j] = enet.coef_

            U, _, Vt = np.linalg.svd(XTX @ self.B, full_matrices=False)
            self.A = U[:, :self.n_components] @ Vt[:self.n_components, :]

            if np.linalg.norm(self.B - B_prev) < self.tol:
                break

        self.V = self.B

    def _get_lambda1(self, j):
        """Get λ1,j for component j"""
        if isinstance(self.lambda1, (list, np.ndarray)):
            return self.lambda1[j]
        return self.lambda1

    def transform(self, X, n_components=None):
        if n_components is None:
            n_components = self.n_components
        return (X - self.mean) @ self.V[:, :n_components]

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.V.T + self.mean
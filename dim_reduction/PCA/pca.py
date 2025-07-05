import torch


class PCA:
    def __init__(self, args, holdout_feature):
        """
        Initialize PCA models.

        Args:
            n_components (int, optional): Number of components to keep. If None, defaults to 5.
        """
        self.n_components = args.n_components
        self.mean = None
        self.U = None
        self.eigen_values = None
        self.V = None
        self.args = args
        self.holdout_feature = holdout_feature
        self.V_n = self.fit(holdout_feature)

    def fit(self, X):
        """
        Fit the PCA models to the data using SVD.

        Args:
            X (torch.Tensor): Input data, shape (n_samples, n_features).
        """
        # Compute mean and center the data
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean.unsqueeze(0)

        # Perform SVD on centered data
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

        self.U = U
        self.eigen_values = S.pow(2)  # Singular values to eigenvalues
        self.V = Vh.T  # Transpose to get right singular vectors

        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1])

        return self.V[:, :self.n_components]

    def transform(self, X, n_components=None):
        """
        Project data onto the principal components.

        Args:
            X (torch.Tensor): Data to transform, shape (n_samples, n_features).
            n_components (int, optional): Number of components to use. Defaults to self.n_components.

        Returns:
            torch.Tensor: Transformed data, shape (n_samples, n_components).
        """
        n = n_components if n_components is not None else self.n_components

        # Center the data
        X_centered = X - self.mean.unsqueeze(0)

        # Project onto principal components
        X_transformed = torch.mm(X_centered, self.V[:, :n])

        return X_transformed

    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space.

        Args:
            X_transformed (torch.Tensor): Reduced data, shape (n_samples, n_components).

        Returns:
            torch.Tensor: Data in original feature space, shape (n_samples, n_features).
        """
        # Project back to original space
        X_reconstructed = torch.mm(X_transformed, self.V[:, :self.n_components].T)

        # Add back the mean
        X_reconstructed += self.mean.unsqueeze(0)

        return X_reconstructed

    def get_components(self, n_components=None):
        """
        Get the principal components (eigenvectors).

        Args:
            n_components (int, optional): Number of components to return.

        Returns:
            torch.Tensor: Principal components, shape (n_features, n_components).
        """
        n = n_components if n_components is not None else self.n_components
        return self.V[:, :n]

    def fit_transform(self, cal_feature, test_feature):
        """Fit and transform the data."""

        if self.args.efficient:
            new_cal_feature = cal_feature @ self.V_n
            new_test_feature = test_feature @ self.V_n
        else:
            new_cal_feature = torch.tensor([], device="cuda")
            new_test_feature = torch.tensor([], device="cuda")
            for i in range(cal_feature.shape[0]):
                input_feature = torch.cat((self.holdout_feature, cal_feature[i].unsqueeze(0)), dim=0)
                out_feature = self.fit_then_transform(input_feature)[-1].unsqueeze(0)
                new_cal_feature = torch.cat((new_cal_feature, out_feature), dim=0)

            for i in range(test_feature.shape[0]):
                input_feature = torch.cat((self.holdout_feature, test_feature[i].unsqueeze(0)), dim=0)
                out_feature = self.fit_then_transform(input_feature)[-1].unsqueeze(0)
                new_test_feature = torch.cat((new_test_feature, out_feature), dim=0)

        return new_cal_feature, new_test_feature

    def fit_then_transform(self, X):
        """Fit and transform the data."""
        self.fit(X)
        return self.transform(X)
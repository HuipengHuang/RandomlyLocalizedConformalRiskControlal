import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import median_filter
from .base_pca import BasePCA

class RobustPCA(BasePCA):
    def __init__(self, args):
        """
        Improved Robust PCA implementation following MATLAB version
        """
        super().__init__(args)
        self.max_iter = getattr(args, 'max_iter', 1000)
        self.tol = args.tol if args.tol else 1e-5
        self.lambda_ = None
        self.mu = None
        self.rho = 1.05  # ADMM parameter update rate
        self.args = args

        # Components
        self.L = None  # Low-rank
        self.S = None  # Sparse
        self.Y = None

    def fit(self, X):
        """
        Solve RPCA via ADMM: min ||L||_* + λ||S||_1 s.t. X = L + S
        Follows MATLAB implementation more closely
        """
        M, N = X.shape
        self.lambda_ = 1.0 / np.sqrt(max(M, N))
        self.mu = 10 * self.lambda_  # Default from MATLAB version

        # Handle missing values (NaN) like MATLAB version
        unobserved = np.isnan(X)
        X[unobserved] = 0
        normX = np.linalg.norm(X, 'fro')

        # Initialize variables
        self.L = np.zeros((M, N))
        self.S = np.zeros((M, N))
        self.Y = np.zeros((M, N))

        for iter in tqdm(range(self.max_iter), desc="RPCA Progress"):
            L_prev, S_prev, Y_prev = self.L, self.S, self.Y
            # Update L (singular value thresholding)
            self.L = self.svt_operator(X - self.S + (1 / self.mu) * self.Y, 1 / self.mu)

            # Update S (shrinkage operator)
            self.S = self.shrinkage_operator(X - self.L + (1 / self.mu) * self.Y,
                                             self.lambda_ / self.mu)

            # Update Y (Lagrange multiplier)
            Z = X - self.L - self.S
            Z[unobserved] = 0  # Skip missing values like MATLAB
            self.Y = self.Y + self.mu * Z

            # Check convergence
            err = np.linalg.norm(Z, 'fro') / normX
            if (iter % 10 == 0) or (iter == 0) or (err < self.tol):
                rank_L = np.linalg.matrix_rank(self.L)
                card_S = np.count_nonzero(self.S[~unobserved])
                print(f"Iter: {iter:04d}\tError: {err:.6f}\tRank(L): {rank_L}\tCard(S): {card_S}")

            if (err < self.tol and iter > 50 and np.allclose(self.L, L_prev, self.tol)
                    and np.allclose(self.S, S_prev, self.tol)):
                print(f"Converged at iteration {iter}")
                break

            # Optional: Save intermediate results
            if (iter + 1) % 100 == 0:
                k = 0
                l_p = f"./output/sustech_video/L_{k}.mp4"
                while os.path.exists(l_p):
                    k += 1
                    l_p = f"./output/sustech_video/L_{k}.mp4"

                k = 0
                S_p = f"./output/sustech_video/S_{k}.mp4"
                while os.path.exists(S_p):
                    k += 1
                    S_p = f"./output/sustech_video/S_{k}.mp4"

                self.save_video(self.L, output_path=l_p, fps=30)
                self.save_video(self.S, output_path=S_p, fps=30)

            # Update mu
            self.mu = min(self.mu * self.rho, 1e6)  # Cap mu to prevent instability

    def svt_operator(self, X, tau):
        """Singular Value Thresholding operator (like MATLAB's Do function)"""
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        s_thresh = np.maximum(s - tau, 0)
        return U @ np.diag(s_thresh) @ Vh

    def shrinkage_operator(self, X, tau):
        """Soft thresholding operator (like MATLAB's So function)"""
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

    def save_video(self, video_array, output_path, fps=30):
        if self.args.dataset == "sustech_video":
            video_array = video_array.reshape(-1, 528, 960, 3)
            self.save_rgb_array_as_video(video_array, output_path, fps)

    def save_rgb_array_as_video(self, video_array, output_path, fps=30):
        """
        Save an RGB video array as a video file.

        Args:
            video_array: NumPy array with shape (num_frames, height, width, 3)
            output_path: Output file path (e.g., 'output.mp4')
            fps: Frames per second
        """
        height, width = video_array.shape[1:3]

        # Use MP4V codec for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Create VideoWriter object (Note: OpenCV expects BGR color order)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in video_array:
            # Ensure correct data type (uint8) and value range (0-255)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            # Convert RGB to BGR (OpenCV uses BGR format)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"Video saved to {output_path}")

    def transform(self, X_train, n_components=None):

        return self.L, self.S

    def _save_video(self, frames, output_dir="./output"):
        """Internal method to save video frames"""
        os.makedirs(output_dir, exist_ok=True)

        # Find next available filename
        i = 0
        while os.path.exists(f"{output_dir}/result_{i}.mp4"):
            i += 1

        # Write video
        height, width = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{output_dir}/result_{i}.mp4",
                              fourcc, 30, (width, height))

        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        out.release()
import os.path
import numpy as np
from .pca import PCA
from .kernel_pca import KernelPCA
from .sparese_pca import SparsePCA
from .ppca import ProbabilisticPCA
from .robust_pca import RobustPCA
import matplotlib.pyplot as plt
import cv2
import numpy as np


def get_pca(args):
    if args.pca == "pca":
        return PCA(args)
    elif args.pca == "kernel_pca":
        return KernelPCA(args)
    elif args.pca == "sparse_pca":
        return SparsePCA(args)
    elif args.pca == "ppca":
        return ProbabilisticPCA(args)
    elif args.pca == "robust_pca":
        return RobustPCA(args)
    else:
        raise NotImplementedError

def visualize_results(X_original, X_transformed, y, args, pca):
    """Plot original vs transformed data with customized styling"""
    # Set common style parameters
    if args.dataset == "make_moon" or args.dataset == "make_circle":
        title_font = {'fontsize': 25, 'fontweight': 'bold', 'color': 'black', "fontname": "Times New Roman"}
        frame_width = 3.0  # Bold frame width

        # Plot 1: Original Data
        plt.figure(figsize=(8, 5))

        # Create scatter plot with frame
        ax = plt.gca()
        ax.scatter(X_original[:, 0], X_original[:, 1], c=y, cmap='viridis', alpha=0.6)

        # Customize frame
        for spine in ax.spines.values():
            spine.set_linewidth(frame_width)

        # Add title at bottom
        ax.set_title("Original Data", fontdict=title_font, y=-0.12, pad=-10)

        plt.tight_layout()

        if args.save == "True":
            os.makedirs("./output", exist_ok=True)
            save_path = f"./output/{args.dataset}/original_data_{args.n_components}.pdf"

            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()

        # Plot 2: Transformed Data
        plt.figure(figsize=(8, 4))
        ax = plt.gca()

        if X_transformed.shape[1] == 1:
            ax.scatter(X_transformed[:, 0], np.zeros_like(X_transformed[:, 0]),
                       c=y, cmap='viridis', alpha=0.6,
                       s=200)  # Larger points (no frame)
        else:
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                       c=y, cmap='viridis', alpha=0.6,
                       s=400)  # Larger points (no frame)

        # Customize frame
        for spine in ax.spines.values():
            spine.set_linewidth(frame_width)

        # Add title at bottom
        if args.pca == "pca":
            ax.set_title(f"Naive PCA (n={args.n_components})",
                         fontdict=title_font, y=-0.12, pad=-20)
        elif args.pca == "kernel_pca":
            ax.set_title(f"Kernel PCA (kernel={args.kernel}, γ={args.gamma}, n={args.n_components})",
                         fontdict=title_font, y=-0.12, pad=-50)
        else:
            raise NotImplementedError

        plt.tight_layout()

        if args.save == "True":
            i = 0
            save_path = f"./output/{args.dataset}/{args.kernel}_{args.gamma}_{args.n_components}_{i}.pdf"
            while os.path.exists(save_path):
                i += 1
                save_path = f"./output/{args.dataset}/{args.kernel}_{args.gamma}_{args.n_components}_{i}.pdf"
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()
    else:
        X_transformed = pca.inverse_transform(X_transformed)
        if args.dataset == "yaleB":
            plt.imshow(X_original[1].reshape(243, 320), cmap='gray')
            plt.axis('off')
            plt.tight_layout()

            if args.save == "True":
                os.makedirs("./output", exist_ok=True)
                save_path = f"./output/{args.dataset}/{args.pca}_original_data_{args.n_components}.pdf"

                plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()

            plt.imshow(X_transformed[1].reshape(243, 320), cmap='gray')
            plt.axis('off')
            plt.tight_layout()

            if args.save == "True":
                os.makedirs("./output", exist_ok=True)
                save_path = f"./output/{args.dataset}/{args.pca}_transformed_{args.n_components}.pdf"

                plt.savefig(save_path, format='pdf', bbox_inches='tight')
            plt.show()
            plt.show()

        elif args.dataset == "ext_yaleB":
            fig, ax = plt.subplots()
            ax.imshow(X_original[1].reshape(168, 192), cmap='gray')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
            plt.savefig(f"./output/ext_yaleB/{args.pca}_origin.png")
            plt.show()

            fig, ax = plt.subplots()
            ax.imshow(X_transformed[1].reshape(168, 192), cmap='gray')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(f"./output/ext_yaleB/{args.pca}_L.png")
            plt.show()


def save_grayscale_array_as_video(video_array, output_path, fps=30):
    """
        Save a grayscale video array as a video file after denoising.

        Args:
            video_array: NumPy array of shape (num_frames, height, width), dtype float or uint8.
            output_path: Path to save the output video.
            fps: Frames per second for the output video.
        """
    # Validate input
    if not isinstance(video_array, np.ndarray) or len(video_array.shape) != 3:
        raise ValueError(
            f"Input 'video_array' must be a 3D NumPy array (num_frames, height, width). Got shape: {video_array.shape}")

    # Normalize and convert to uint8 if necessary

    video_array = video_array.astype(np.uint8)

    # Log input details
    print(
        f"video_array shape: {video_array.shape}, dtype: {video_array.dtype}, min: {video_array.min()}, max: {video_array.max()}")

    # Apply Bilateral Filter denoising to all frames
    try:
        denoised_array = np.array([
            cv2.bilateralFilter(f, d=10, sigmaColor=30, sigmaSpace=30)
            for f in video_array
        ], dtype=np.uint8)
    except cv2.error as e:
        raise RuntimeError(f"Bilateral Filter denoising failed: {str(e)}")

    # Verify output shape
    print(f"denoised_array shape: {denoised_array.shape}, dtype: {denoised_array.dtype}")

    # Initialize video writer
    height, width = denoised_array.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame in denoised_array:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")


def save_rgb_array_as_video(video_array, output_path, fps=30):
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

def save_video(video_array, output_path, fps=30):
    if video_array.ndim == 4:
        save_rgb_array_as_video(video_array, output_path, fps)
    else:
        save_grayscale_array_as_video(video_array, output_path, fps)



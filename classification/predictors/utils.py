import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os


def plot_feature_distance(args, cal_feature, test_feature, cal_target=None, test_target=None):
    if cal_target is not None and test_target is not None:
        plot_class_distance(cal_feature, test_feature, cal_target, test_target)

    d = test_feature.shape[1]
    cal_distance = torch.zeros(size=(test_feature.shape[0], cal_feature.shape[0]), device="cuda")
    for i in range(test_feature.shape[0]):
        cal_distance[i] = torch.sum(((cal_feature - test_feature[i]) / d) ** 2, dim=-1) ** (0.5)
    min_val = torch.min(cal_distance)
    max_val = torch.max(cal_distance)
    normalized_distance = (cal_distance - min_val) / (max_val - min_val + 1e-8)

    plt.hist(normalized_distance.view(-1).cpu().numpy(), bins=100)
    i = 0
    path = f"./plot_results/distance{i}.pdf"

    while os.path.exists(path):
        i += 1
        path = f"./plot_results/distance{i}.pdf"

    plt.savefig(path)
    plt.show()


def plot_class_distance(cal_feature, test_feature, cal_target, test_target):
    feature = torch.cat((cal_feature, test_feature), dim=0)
    target = torch.cat((cal_target, test_target), dim=0)

    feature_sqnorms = torch.sum(feature ** 2, dim=1)

    # Compute dot products between all vectors [n, n]
    feature_dot = feature @ feature.T

    # Compute squared distances: ||x_i - x_j||^2 = ||x_i||^2 - 2<x_i,x_j> + ||x_j||^2
    # Using broadcasting: x_sqnorms[:, None] is [n,1] and x_sqnorms[None, :] is [1,n]
    dist_sq = feature_sqnorms[:, None] - 2 * feature_dot + feature_sqnorms[None, :]

    # Handle numerical errors from negative values (due to floating point)
    dist_sq = torch.clamp(dist_sq, min=0.0)

    distances = torch.sqrt(dist_sq)
    distance_of_same_class = []
    distance_of_different_class = []
    unique_classes = torch.unique(target)

    for cls in unique_classes:
        # Create mask for current class
        class_mask = (target == cls)
        class_indices = torch.where(class_mask)[0]

        # Same-class distances (upper triangular to avoid duplicates)
        triu_indices = torch.triu_indices(len(class_indices), len(class_indices), offset=1)
        same_pairs = distances[class_indices[triu_indices[0]], class_indices[triu_indices[1]]]
        distance_of_same_class.append(same_pairs.view(-1))

        # Different-class distances
        other_indices = torch.where(target != cls)[0]
        if len(other_indices) > 0:
            # Get all cross-class pairs
            i, j = torch.meshgrid(class_indices, other_indices)
            diff_pairs = distances[i.flatten(), j.flatten()]
            distance_of_different_class.append(diff_pairs.view(-1))

    distance_of_same_class = torch.cat(distance_of_same_class, dim=0)
    distance_of_different_class = torch.cat(distance_of_different_class, dim=0)

    max_value = max(torch.max(distance_of_same_class).item(), torch.max(distance_of_different_class).item())
    min_value = min(torch.min(distance_of_same_class).item(), torch.min(distance_of_different_class).item())

    normalized_same = (distance_of_same_class - min_value) / (max_value - min_value)
    normalized_diff = (distance_of_different_class - min_value) / (max_value - min_value)

    plt.figure(figsize=(10, 6))

    # Plot histograms with different colors and transparency
    plt.hist(normalized_same.cpu().numpy(),
             bins=100,
             alpha=0.6,
             color='blue',
             density=True,
             label='Same Class')

    plt.hist(normalized_diff.cpu().numpy(),
             bins=100,
             alpha=0.6,
             color='red',
             density=True,
             label='Different Class')

    # Add plot decorations
    plt.xlabel('Normalized Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Normalized Distances by Class Relationship', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    i = 0
    path = f"./plot_results/class_distance{i}.pdf"

    while os.path.exists(path):
        i += 1
        path = f"./plot_results/class_distance{i}.pdf"

    plt.savefig(path)

def plot_histogram(class_coverage_numpy, alpha, args):

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot histogram
    ax.hist(class_coverage_numpy, bins=40, alpha=0.7, density=True)

    # Customize plot
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Density')
    #ax.axvline(
     #   x=torch.mean(class_coverage_tensor).item(),
      #  c='black',
       # linestyle='--',
        #alpha=0.7,
        #label=f'Empirical: {torch.mean(class_coverage_tensor).item():.3f}'  # Format to 2 decimal places
    #)

    ax.axvline(x=1 - alpha, c='#999999', linestyle='--', alpha=0.7, label=f'1-α={1-alpha}')
    ax.locator_params(axis='x', nbins=10)
    sns.despine(top=True, right=True)

    # Add legend and save
    ax.legend()
    plt.tight_layout()

    if args.output_dir:
        # Create base filename
        if args.kernel_function != "naive":
            base_name = f"{args.num_runs}_{str(alpha).replace('.', '_')}_{args.kernel_function}_{args.dataset}"
            if args.pca is not None:
                base_name = str(args.pca) + f"_{args.n_components}_" + base_name
            if args.vae is not None:
                base_name = str(args.vae) + f"_" + base_name
            if args.efficient_calibration_size is not None:
                base_name = str(args.efficient_calibration_size) + f"_" + base_name
        else:
            base_name = f"{str(alpha).replace('.', '_')}_{args.dataset}"
        base_name = f"cls_{base_name}"
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Find available filename
        counter = 1
        filename = f"{base_name}.pdf"
        while os.path.exists(os.path.join(args.output_dir, filename)):
            filename = f"{base_name}_{counter}.pdf"
            counter += 1

        # Save the figure
        plt.savefig(os.path.join(args.output_dir, filename))

    plt.show()
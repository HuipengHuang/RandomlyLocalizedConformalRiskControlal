import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os

def plot_histogram(class_coverage_tensor, alpha, args):
    risk_data = class_coverage_tensor.cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot histogram
    ax.hist(risk_data, bins=40, alpha=0.7, density=True)

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

    ax.axvline(x=alpha, c='#999999', linestyle='--', alpha=0.7, label=f'1-α={1-alpha}')
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
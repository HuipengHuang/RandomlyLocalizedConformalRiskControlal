import os
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from utils.dataloader import PolypDataset


def jpg2png(folder_path):
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            # Construct full file paths
            jpg_path = os.path.join(folder_path, filename)
            png_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.png')

            # Convert and save
            try:
                img = Image.open(jpg_path)
                img.save(png_path)
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

def get_dataset(args):
    if args.dataset == "all":
        ds_name_list = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', "HyperKvasir",
                        "polypgen_positive"]
    elif args.dataset == "Kvasirfamily":
        ds_name_list = ["Kvasir", "HyperKvasir"]
    else:
        ds_name_list = [args.dataset]
    test_ds_list = []
    for _data_name in ds_name_list:
        data_path = './data/TestDataset/{}/'.format(_data_name)
        save_path = './results/PraNet/{}/'.format(_data_name)

        os.makedirs(save_path, exist_ok=True)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        # test_loader = test_dataset(image_root, gt_root, args.testsize)
        test_ds_list.append(PolypDataset(image_root, gt_root, args.testsize))
    cal_test_dataset = ConcatDataset(test_ds_list)

    holdout_dataset = None
    if args.holdout_dataset is not None:
        data_path = './data/TestDataset/{}/'.format(args.holdout_dataset)
        save_path = './results/PraNet/{}/'.format(args.holdout_dataset)

        os.makedirs(save_path, exist_ok=True)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)

        holdout_dataset = PolypDataset(image_root, gt_root, args.testsize)

    return cal_test_dataset, holdout_dataset

def rlcrc(cal_res, cal_gt, model, test_loader, kernel_function, args ,alpha=0.1, B=1,):
    max_iters = 1000
    tol = 1e-8
    _lambda = None


    fdr_tensor = torch.tensor([], device="cuda")
    size = 0
    for idx, (image, gt, origin_image_path) in tqdm(enumerate(test_loader), desc="Testing"):
        image = image.cuda()
        gt = gt.cuda()

        # Forward pass
        res5, res4, res3, res2 = model(image)
        res = res2
        # Get dimensions
        bsz, c, h, w = gt.shape

        upsample = nn.Upsample(size=(h, w), mode='bilinear')
        res = upsample(res)
        res = (res / args.T).sigmoid().data

        #min_values = torch.min(res, dim=-1, keepdim=True).values
        #max_values = torch.max(res, dim=-1, keepdim=True).values
        #res = (res - min_values) / (max_values - min_values + 1e-6)

        weight = kernel_function.get_weight(cal_res.view(cal_res.shape[0], -1), res.view(res.shape[0], -1))

        for j in range(bsz):
            low = 0
            high = 1
            _lambda = None
            for _ in range(max_iters):
                _lambda = (low + high) / 2

                pred = (cal_res >= 1 - _lambda).to(torch.int)
                R_n_list = get_fnr_list(pred, cal_gt)
                risk = weight[j, :-1] @ R_n_list + (weight[j, -1] * B)

                if risk == alpha:
                    break
                elif risk < alpha:
                    high = _lambda
                else:
                    low = _lambda

                if high - low < tol:
                    break
            _lambda = low
            pred = (res[j] >= 1 - _lambda).to(torch.int)

            size += torch.sum(pred).item()
            fdr = get_fnr_list(pred, gt[j])


            fdr_tensor = torch.cat((fdr_tensor, fdr), dim=0)

    result_dict = {"MeanFDR": torch.mean(fdr_tensor).item(),
                   "Var": torch.var(fdr_tensor).item(),
                   "Avg_size": size / fdr_tensor.shape[0]}

    return result_dict, fdr_tensor


def crc(cal_res, cal_gt, model, test_loader, args, alpha=0.1, B=1):
    max_iters = 1000
    tol = 1e-8
    n = cal_res.shape[0]
    low = 0
    high = 1
    _lambda = None

    for _ in range(max_iters):
        _lambda = (low + high) / 2

        pred = (cal_res >= 1 - _lambda).to(torch.int)
        R_n = torch.mean(get_fnr_list(pred, cal_gt))
        risk = (n / (n + 1)) * R_n + (B / (n + 1))


        if risk == alpha:
            break
        elif risk < alpha:
            high = _lambda
        else:
            low = _lambda

        if high - low < tol:
            break
    _lambda = low
    fdr_tensor = torch.tensor([], device="cuda")
    size = 0
    for idx, (image, gt, origin_image_path) in tqdm(enumerate(test_loader), desc="Testing"):
        image = image.cuda()
        gt = gt.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        # Get dimensions
        bsz, c, h, w = gt.shape

        upsample = nn.Upsample(size=(h, w), mode='bilinear')
        res = upsample(res)
        res = (res / args.T).sigmoid().data

        #min_values = torch.min(res, dim=-1, keepdim=True).values
        #max_values = torch.max(res, dim=-1, keepdim=True).values
        #res = (res - min_values) / (max_values - min_values + 1e-6)

        pred = (res >= 1 - _lambda).to(torch.int)

        size += torch.sum(pred).item()
        fdr = get_fnr_list(pred, gt)

        fdr_tensor = torch.cat((fdr_tensor, fdr), dim=0)

    result_dict = {"MeanFDR": torch.mean(fdr_tensor).item(),
                   "Var":torch.var(fdr_tensor).item(),
                   "Avg_size": size / fdr_tensor.shape[0]}

    return result_dict, fdr_tensor



def get_fnr_list(pred_masks, gt_masks):
    # Input shapes: (B, H, W)
    sub = torch.sum( (pred_masks * gt_masks).view(-1 ,352*352), dim=-1)
    fdr_list = 1 - sub / (torch.sum(gt_masks.view(-1, 352*352), dim=-1) + 1e-6)

    return fdr_list


def plot_histogram(fdr_tensor, alpha, args):
    risk_data = fdr_tensor.cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot histogram
    ax.hist(risk_data, bins=40, alpha=0.7, density=True)

    # Customize plot
    ax.set_xlabel('Risk')
    ax.set_ylabel('Density')
    ax.axvline(x=alpha, c='#999999', linestyle='--', alpha=0.7, label=f'α={alpha}')
    ax.locator_params(axis='x', nbins=10)
    sns.despine(top=True, right=True)

    # Add legend and save
    ax.legend()
    plt.tight_layout()

    if args.output_dir:
        # Create base filename
        if args.kernel_function != "naive":
            base_name = f"{str(alpha).replace('.', '_')}_{args.kernel_function}_{args.dataset}_risk_histogram"
        else:
            base_name = f"{str(alpha).replace('.', '_')}_{args.dataset}_risk_histogram"
            if args.pca is not None:
                base_name = str(args.pca) + base_name
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

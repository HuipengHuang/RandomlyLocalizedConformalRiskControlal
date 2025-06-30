import os
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

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

def rlcrc(cal_res, cal_gt, model, test_loader, kernel_function, args ,alpha=0.1, B=1,):
    max_iters = 1000
    tol = 1e-8
    _lambda = None


    fdr_tensor = torch.tensor([], device="cuda")
    size = 0
    for idx, (image, gt, origin_image_path) in enumerate(test_loader):
        image = image.cuda()
        gt = gt.cuda()

        # Forward pass
        res5, res4, res3, res2 = model(image)
        res = res2
        # Get dimensions
        bsz, c, h, w = gt.shape

        upsample = nn.Upsample(size=(h, w), mode='bilinear')
        res = upsample(res)
        res = (res).sigmoid().data
        weight = kernel_function.get_weight(cal_res, res)

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

    result_dict = {"Mean FDR": torch.mean(fdr_tensor).item(),
                   "Var": torch.var(fdr_tensor).item(),
                   "Avg_size": size / fdr_tensor.shape[0]}
    if args.plot:
        plot_histgram(fdr_tensor, alpha, args)
    return result_dict


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
    for idx, (image, gt, origin_image_path) in enumerate(test_loader):
        image = image.cuda()
        gt = gt.cuda()

        # Forward pass
        res5, res4, res3, res2 = model(image)
        res = res2
        # Get dimensions
        bsz, c, h, w = gt.shape

        upsample = nn.Upsample(size=(h, w), mode='bilinear')
        res = upsample(res)
        res = (res).sigmoid().data
        pred = (res >= 1 - _lambda).to(torch.int)

        size += torch.sum(pred).item()
        fdr = get_fnr_list(pred, gt)

        fdr_tensor = torch.cat((fdr_tensor, fdr), dim=0)

    result_dict = {"Mean FDR": torch.mean(fdr_tensor).item(),
                   "Var":torch.var(fdr_tensor).item(),
                   "Avg_size": size / fdr_tensor.shape[0]}
    if args.plot:
        plot_histgram(fdr_tensor, alpha, args)
    return result_dict




def get_fnr_list(pred_masks, gt_masks):
    # Input shapes: (B, H, W)

    sub = torch.sum( (pred_masks * gt_masks).view(-1 ,352*352), dim=-1)
    fdr_list = 1 - sub / (torch.sum(gt_masks.view(-1, 352*352), dim=-1) + 1e-6)

    return fdr_list

def fnr_weight_avg(pred_masks, gt_masks, weight, B=1):
    TP = torch.sum((pred_masks == 1) & (gt_masks == 1), dim=(2, 3)).float()
    FP = torch.sum((pred_masks == 1) & (gt_masks == 0), dim=(2, 3)).float()
    fdr = TP / (TP + FP)
    return weight[:-1] @ fdr + weight[-1] * B


def plot_histgram(fdr_tensor, alpha, args):
    #fdr_tensor = fdr_tensor[fdr_tensor <= 0.3]
    #plt.hist(fdr_tensor.cpu().numpy(), bins=50, density=True)
    #plt.xlabel("Risk")
    #plt.ylabel("Density")
    #plt.show()
    risk_data = fdr_tensor[fdr_tensor<=0.3].cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot histogram
    ax.hist(risk_data, bins=20, alpha=0.7, density=True)

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
        if args.kernel_function != "naive":
            filename = f"{str(alpha).replace('.', '_')}_{args.kernel_function}_risk_histogram.pdf"
        else:
            filename = f"{str(alpha).replace('.', '_')}_risk_histogram.pdf"
        plt.savefig(f"{args.output_dir}/{filename}")
    plt.show()

import os
from PIL import Image
import torch
import torch.nn as nn

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



def crc(cal_res, cal_gt, model, test_loader, alpha=0.1, B=1):
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
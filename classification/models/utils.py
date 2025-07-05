import os
from .model import get_model
import torch
import torchvision.models as models
def build_model(args, num_classes):
    model_type= args.model
    pretrained = args.pretrained

    if model_type == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet101":
        net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet152":
        net = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet161":
        net = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnext50":
        net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if args.dataset != "imagenet":
        if hasattr(net, "fc"):
            net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
        else:
            net.classifier = torch.nn.Linear(net.classifier.in_features, num_classes)

    if args.load == "True":
        load_model(args, net)
    net = net.to("cuda")

    model = get_model(net, args)
    return model

def load_model(args, net):
        p = f"./data/{args.dataset}_{args.model}{0}net.pth"

        if args.model == "resnet50":
            net.resnet.load_state_dict(torch.load(p))
        else:
            net.load_state_dict(torch.load(p))

def save_model(args, net):
    i = 0
    while (True):
        p = f"./data/{args.dataset}_{args.model}{i}net.pth"

        if os.path.exists(p):
            i += 1
            continue
        torch.save(net.state_dict(), p)
        break
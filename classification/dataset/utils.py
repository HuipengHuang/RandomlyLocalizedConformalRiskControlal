import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10

def build_dataloader(args):
    dataset_name = args.dataset

    if dataset_name == "cifar10":
        num_classes = 10
        cal_test_dataset = CIFAR10(root='./data/dataset', train=False, download=False,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name == "cifar100":
        num_classes = 100

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        cal_test_dataset = CIFAR100(root='/mnt/sharedata/ssd3/common/datasets/cifar-100-python', download=True, train=False,
                                 transform=val_transform)

    elif dataset_name == "imagenet":
        num_classes = 1000

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        cal_test_dataset = torchvision.datasets.ImageFolder(
            root="/mnt/sharedata/ssd3/common/datasets/imagenet/images/val",
            transform=val_transform
        )

    cal_size = int(len(cal_test_dataset) * args.cal_ratio)
    test_size = len(cal_test_dataset) - cal_size
    holdout_dataset, cal_dataset, test_dataset = random_split(cal_test_dataset, [1000, cal_size, test_size - 1000])
    args.cal_size = len(cal_dataset)

    args.num_classes = num_classes
    holdout_dataloader = DataLoader(holdout_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    cal_dataloader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return holdout_dataloader, cal_dataloader, test_dataloader, num_classes
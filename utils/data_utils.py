import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

# Notes for Zhang: This is the first time I do a transformer. I add all the "Notes:" comments only for myself to interpret code faster.

# Notes: The original code imported all the necessary classes from torchvision.datasets, such as CIFAR10 and CIFAR100. I added support for ImageFolder to make it easier to see which classes and functionalities are being imported, and most importantly to highlight the changes I made. / 原始代码从 torchvision.datasets 统一导入了所有需要的类，例如 CIFAR10 和 CIFAR100。新增了对 ImageFolder 的使用，为了让读代码的人更容易理解哪些类和功能被引入，以及突出修改点
from torchvision.datasets import ImageFolder

import os

logger = logging.getLogger(__name__)

# New Function Block: 检查数据集完整性 / Check dataset integrity
def check_dataset_integrity(dataset_path, required_classes):
    """
    检查数据集是否完整，包括路径是否存在、类别是否存在以及每个类别是否有足够的样本。
    Check if the dataset is complete, including whether the path exists,
    whether the required classes exist, and if each class has enough samples.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # 检查每个类别是否存在 / Check if each class exists
    for cls in required_classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.exists(class_path) or not os.listdir(class_path):
            raise ValueError(f"Class '{cls}' is missing or empty in {dataset_path}")
    
    logger.info(f"Dataset integrity check passed for {dataset_path}")

# Notes: 获取训练和验证数据的加载器，并根据数据集类型和路径设置加载逻辑。 / Get data loaders for training and validation datasets, adapting to dataset type and paths.
def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Notes: 数据增强和预处理 / Data augmentation and preprocessing
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Notes: 数据集选择逻辑 / Dataset selection logic
    
    # Adaptation: 只保留 hymenoptera 数据集 / Keep only hymenoptera dataset
    train_path = "./data/hymeno/hymenoptera_data/train"
    val_path = "./data/hymeno/hymenoptera_data/val"
    required_classes = ["ants", "bees"]

    check_dataset_integrity(train_path, required_classes)
    check_dataset_integrity(val_path, required_classes)

    trainset = ImageFolder(root=train_path, transform=transform_train)
    testset = ImageFolder(root=val_path, transform=transform_test) if args.local_rank in [-1, 0] else None

     # Adaptation: Original Code. Not deleted in case it is needed.
    """    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None 
    """

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset) if testset is not None else None
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

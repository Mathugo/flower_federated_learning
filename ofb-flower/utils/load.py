import client.src.pipeline as pl
from torch.utils.data import DataLoader, random_split
from client.src.pipeline.transforms import resnet18_transform, mobile_ViT_transform
import torchvision, sys, os
from typing import Dict
sys.path.append("..")
from models import *
from pathlib import Path
from typing import Tuple
from torchsummary import summary 
import torch.nn as nn

DATA_ROOT = Path("data")

def load_classify_datasets(dataset_dir: str, num_clients: int, transforms: Dict[str, torchvision.transforms.Compose]):
    d_train, d_test = load_classify_dataset(dataset_dir=dataset_dir, transforms=transforms)
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = int(len(d_train) // num_clients)
    lengths = [partition_size] * num_clients
    print("Len d_train {}".format(len(d_train)))
    print("Partition size {} Lengths {}".format(partition_size, lengths))
    datasets = random_split(d_train, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
        
    testloader = DataLoader(d_test, batch_size=32)
    return trainloaders, valloaders, testloader

def load_classify_dataset(dataset_dir: str, transforms: Dict[str, torchvision.transforms.Compose]) -> Tuple[pl.ClassifyDataset, pl.ClassifyDataset]:
    train = os.path.join(dataset_dir, "train")
    test = os.path.join(dataset_dir, "valid")
    d_train = pl.ClassifyDataset(train, transform=transforms["train"])
    d_test = pl.ClassifyDataset(test, transform=transforms["test"])
    return d_train, d_test

def load_model(model_name: str, n_classes: int=3, input_shape: Tuple[int, int]= (3, 224, 224)) -> Tuple[nn.Module, Dict[str, torchvision.transforms.Compose]]:
    """ return a pytorch model and its associated transformation for training and testing """
    config = None
    if model_name == "HugoNet":
        config = (HugoNet(), resnet18_transform)
    elif model_name == "ResNet18":
        config = (ResNet18(n_classes=n_classes), resnet18_transform)
    elif model_name == "ViT":
        config = (ViT_B_16(n_classes=n_classes), resnet18_transform)
        #return ViT(n_classes=n_classes, img_size=img_size, emb_size=98)
    if config == None:
        raise NotImplementedError(f"model {model_name} is not implemented")
    summary(config[0], input_shape)
    return config

def load_scripted_model(filename: str) -> nn.Module:
    return torch.jit.load(filename)  
# pylint: disable=unused-argument
def load_cifar(download=True) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=True, download=download, transform=transform
    )
    testset = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=False, download=download, transform=transform
    )
    return trainset, testset

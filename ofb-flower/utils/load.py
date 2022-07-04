import client.src.pipeline as pl
from torch.utils.data import DataLoader, random_split
from client.src.pipeline.transforms import hugonet_transform, mobile_ViT_transform
import torchvision, sys, os
from typing import Dict
sys.path.append("..")
from models import *
from pathlib import Path
from typing import Tuple
from torchsummary import summary 
import torch.nn as nn
from torch.utils.data import ConcatDataset

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

def load_classify_dataset(dataset_dir: str, transforms: Dict[str, torchvision.transforms.Compose], data_augmentation: bool=False) -> Tuple[pl.ClassifyDataset, pl.ClassifyDataset]:
    train = os.path.join(dataset_dir, "train")
    test = os.path.join(dataset_dir, "valid")
    d_train = pl.ClassifyDataset(train, transform=transforms["train"])
    d_test = pl.ClassifyDataset(test, transform=transforms["test"])
    if data_augmentation:
        d_aug = pl.ClassifyDataset(train, transform=transforms["aug"])
        d_train = ConcatDataset([d_train, d_aug])
    return d_train, d_test

def load_model(model_name: str, onServer: bool, n_classes: int=3, input_shape: Tuple[int, int]= (3, 224, 224)) -> Tuple[FederatedModel, Dict[str, torchvision.transforms.Compose]]:
    """Return a pytorch model and its associated transformation for training and testing """
    config = None
    # CNN Models
    if model_name == "HugoNet":
        config = (HugoNet(onServer), hugonet_transform)
    # Resnet models
    elif model_name == "ResNet18":
        config = (resnet18(3, n_classes), hugonet_transform)
    elif model_name == "ResNet34":
        config = (resnet34(3, n_classes), hugonet_transform)
    elif model_name == "ResNet50":
        config = (resnet50(3, n_classes), hugonet_transform)
    elif model_name == "ResNet101":
        config = (resnet101(3, n_classes), hugonet_transform)
    elif model_name == "ResNet152":
        config = (resnet34(3, n_classes), hugonet_transform)
    # Transformers
    elif model_name == "ViT":
        raise NotImplementedError(f"model ViT is not implemented")
    if config == None:
        raise NotImplementedError(f"model {model_name} is not implemented")
    summary(config[0], input_shape)
    return config

def load_scripted_model(filename: str) -> nn.Module:
    return torch.jit.load(filename)  

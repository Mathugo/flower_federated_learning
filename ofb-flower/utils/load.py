import client.src.pipeline as pipe
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
import mlflow
from .utils import get_weights, set_weights

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

def load_classify_dataset(dataset_dir: str, transforms: Dict[str, torchvision.transforms.Compose], data_augmentation: bool=False) -> Tuple[pipe.ClassifyDataset, pipe.ClassifyDataset]:
    train = os.path.join(dataset_dir, "train")
    test = os.path.join(dataset_dir, "valid")
    d_train = pipe.ClassifyDataset(train, transform=transforms["train"])
    d_test = pipe.ClassifyDataset(test, transform=transforms["test"])
    if data_augmentation:
        d_aug = pipe.ClassifyDataset(train, transform=transforms["aug"])
        d_train = ConcatDataset([d_train, d_aug])
    return d_train, d_test

def load_pytorch_mlflow_model(registered_name: str, previous_model: FederatedModel, version: int=None) -> FederatedModel:
    #TODO put staging mode
        print(f"[MODEL] Loading latest mlflow model for {previous_model.Basename} ..", file=sys.stderr)
        try:
            if version == None:
                model = mlflow.pytorch.load_model(
                    model_uri=f"models:/{registered_name}/latest"
                )
            elif version != None:
                model = mlflow.pytorch.load_model(
                    model_uri=f"models:/{registered_name}/{version}"
                )
            print(f"[MODEL] Done ! New model {model}", file=sys.stderr)
            print(f"[MODEL] Transferring weights to our model instance ..", file=sys.stderr)
            previous_model.set_weights(get_weights(model))
            print(f"[MODEL] Done !", file=sys.stderr)
            return previous_model
        except Exception as e:
            print(f"[MODEL] Error exception {e}", file=sys.stderr)
            return previous_model

def load_model(model_name: str, onServer: bool, n_classes: int=3, input_shape: Tuple[int, int]= (3, 224, 224), load_mlflow_model: bool=False, registered_model_name: str=None) -> Tuple[FederatedModel, Dict[str, torchvision.transforms.Compose]]:
    """Return a pytorch model and its associated transformation for training and testing """
    config = None
    # CNN Models
    if model_name == "hugonet":
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
    
    if load_mlflow_model and registered_model_name != None:
        model, tf = config
        model = load_pytorch_mlflow_model(registered_model_name, model)
        config = (model, tf)

    summary(config[0], input_shape)        
    return config

def load_scripted_model(filename: str) -> nn.Module:
    return torch.jit.load(filename)  

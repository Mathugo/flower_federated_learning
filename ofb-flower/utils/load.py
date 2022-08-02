import torchvision, sys, os
from .transforms import hugonet_transform, mobile_ViT_transform, tfm
from .dataset import ClassifyDataset
from torch.utils.data import DataLoader, random_split
from typing import Dict
from models.base_models import FederatedModel
from models import HugoNet, ResHugoNet, HubModel
from pathlib import Path
from typing import Tuple
from torchsummary import summary 
import torch.nn as nn
import torch
from torch.utils.data import ConcatDataset
import mlflow
from .utils import get_weights
from models.config import TrainingConfig
from timm.data import create_transform
import torch.nn.functional as F

DATA_ROOT = Path("data")

def load_classify_datasets(dataset_dir: str, num_clients: int, transforms: Dict[str, torchvision.transforms.Compose]):
    """
    `load_classify_datasets` takes a dataset directory, a number of clients, and a dictionary of
    transforms, and returns a list of training dataloaders, a list of validation dataloaders, and a test
    dataloader
    
    Args:
      dataset_dir (str): The directory where the dataset is stored.
      num_clients (int): The number of clients in the federated learning simulation.
      transforms (Dict[str, torchvision.transforms.Compose]): A dictionary of transforms to apply to the
    dataset.
    """
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

def load_classify_dataset(dataset_dir: str, transforms: Dict[str, torchvision.transforms.Compose], data_augmentation: bool=False) -> Tuple[ClassifyDataset, ClassifyDataset]:
    """
    It loads the dataset and applies the transforms
    
    Args:
      dataset_dir (str): the directory where the dataset is stored.
      transforms (Dict[str, torchvision.transforms.Compose]): a dictionary of
    torchvision.transforms.Compose objects, one for training and one for testing.
      data_augmentation (bool): Whether to use data augmentation or not. Defaults to False
    
    Returns:
      A tuple of two datasets, one for training and one for testing.
    """
    
    train = os.path.join(dataset_dir, "train")
    test = os.path.join(dataset_dir, "valid")
    d_train = ClassifyDataset(train, transform=transforms["train"])
    d_test = ClassifyDataset(test, transform=transforms["test"])
    if data_augmentation:
        print("[DATASET] Data augmented !", file=sys.stderr)
        d_aug = ClassifyDataset(train, transform=create_transform(224, is_training=True, auto_augment='rand-m7-n4-mstd0.5'))
        d_train = ConcatDataset([d_train, d_aug])
    
    print(f"[DATASET] {len(d_train)} for training and {len(d_test)} for testing ", file=sys.stderr)
    return d_train, d_test

def load_pytorch_mlflow_model(registered_name: str, previous_model: FederatedModel, version: int=None, stage: str=None) -> FederatedModel:
    """
    > Loads a model from mlflow, and transfers the weights to our model instance
    
    Args:
      registered_name (str): The name of the model in the mlflow registry
      previous_model (FederatedModel): The model instance that we want to load the weights into.
      version (int): The version of the model to load. If None, the latest version will be loaded.
      stage (str): The stage of the model. This is a string that can be used to group models together.
    
    Returns:
      The model with the weights loaded from the mlflow registry.
    """
    print(f"[MODEL] Loading mlflow model for {previous_model.Basename} ..", file=sys.stderr)
    try:
        if version == None and stage != None:
            print(f"[MODEL] Loading latest version of {registered_name} model in registry with stage {stage}", file=sys.stderr)
            model = mlflow.pytorch.load_model(
                model_uri=f"models:/{registered_name}/{stage}"
            )
        elif version != None and stage == None:
            model = mlflow.pytorch.load_model(
                model_uri=f"models:/{registered_name}/{version}"
            )
        elif version == None and stage == None:
            model = mlflow.pytorch.load_model(
                model_uri=f"models:/{registered_name}/latest"
            )
        print(f"[MODEL] {registered_name} Loaded from mlflow !", file=sys.stderr)
        print(f"[MODEL] Transferring weights to our model instance ..", file=sys.stderr)
        previous_model.set_weights(get_weights(model))
        print(f"[MODEL] Done !", file=sys.stderr)
        return previous_model
        
    except Exception as e:
        print(f"[MODEL] Error exception {e}, loading vanilla model ..", file=sys.stderr)
        if previous_model.HasPretrainedWeights:
            previous_model.load_pretrained_weights()
        return previous_model

def load_model(trainconfig: TrainingConfig, onServer: bool, load_mlflow_model: bool=False, registered_model_name: str=None, model_stage: str=None) -> Tuple[FederatedModel, Dict[str, torchvision.transforms.Compose]]:
    """Return a pytorch model and its associated transformation for training and testing """
    
    config = None
    if trainconfig.model == "hugonet":
        config = (HugoNet(onServer, trainconfig), hugonet_transform)
    elif trainconfig.model == "reshugonet":
        config = (ResHugoNet(onServer, trainconfig), hugonet_transform)
    if config == None:
        print(f"model {trainconfig.model} is not implemented, trying to catch it on the hub ..", file=sys.stderr)
        #try:
        config = (HubModel(onServer, trainconfig), hugonet_transform)
    #    except:
    #        raise NotImplementedError(f"model {trainconfig.model} is not implemented")

    if load_mlflow_model != False and registered_model_name != None:
        model, tf = config
        model = load_pytorch_mlflow_model(registered_model_name, model, stage=model_stage)
        config = (model, tf)

    summary(config[0], (3, 224, 224))        
    return config

def load_scripted_model(filename: str) -> nn.Module:
    return torch.jit.load(filename)  

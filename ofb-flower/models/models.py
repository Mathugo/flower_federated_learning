from collections import OrderedDict
from typing import Tuple
from sqlalchemy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor, dropout
from torchvision import datasets
from torchvision.models import resnet18, convnext_tiny, mobilenet_v3_small, vit_b_16, vit_l_16
from .vit_modules import *
import flwr as fl
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import VisionTransformer
import os
from datetime import *

# pylint: disable=unsubscriptable-object
class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""
    def __init__(self, n_classes: int=3) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

class HugoNet(nn.Module):
    """Simple 3 layers deep CNN"""
    def __init__(self, n_classes: int=3, channels: int=3, input_shape: Tuple[int, int]= (224, 224), root_dir: str="models", weights_folder: str="aggr_weights") -> None:
        super(HugoNet, self).__init__()
        self._root_dir = root_dir
        self._model_folder = os.path.join(self._root_dir, self.toString())
        self._aggr_weight_folder = os.path.join(self._model_folder, weights_folder)
        self._create_folders()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(2,2), stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(64, affine=False)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(2,2), stride=2, padding=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(128, affine=False)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(2,2), stride=2, padding=3) 
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(256, affine=False)

        self.flatten = nn.Flatten() # -> 32x215296
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(215296, n_classes)

        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)

    def _create_folders(self) -> None:
        """Create folders to save models and weights""" 

        if not os.path.isdir(self._root_dir):
            print("[MODEL] {self._root_dir} doesn't exist, creating folder structure ..")
            os.mkdir(self._root_dir)
            os.mkdir(self._model_folder)
            os.mkdir(self._aggr_weight_folder)

        elif not os.path.isdir(self._model_folder):
            os.mkdir(self._model_folder)
            print("[MODEL] {self._model_folder} doesn't exist, creating folder structure ..")

        if not os.path.isdir(self._aggr_weight_folder):
            os.mkdir(self._aggr_weight_folder)
            print("[MODEL] {self._aggr_weight_folder} doesn't exist, creating folder structure ..")
             
    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = self.batchnorm3(x)

        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
#        output.view(-1, 3)
        return output

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)
    
    def save(self, filename: str=None):
        if filename == None:
            filename = "model-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        filename =  os.path.join(self._model_folder, filename)
        print("[MODEL] Saving to {}".format(filename))
        torch.save(self.state_dict(), filename)
        print("[DONE]")
    
    def save_script(self, filename: str):
        filename = os.path.join(self._model_folder, filename)
        model_scripted = torch.jit.script(self) # Export to TorchScript
        model_scripted.save(filename) # Save

    def toString(self):
        return "hugonet"
    
    @property
    def aggr_weight_folder(self):
        return self._aggr_weight_folder
        
    @property
    def model_folder(self):
        return self._model_folder

"""
class ViT_b_16(VisionTransformer):
    def __init__(self, n_classes: int=3, input_size: Tuple[int, int]= (224, 224), channels: int=3):
        super(image_size=224, patch_size=).__init__()
        m = vit_b_16(true, progress=true, dropout=0.8, attention_dropout=0.8, image_size=224, num_classes=n_classes)
        self.heads.head = torch.nn.Linear(in_feature=768, out_features=n_classes, bias=True)
"""
def ViT_B_16(n_classes: int=10, input_size: Tuple[int, int]= (224, 224), channels: int=3):
    model = ViT_b_16()
    print(model)
    train_nodes, eval_nodes = get_graph_node_names(model)
    print("Train nodes {}\n Eval nodes {}".format(train_nodes, eval_nodes))
    """ 
    image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    """
    return model

def ResNet18(n_classes: int=10):
    """Returns a ResNet18 model from TorchVision"""
    model = resnet18(num_classes=n_classes, progress=True)
    # replace w/ smaller input layer
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    # no need for pooling if training for CIFAR-10
    #model.maxpool = torch.nn.Identity()
    return model

class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes))

        self._transform = transforms.Compose([transforms.Resize((224, 224)), 
        transforms.ToTensor()])

class MyMobileTransformer(nn.Module):
    """ Mobile Transformer inspired by ViT paper --> An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"""
    def __init__(self) -> None:
        super(Net, self).__init__()
    
    def _transform(self, x) -> Tensor:
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """ Compute forward pass """
        pass
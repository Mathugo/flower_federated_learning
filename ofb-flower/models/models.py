from collections import OrderedDict
from typing import Dict, Tuple, List
from sklearn import multiclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.models import resnet18, convnext_tiny, mobilenet_v3_small, vit_b_16, vit_l_16
from .vit_modules import *
import flwr as fl
import pytorch_lightning as pl
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import VisionTransformer
import os, mlflow
from .resnet_modules import *
from datetime import *

class QuantizedModel(nn.Module):
    """Base class for quantized models"""
    def __init__(self):
        pass

class PlModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(PlModel, self).__init__(*args, **kwargs)
        self._criterion = F.cross_entropy
        
    def test_step(self, batch, batch_idx) -> float:
        x, y = batch
        output = self(x)
        loss = self._criterion(output, y).item()
        _, predicted = torch.max(output.data, -1)  # pylint: disable=no-member
        _, labels = torch.max(y, -1)
        # Use the current of Pytorch logger
        accuracy = torchmetrics.functional.accuracy(predicted, labels, num_classes=self._n_classes, multiclass=True, average="macro")
        recall = torchmetrics.functional.recall(predicted, labels, num_classes=self._n_classes, multiclass=True, average="macro")
        precision = torchmetrics.functional.precision(predicted, labels, num_classes=self._n_classes, multiclass=True, average="macro")
        f1 = torchmetrics.functional.f1_score(predicted, labels, num_classes=self._n_classes, multiclass=True, average="macro")

        #pytorch lp doesn't support log on testing
        self.log("test_loss", loss, on_step=True)
        self.log("accuracy", accuracy, on_step=True)
        self.log("recall", recall, on_step=True)
        self.log("precision", precision, on_step=True)
        self.log("f1", f1, on_step=True)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)

        return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = self._criterion(logits, y)
        _, predicted = torch.max(logits.data, -1)  # pylint: disable=no-member
        _, labels = torch.max(y, -1)
        accuracy = torchmetrics.functional.accuracy(predicted, labels, num_classes=self._n_classes, multiclass=True, average="macro")
        # Use the current of PyTorch logger
        self.log("train_loss_step", loss, on_step=True)
        self.log("train_loss_epoch", loss, on_epoch=True)
        self.log("accuracy", accuracy, on_step=True)
        return loss

    def configure_optimizers(self):
        # Todo decays, lr scheduler, weight regularization
        return torch.optim.SGD(self.parameters(), lr=0.003, momentum=0.8)

class FederatedModel(nn.Module):
    """Base class for federated models
        Attributes:
    """
    def __init__(self, 
        basename: str, 
        onServer: bool, 
        root_dir: str="models", 
        weights_folder: str="aggr_weights", 
        channels: int=3, 
        quantization_fusion_layer: List=None,
        input_shape: Tuple[int, int]=(224, 224),
        n_classes: int = 3
        ):
        super(FederatedModel, self).__init__()
        self._root_dir = root_dir
        self._base_name = basename
        self._n_classes = n_classes
        self._model_folder = os.path.join(self._root_dir, self.Basename)
        self._aggr_weight_folder = os.path.join(self._model_folder, weights_folder)
        self._input_shape=input_shape
        self._channels = channels
        self._on_server = onServer
        self._fusion_layer=quantization_fusion_layer
        self._create_folders()

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
        """Save the weights to the .pth pytorch format"""
        if filename == None:
            filename = "model-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        filename =  os.path.join(self._model_folder, filename)
        print("[MODEL] Saving to {}".format(filename))
        torch.save(self.state_dict(), filename)
        print("[DONE]")
    
    def save_script(self, filename: str, model: torch.nn=None):
        """Save model in TorchScript format"""
        filename = os.path.join(self._model_folder, filename)
        model_to_script = self
        if isinstance(model):
            model_to_script = model
        model_scripted = torch.jit.script(model_to_script) # Export to TorchScript
        model_scripted.save(filename) # Save

    def save_pqt_quantized(self, representative_dataset) -> bool:
        """ quantized the model using PQT : Post training quantization"""
        model_fp32 = self
        if self._on_server:
            # configure for server inference
            qconfig = 'fbgemm'
        else:
            # config for mobile inference
            qconfig = 'qnnpack'
        model_fp32.qconfig = torch.quantization.get_default_qconfig(qconfig)

        # Fuse the activations to preceding layers, where applicable.
        if self._fusion_layer == None:
            print("[MODEL] No fusion_layer provied for static quantization, aborting ..")
            return False
        else:
            model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [self._fusion_layer])
            model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

            # calibrate prepared model to dertermine quantization parameters for activations
            #input_fp32 = torch.randn(4, 1, 4, 4)
            model_fp32_prepared(representative_dataset)

            # Convert the observed model to a quantized model. This does several things:
            # quantizes the weights, computes and stores the scale and bias value to be
            # used with each activation tensor, and replaces key operators with quantized
            # implementations.
            model_int8 = torch.quantization.convert(model_fp32_prepared)
            filename = "model-qt8-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            self.save_script(filename, model_int8)

    @property
    def Basename(self):
        """Basename for saving and model registry purposes"""
        return self._base_name

    @Basename.setter
    def Basename(self, value: str):
        self._base_name = value

    @property
    def aggr_weight_folder(self):
        """Aggregated weight folder"""
        return self._aggr_weight_folder
        
    @property
    def model_folder(self):
        return self._model_folder

class HugoNet(FederatedModel, PlModel):
    """Simple 3 layers deep CNN"""
    def __init__(self, onServer: bool, basename: str="hugonet", config: Dict=None, n_classes: int=3, **kwargs) -> None:
        self._n_classes = n_classes
        FederatedModel.__init__(self,  basename, onServer, quantization_fusion_layer=['conv', 'batchnorm', 'relu'], n_classes=n_classes, **kwargs)
        PlModel.__init__(self)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(2,2), stride=2, padding=3)
        self.batchnorm1 = nn.BatchNorm2d(64, affine=False)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=0)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(2,2), stride=2, padding=3)
        self.batchnorm2 = nn.BatchNorm2d(128, affine=False)
        self.relu2 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=0)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(2,2), stride=2, padding=3) 
        self.batchnorm3 = nn.BatchNorm2d(256, affine=False)
        self.relu3 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=0)

        self.flatten = nn.Flatten() # -> 32x215296
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(215296, self._n_classes)

        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        # TODO Post Training Quantization -> Static Quantization for CNN
        #self._config_quantized(onServer)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output

class ResNet(FederatedModel):
    """Residual neural network"""
    def __init__(self, n_classes, basename: str="resnet", *args, **kwargs):
        super(ResNet, self).__init__(basename, **kwargs)
        self.encoder = ResNetEncoder((self._channels, self._input_shape[0], self._input_shape[1]), *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Deit(FederatedModel):
    """Data-efficient image Transformers: A promising new technique for image classification"""
    def __init__(self, basename: str="deit", **kwargs):
        # To be implemented
        super(HugoNet, self).__init__(basename, **kwargs)

        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        model.eval()

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

def resnet18(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)

def resnet34(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

def resnet50(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

def resnet101(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 23, 3], *args, **kwargs)

def resnet152(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 8, 36, 3], *args, **kwargs)

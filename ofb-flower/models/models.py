import sys
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18, convnext_tiny, mobilenet_v3_small, vit_b_16, vit_l_16
from .base_models import FederatedModel, PlModel, QuantizedModel
from datetime import *
from typing import Tuple
import timm
import torch

# Criterions : Computes the differences between two probability distributions
# F.cross_entropy : Cross entropy penalizes greatly for being very confident and wrong. -> Creating confident models—the prediction will be accurate and with a higher probability.
# kullback-leibler divergence (KL Divergence) : Its output tells you the proximity of two probability distributions. Multi-class classification tasks
# Cross-Entropy punishes the model according to the confidence of predictions, and KL Divergence doesn’t. KL Divergence only assesses how the probability distribution prediction is different from the distribution of ground truth.
# kl_loss = nn.KLDivLoss(reduction = 'batchmean')
# Focal Loss

class HugoNet(FederatedModel, PlModel):
    """Simple 3 layers deep CNN"""
    def __init__(self, onServer: bool, basename: str="hugonet", config: Dict=None, n_classes: int=3, **kwargs: Any) -> None:
        self._n_classes = n_classes
        FederatedModel.__init__(self, basename, onServer, quantization_fusion_layer=['conv', 'batchnorm', 'relu'], n_classes=n_classes, **kwargs)
        PlModel.__init__(self, learning_rate=1e-4, lr_scheduler=True)

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

    def test_step(self, batch, batch_idx) -> float:
        return super().test_step(batch, batch_idx, self._n_classes)

    def training_step(self, batch, batch_nb, optimizer_idx):
        return super().training_step(batch, batch_nb, optimizer_idx, self._n_classes)

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

class FlResnet18(FederatedModel, PlModel):
    def __init__(self, onServer: bool, input_shape: Tuple[int, int, int], n_classes: int, basename: str="resnet18", learning_rate=3e-4, transfer=False, *arg, **kwargs):
        FederatedModel.__init__(self,  basename, onServer, quantization_fusion_layer=['conv', 'batchnorm', 'relu'], n_classes=n_classes, **kwargs)
        PlModel.__init__(self, learning_rate=5e-4, lr_scheduler=True)
        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = n_classes
        # transfer learning if pretrained=True and onServer (client don't need to dl the weights, it will be send via gRpc)
        self.feature_extractor = resnet18(pretrained=(transfer and onServer))
        
        if transfer:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        n_sizes = self._get_conv_output(input_shape)
        self.classifier = nn.Linear(n_sizes, self._n_classes)
  
    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(tmp_input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x
    # will be used during inference
    def forward(self, x):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       x = self.classifier(x)
       return x

    def training_step(self, batch, batch_nb, optimizer_idx):
        x, y = batch
        logits = self(x)
        loss = self._criterion(logits, y)
        _, predicted = torch.max(logits.data, -1) 
        _, labels = torch.max(y, -1)
        return super().test_step(batch, batch_nb, optimizer_idx, predicted, labels, loss, self._n_classes)

    def test_step(self, batch, batch_idx):
        # overload of test_step from pl lightning
        x, y = batch
        output = self(x)
        loss = self._criterion(output, y).item()
        _, predicted = torch.max(output.data, -1)  # pylint: disable=no-member
        _, labels = torch.max(y, -1)
        return super().test_step(batch, batch_idx, predicted, labels, loss, self._n_classes)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 

class HubModel(FederatedModel, PlModel):
    """Hub Model"""
    def __init__(self, onServer: bool, basename: str, n_classes: int, pretrained: bool=True, learning_rate: float=3e-4, alpha: float=None, *args: Any, **kwargs: Any) -> None:
        self._n_classes = n_classes
        self.alpha = alpha
        self.onserver = onServer
        self.pretrained = pretrained

        FederatedModel.__init__(self, basename, onServer, n_classes=n_classes, **kwargs)
        PlModel.__init__(self, learning_rate=learning_rate, has_pretrained_weights=(pretrained and onServer))
        #If the model was not found in mlflow registry -> we load pretrained weights from hub
        self.feature_extractor = timm.create_model(basename, pretrained= (onServer and pretrained), num_classes=n_classes)
        if pretrained:
            self.feature_extractor.reset_classifier(n_classes)

        if self.alpha != None:
            print(f"[MODEL] Freezing neural networks with coeff {self.alpha} ..")
            self.do_freeze()

    def do_freeze(self) -> None:
        """Freeze parameters in a network using a freezing coeff 0 < alpha < 1"""
        nb_parameters =sum(1 for x in self.feature_extractor.parameters())
        nb_to_freeze = int(nb_parameters * self.alpha)
        print(f"[MODEL] {nb_parameters} of parameters, {nb_to_freeze} to freeze with alpha {self.alpha}", file=sys.stderr)
        # freeze params
        for i, param in enumerate(self.feature_extractor.parameters()):
            param.requires_grad = False
            if i >= nb_to_freeze:
                print("[MODEL] Done", file=sys.stderr)
                break
            
    def load_pretrained_weights(self) -> None:
        print(f"[MODEL] Loading pretrained weights for {self.Basename} ..", file=sys.stderr)
        self.feature_extractor = timm.create_model(self.Basename, pretrained= (self.onserver and self.pretrained), num_classes=self._n_classes)
        self.feature_extractor.reset_classifier(self._n_classes)
        if self.alpha != None:
            self.do_freeze()
            
        print("[MODEL] Done")

    def forward(self, x):
        return self.feature_extractor(x)

    def training_step(self, batch, batch_nb, optimizer_idx):
        return super().training_step(batch, batch_nb, optimizer_idx, self._n_classes)

    def test_step(self, batch, batch_idx):
        return super().test_step(batch, batch_idx, self._n_classes)


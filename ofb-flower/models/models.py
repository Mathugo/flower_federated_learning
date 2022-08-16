import sys, timm, mlflow, torch
from typing import Any, Dict
import torch.nn as nn
import torch.nn.functional as F
from models.config import TrainingConfig
from torch import Tensor
from .base_models import FederatedModel, PlModel
from datetime import *

# Criterions : Computes the differences between two probability distributions
# F.cross_entropy : Cross entropy penalizes greatly for being very confident and wrong. -> Creating confident models—the prediction will be accurate and with a higher probability.
# kullback-leibler divergence (KL Divergence) : Its output tells you the proximity of two probability distributions. Multi-class classification tasks
# Cross-Entropy punishes the model according to the confidence of predictions, and KL Divergence doesn’t. KL Divergence only assesses how the probability distribution prediction is different from the distribution of ground truth.
# kl_loss = nn.KLDivLoss(reduction = 'batchmean')
# Focal Loss

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet(FederatedModel, PlModel):
    def __init__(self, onServer: bool, trainconfig: TrainingConfig, in_channels: int=3, resblock: nn.Module=ResBlock, *args: Any, **kwargs: Any) -> None:
        self._n_classes=trainconfig.n_classes
        self.in_channels=in_channels
        FederatedModel.__init__(self, trainconfig, onServer, *args, **kwargs)
        PlModel.__init__(self, trainconfig, has_pretrained_weights=False, *args, **kwargs)

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )
        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )
        self.layer3 = nn.Sequential(
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False)
        )
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(128, self._n_classes)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.gap(input)
        input = torch.squeeze(input)
        input = input.view(input.size(0), 1, 128)
        input = self.fc(input)
        input = torch.squeeze(input)
        return F.log_softmax(input, dim=1)

    def test_step(self, batch, batch_idx) -> float:
        return super().test_step(batch, batch_idx, self._n_classes)

    def training_step(self, batch, batch_nb, optimizer_idx):
        return super().training_step(batch, batch_nb, optimizer_idx, self._n_classes)

class ResHugoNet(FederatedModel, PlModel):
    """Simple 4 layers deep CNN with residual connections"""
    def __init__(self, onServer: bool, trainconfig: TrainingConfig, in_channels: int=3, out_channels_res_block: int=128,  *args, **kwargs: Any) -> None:
        self._n_classes=trainconfig.n_classes
        self.in_channels=in_channels
        self.out_channels=out_channels_res_block

        FederatedModel.__init__(self, trainconfig, onServer, *args, **kwargs)
        PlModel.__init__(self, trainconfig, has_pretrained_weights=False, *args, **kwargs)
        
        self.res_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels_res_block, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_res_block, affine=False),
            nn.ReLU(inplace=True),
        )
        self.res_block_main = nn.Sequential(
            nn.Conv2d(out_channels_res_block, out_channels_res_block, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_res_block, affine=False),
            nn.ReLU(inplace=True),
        )
        self.avg = nn.AvgPool2d(kernel_size=7, stride=1, padding=1)

        self.end = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, self._n_classes))
        
    def test_step(self, batch, batch_idx) -> float:
        return super().test_step(batch, batch_idx, self._n_classes)

    def training_step(self, batch, batch_nb, optimizer_idx):
        return super().training_step(batch, batch_nb, optimizer_idx, self._n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        # inspired by https://arxiv.org/pdf/1608.06993.pdf Densely Connected Convolutional Networks

        x = self.res_block_1(x)
        residual_1 = x
        x = self.res_block_main(x)
        x += residual_1

        residual_2 = x        
        x = self.res_block_main(x)
        x+= residual_2 + residual_1

        residual_3 = x
        x = self.res_block_main(x)
        x += residual_1 + residual_2 + residual_3

        x = self.res_block_main(x)
        x = self.avg(x)
        x = x.view(-1, 1024)
        x = self.end(x)
     
        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output

class HugoNet(FederatedModel, PlModel):
    """Simple 3 layers deep CNN"""
    def __init__(self, onServer: bool, trainconfig: TrainingConfig, *args: Any, **kwargs: Any) -> None:
        self._n_classes=trainconfig.n_classes
        FederatedModel.__init__(self, trainconfig, onServer, quantization_fusion_layer=['conv', 'batchnorm', 'relu'], *args, **kwargs)
        PlModel.__init__(self, trainconfig, has_pretrained_weights=False, *args, **kwargs)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(2,2), stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64, affine=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=0)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(2,2), stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128, affine=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=0)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(2,2), stride=2, padding=1) 
        self.batchnorm3 = nn.BatchNorm2d(256, affine=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=0)

        self.flatten = nn.Flatten() # -> 32x215296
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*676, self._n_classes)

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

class HubModel(FederatedModel, PlModel):
    """Hub Model"""
    def __init__(self, onServer: bool, trainconfig: TrainingConfig, pretrained: bool=True, *args: Any, **kwargs: Any) -> None:
        self._n_classes = trainconfig.n_classes
        self.alpha = trainconfig.freezing_coeff
        print(f"[MODEL] Alpha freezing coeff {self.alpha}", file=sys.stderr)
        self.onserver = onServer
        self.pretrained = pretrained
        FederatedModel.__init__(self, trainconfig, onServer, *args, **kwargs)
        PlModel.__init__(self, trainconfig, has_pretrained_weights=(pretrained and onServer), *args, **kwargs)
        #If the model was not found in mlflow registry -> we load pretrained weights from hub
        self.feature_extractor = timm.create_model(self.Basename, pretrained= (onServer and pretrained), num_classes=self._n_classes)
        
        if pretrained:
            self.feature_extractor.reset_classifier(self._n_classes)
        # TODO if alpha == 1 then we freeze the entire network and reset the classifier
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


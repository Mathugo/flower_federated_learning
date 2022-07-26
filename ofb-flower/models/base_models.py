from collections import OrderedDict
import sys
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
import os, mlflow
import flwr as fl
from datetime import datetime

class QuantizedModel(nn.Module):
    """Base class for quantized models"""
    def __init__(self):
        pass

class PlModel(pl.LightningModule):
    def __init__(self, learning_rate: float=1e-4, lr_scheduler: bool=True, criterion=F.cross_entropy, has_pretrained_weights: bool=False, *args, **kwargs):
        super(PlModel, self).__init__(*args, **kwargs)
        self._criterion = criterion
        self._learning_rate = learning_rate
        self._lr_scheduler = lr_scheduler
        self._has_pretrained_weights = has_pretrained_weights
    
    def test_step(self, batch, batch_idx, n_classes, *args, **kwargs):
        # Use the current of Pytorch logger
        x, y = batch
        output = self(x)
        loss = self._criterion(output, y).item()
        _, predicted = torch.max(output.data, -1)  # pylint: disable=no-member
        _, labels = torch.max(y, -1)
        accuracy = torchmetrics.functional.accuracy(predicted, labels, num_classes=n_classes, multiclass=True, average="macro")
        recall = torchmetrics.functional.recall(predicted, labels, num_classes=n_classes, multiclass=True, average="macro")
        precision = torchmetrics.functional.precision(predicted, labels, num_classes=n_classes, multiclass=True, average="macro")
        f1 = torchmetrics.functional.f1_score(predicted, labels, num_classes=n_classes, multiclass=True, average="macro")

        #pytorch lp doesn't support log on testing
        self.log("test_loss", loss)
        self.log("accuracy", accuracy)
        self.log("recall", recall)
        self.log("precision", precision)
        self.log("f1", f1)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)
        return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}

    def training_step(self, batch, batch_nb, optimizer_idx, n_classes, *args, **kwargs):
        x, y = batch
        logits = self(x)
        loss = self._criterion(logits, y)
        _, predicted = torch.max(logits.data, -1) 
        _, labels = torch.max(y, -1)

        accuracy = torchmetrics.functional.accuracy(predicted, labels, num_classes=n_classes, multiclass=True, average="macro")
        # Use the current of PyTorch logger
        self.log("loss_step", loss, on_step=True)
        self.log("accuracy", accuracy)
        return {"loss": loss, "accuracy": accuracy}
    
    def configure_optimizers(self):
        # we can put multiple optimizers 
        opt_adam = torch.optim.Adam(self.parameters(), lr=self._learning_rate) 
        opt_sgd = torch.optim.SGD(self.parameters(), lr=self._learning_rate, momentum=0.8)
        if self._lr_scheduler:
            #lr_scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_adam, T_0=10, T_mult=2)
            lr_scheduler_adam = torch.optim.lr_scheduler.StepLR(opt_adam, step_size=1)
            lr_scheduler_sgd = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_sgd, T_0=10, T_mult=2)
            return [opt_adam, opt_sgd], [lr_scheduler_adam, lr_scheduler_sgd]
        return [ {"optimizer": opt_sgd, "frequency": 5}, {"optimizer": opt_adam, "frequency": 10}], [lr_scheduler_sgd, lr_scheduler_adam]
    @property
    def HasPretrainedWeights(self) -> bool:
        return self._has_pretrained_weights

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
        
        #self._create_folders()

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
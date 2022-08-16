from collections import OrderedDict
from models.config import TrainingConfig
from typing import Any, Dict, Tuple, List
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import os, mlflow, torch, numpy, torch
import flwr as fl
from datetime import datetime
import torch.nn.functional as F
import torch.nn as nn

# It's a wrapper around a model that quantizes the weights and activations of the model
class QuantizedModel(nn.Module):
    """Base class for quantized models : TO BE DONE"""
    def __init__(self):
        pass

# > The `PlModel` class is a subclass of the `pl.LightningModule` class
class PlModel(pl.LightningModule):
    def __init__(self, trainconfig: TrainingConfig, has_pretrained_weights: bool=False, *args, **kwargs):
        """
        The function takes in a training configuration object, and a boolean value indicating whether
        the model has pretrained weights. It then sets the criterion to either cross entropy or KL
        divergence loss, depending on the training configuration. It also sets the learning rate and
        learning rate scheduler to the values in the training configuration
        
        Args:
          trainconfig (TrainingConfig): TrainingConfig
          has_pretrained_weights (bool): Whether the model has pretrained weights. Defaults to False
        """
        super(PlModel, self).__init__(*args, **kwargs)

        # get criterion from str
        criterion_str = trainconfig.criterion
        if criterion_str == "cross_entropy":
            self._criterion = F.cross_entropy
        elif criterion_str == "KLDivLoss":
            self._criterion = nn.KLDivLoss(reduction = 'batchmean')
        self._learning_rate = trainconfig.lr
        self._lr_scheduler = trainconfig.lr
        self._has_pretrained_weights = has_pretrained_weights
        self._last_loss = None
    
    def test_step(self, batch, batch_idx, n_classes, *args, **kwargs):
        """
        It takes in a batch of data, runs it through the model, and then calculates the loss, accuracy,
        recall, precision, and f1 score. 
        The function then logs the loss, accuracy, recall, precision, and f1 score to the Lightning
        logger and to MLflow. 
        The function returns a dictionary of the accuracy, recall, precision, and f1 score. 
        The function is called in the `test_epoch_end` function, which is called at the end of each
        epoch. 
        The `test_epoch_end` function is called in the `test` function, which is called at the end of
        the training. 
        The `test` function is called in the `fit` function, which is called when you run
        `trainer.fit(model)`. 

        The `fit` function is called when you run `trainer.fit(model)`.
        
        Args:
          batch: The batch of data that was passed to the test_step function.
          batch_idx: The index of the current batch.
          n_classes: number of classes in the dataset
        
        Returns:
          The return value is a dictionary of the metrics that are being logged.
        """
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
        """
        > We take a batch of data, pass it through the model, calculate the loss, and then calculate the
        accuracy
        
        Args:
          batch: The input and target for the model.
          batch_nb: the current batch number
          optimizer_idx: The index of the optimizer to use.
          n_classes: number of classes in the dataset
        
        Returns:
          The loss and accuracy for the current batch.
        """

        x, y = batch
        logits = torch.squeeze(self(x))
        loss = self._criterion(logits, y)
        _, predicted = torch.max(logits.data, -1) 
        _, labels = torch.max(y, -1)

        accuracy = torchmetrics.functional.accuracy(predicted, labels, num_classes=n_classes, multiclass=True, average="macro")
        # Use the current of PyTorch logger
        self.log("loss_step", loss, on_step=True)
        self.log("accuracy", accuracy)
        return {"loss": loss, "accuracy": accuracy}

    def training_epoch_end(self, outputs):
        loss = dict(outputs[-1][1])["loss"]
        self.LastLoss = loss

    def configure_optimizers(self):
        """
        We are returning a list of dictionaries, where each dictionary contains an optimizer and a
        frequency. 
        The frequency is the number of batches that will be processed before the optimizer is called. 
        In this case, we are using the Adam optimizer for every 10 batches and the SGD optimizer for
        every 5 batches. 
        We are also using a learning rate scheduler for both optimizers. 
        The learning rate scheduler is called after each batch. 
        The learning rate scheduler is not required, but it is a good idea to use one. 
        The learning rate scheduler will decay the learning rate over time. 
        This is important because the learning rate should be decreased as the model converges. 
        If the learning rate is too high, the model will not converge. 
        If the learning rate is too low, the model will converge very slowly. 
        The
        """
        opt_adam = torch.optim.Adam(self.parameters(), lr=self._learning_rate) 
        opt_sgd = torch.optim.SGD(self.parameters(), lr=self._learning_rate, momentum=0.8)
        if self._lr_scheduler:
            lr_scheduler_adam = torch.optim.lr_scheduler.StepLR(opt_adam, step_size=1)
            lr_scheduler_sgd = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_sgd, T_0=10, T_mult=2)
            return [opt_adam, opt_sgd], [lr_scheduler_adam, lr_scheduler_sgd]
        return [ {"optimizer": opt_sgd, "frequency": 5}, {"optimizer": opt_adam, "frequency": 10}], [lr_scheduler_sgd, lr_scheduler_adam]
    
    @property
    def HasPretrainedWeights(self) -> bool:
        """
        > This function returns a boolean value indicating whether the model has pretrained weights or
        not
        
        Returns:
          A boolean value.
        """
        return self._has_pretrained_weights
    
    @property
    def LastLoss(self) -> float:
        return self._last_loss.item()
    
    @LastLoss.setter
    def LastLoss(self, value):
        self._last_loss = value

class FederatedModel(nn.Module):
    """Base class for federated models
        Attributes:
    """
    def __init__(self, 
        trainconfig: TrainingConfig,
        onServer: bool,
        quantization_fusion_layer: List=None,
        *args : Any,
        **kwargs: Any
        ):
        super(FederatedModel, self).__init__(trainconfig, onServer, *args, **kwargs)
        self._base_name = trainconfig.model
        self._on_server = onServer
        self._fusion_layer=quantization_fusion_layer

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def get_weights(self):
        """
        It returns a list of NumPy ndarrays, where each ndarray is a weight matrix
        
        Returns:
          A list of NumPy ndarrays.
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: numpy.ndarray) -> None:
        """
        `set_weights` takes a list of NumPy arrays and converts them to a dictionary of PyTorch tensors
        
        Args:
          weights (numpy.ndarray): a list of NumPy ndarrays
        """
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
        """
        > We take the model we want to save, convert it to TorchScript, and save it to a file
        
        Args:
          filename (str): The name of the file to save the model to.
          model (torch.nn): The model to be saved. If not specified, the model that was used to create
        the instance of the class is used.
        """
        """Save model in TorchScript format"""
        
        filename = os.path.join(self._model_folder, filename)
        model_to_script = self
        if isinstance(model):
            model_to_script = model
        model_scripted = torch.jit.script(model_to_script) # Export to TorchScript
        model_scripted.save(filename) # Save

    def save_pqt_quantized(self, representative_dataset) -> bool:
        """#TODO NOT FULLY IMPLEMENTED YET quantized the model using PQT : Post training quantization"""
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
        """
        The function Basename() returns the base name of the model
        
        Returns:
          The base name of the model.
        """
        return self._base_name

    @Basename.setter
    def Basename(self, value: str):
        self._base_name = value
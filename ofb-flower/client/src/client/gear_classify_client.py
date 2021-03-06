"""Flower client example using PyTorch for Gear image classification."""
import torch, timeit
import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
from ..pipeline import ClassifyDataset
from utils.utils import set_weights, get_weights, print_auto_logged_info
from utils.mlflow_client import MLFlowClient
from models.models import FederatedModel
import pytorch_lightning as pl
import mlflow
from pytorch_lightning import loggers as pl_loggers

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

class GearClassifyClient(fl.client.Client):
    """Flower client implementing Gear image classification using PyTorch."""
    def __init__(self, cid: str, name: str, model: FederatedModel, trainset: ClassifyDataset, testset: ClassifyDataset, mlflow_client: MLFlowClient) -> None:
        self.cid = cid
        self._model = model
        self._name = name
        self._model_registry_name = f"{self.cid}-{self._name}-{self._model.Basename}"
        self._trainset = trainset
        self._testset = testset
        self._mlflow_client = mlflow_client

    def get_parameters(self) -> None:
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, ins: FitIns) -> FitRes:
        """Train the client"""

        print(f"Client {self.cid}: fit, config: {ins.config}")
        #Get config for training and experiment
        self._get_config(ins)
        print("[CLIENT] Fitting ..")
        # Set model parameters
        set_weights(self._model, self._weights)
        if torch.cuda.is_available():
            kwargs = {
                "num_workers": self._num_workers,
                "pin_memory": self._pin_memory,
                "drop_last": True,
            }
        else:
            kwargs = {"drop_last": True}
        # Train model
        trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=self._batch_size, shuffle=True
        )
        print("Len train dataset {} len trailoader {}".format(len(self._trainset), len(trainloader)))
        # # Initialize a trainer with accelerator="gpu"
        trainer = pl.Trainer(max_epochs=self._epochs, progress_bar_refresh_rate=1, log_every_n_steps=5)
        # Auto log all MLflow entities
        mlflow.pytorch.autolog(log_every_n_step=1, registered_model_name=self._model_registry_name, log_models=False)
        with mlflow.start_run(run_name="train", nested=True) as run:
            trainer.fit(self._model, trainloader)
            mlflow.pytorch.log_model(
            self._model, 
            self._model.Basename,
            registered_model_name=self._model_registry_name
            )
            
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id), self._model.Basename)
        print("[CLIENT] Done")
        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = get_weights(self._model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self._trainset)
        metrics = {"duration": timeit.default_timer() - self._fit_begin}
        print("[CLIENT] Number of trainning examples {}".format(num_examples_train))
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics
        )
    
    def _get_config(self, ins: FitIns):
        """Get configuration from server answer"""
        # get weights from server
        self._weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        self._config = ins.config
        self._fit_begin = timeit.default_timer()
        # Get training config
        self._epochs = int(self._config["epochs"])
        self._batch_size = int(self._config["batch_size"])
        self._pin_memory = bool(self._config["pin_memory"])
        self._num_workers = int(self._config["num_workers"])

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate training on client side """

        print(f"Client {self.cid}: evaluate")
        weights = fl.common.parameters_to_weights(ins.parameters)
        # Use provided weights to update the local model
        set_weights(self._model, weights)
        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self._testset, batch_size=32
        )
        # Auto log all MLflow entities
        mlflow.pytorch.autolog(log_every_n_step=1, log_models=False)
        
        pl_loggers.MLFlowLogger()

        with mlflow.start_run(run_name="test", nested=True) as run:
            trainer = pl.Trainer(progress_bar_refresh_rate=1, log_every_n_steps=1)
            results = trainer.test(self._model, testloader)[0]
            # returned metrics
            accuracy= results["accuracy_epoch"]
            precision = results["precision_epoch"]
            recall = results["recall_epoch"]
            f1 = results["f1_epoch"]
            test_loss = results["test_loss_epoch"]

            #mlflow.log_metric("accuray", f"{accuracy}")
            #mlflow.log_metric("test_loss", f"{train_loss}")
            print(f"[CLIENT] Test Results {results}")

        # Return the number of evaluation examples and the evaluation result (loss)
        metrics = {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}
        return EvaluateRes(
            loss=test_loss, num_examples=len(self._testset), metrics=metrics
        )
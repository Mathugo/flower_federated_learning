"""Flower client example using PyTorch for Gear image classification."""
from argparse import ArgumentParser
import torch, timeit, sys, mlflow
import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
from utils.utils import print_auto_logged_info
from models.config import TrainingConfig
from utils.mlflow_client import MLFlowClient
from utils.load import load_classify_dataset, load_model
from models.models import FederatedModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-membe

class GearClassifyClient(fl.client.Client):
    """Flower client implementing Gear image classification using PyTorch."""
    def __init__(self, args: ArgumentParser, mlflow_client: MLFlowClient) -> None:
        """
        The function takes in two arguments, an ArgumentParser object and an MLFlowClient object. It
        then sets the class variables args, cid, _name, _mlflow_client, trainconfig, _model, and
        _has_already_load_dataset_model
        
        Args:
          args (ArgumentParser): The arguments passed to the program.
          mlflow_client (MLFlowClient): This is the MLFlowClient object that we created in the previous
        section.
        """
        self.args = args
        self.cid = args.cid
        self._name = args.name
        self._mlflow_client = mlflow_client
        self.trainconfig = None
        self._model = None
        self._has_already_load_dataset_model = False

    def get_parameters(self) -> None:
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def _get_config(self, ins: FitIns):
        """
        > The function takes in a FitIns object, which contains the parameters and configuration for the
        model, and then sets the weights, configuration, and training configuration for the model
        
        Args:
          ins (FitIns): FitIns
        """
        # get weights from server
        self._weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        self._config = ins.config
        self.trainconfig = TrainingConfig()
        self._fit_begin = timeit.default_timer()
        # Get training config
        self.trainconfig.epochs = int(self._config["epochs"])
        self.trainconfig.batch_size = int(self._config["batch_size"])
        self.trainconfig.lr = float(self._config["lr"])
        self.trainconfig.lr_scheduler = bool(self._config["lr_scheduler"])
        self.trainconfig.criterion = self._config["criterion"]
        freezing_coeff_str = self._config["freezing_coeff"]
        if freezing_coeff_str != 'None':
            self.trainconfig.freezing_coeff = float(freezing_coeff_str)
        else:
            self.trainconfig.freezing_coeff = None

        self.trainconfig.model = str(self._config["model"])
        self._model = self.trainconfig.model
        self._model_registry_name = f"{self.cid}-{self._name}-{self._model}"
        self.trainconfig.n_classes = int(self._config["n_classes"])
        self.trainconfig.data_augmentation = bool(self._config["data_augmentation"])
        self.trainconfig.pin_memory = bool(self._config["pin_memory"])
        self.trainconfig.num_workers = int(self._config["num_workers"])
    
    def _load_model_and_dataset(self):
        """
        > Loads the model and dataset from the trainconfig file
        """
        self._model, trf = load_model(self.trainconfig, False)
        self._model.to(DEVICE)
        self._trainset, self._testset = load_classify_dataset(self.args.data_dir, transforms=trf, data_augmentation=self.trainconfig.data_augmentation)
    
    def fit(self, ins: FitIns) -> FitRes:
        """
        > The client loads the model and dataset, sets the weights from the server, trains the model,
        and returns the refined weights and the number of examples used for training
        
        Args:
          ins (FitIns): FitIns
        
        Returns:
          The refined weights and the number of examples used for training
        """
        print(f"Client {self.cid}: fit, config: {ins.config}", file=sys.stderr)
        self._get_config(ins)
        if not self._has_already_load_dataset_model: 
            self._load_model_and_dataset()
            self._has_already_load_dataset_model = True
        
        print("[CLIENT] Setting weights from server ..", file=sys.stderr)
        self._model.set_weights(self._weights)
        if torch.cuda.is_available():
            kwargs = {
                "num_workers": self.trainconfig.num_workers,
                "pin_memory": self.trainconfig.pin_memory,
                "drop_last": True,
            }
        else:
            kwargs = {"drop_last": True}

        trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=self.trainconfig.batch_size, shuffle=True
        )
        print("[DATASET] Len train dataset {} len trailoader".format(len(self._trainset)), file=sys.stderr)
        # # Initialize a trainer with accelerator="gpu"
        trainer = pl.Trainer(max_epochs=self.trainconfig.epochs, callbacks=[TQDMProgressBar(refresh_rate=1)], log_every_n_steps=5)
        # Auto log all MLflow entities
        mlflow.pytorch.autolog(log_every_n_step=1, registered_model_name=self._model_registry_name, log_models=True)
       
        with mlflow.start_run(run_name="train", nested=True) as run:
            print("[CLIENT] Fitting ..", file=sys.stderr)
            trainer.fit(self._model, trainloader)
            
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id), self._model.Basename)
        print("[CLIENT] Done", file=sys.stderr)
        weights_prime: Weights = self._model.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self._trainset)
        metrics = {"duration": timeit.default_timer() - self._fit_begin}
        print("[CLIENT] Number of trainning examples {}".format(num_examples_train), file=sys.stderr)
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics
        ) 

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        We use the provided weights to update the local model, then evaluate the updated model on the
        local dataset
        
        Args:
          ins (EvaluateIns): EvaluateIns
        
        Returns:
          The loss, number of examples, and metrics
        """
        print(f"Client {self.cid}: evaluate", file=sys.stderr)
        weights = fl.common.parameters_to_weights(ins.parameters)
        self._model.set_weights(weights)
        testloader = torch.utils.data.DataLoader(
            self._testset, batch_size=self.trainconfig.batch_size
        )
        # Auto log all MLflow entities
        mlflow.pytorch.autolog(log_every_n_step=1, log_models=False)

        with mlflow.start_run(run_name="test", nested=True) as run:
            trainer = pl.Trainer(callbacks=[TQDMProgressBar(refresh_rate=1)], log_every_n_steps=1)
            results = trainer.test(self._model, testloader)[0]
            # returned metrics
            accuracy= results["accuracy"]
            precision = results["precision"]
            recall = results["recall"]
            f1 = results["f1"]
            test_loss = results["test_loss"]
            print(f"[CLIENT] Test Results {results}")

        # Return the number of evaluation examples and the evaluation result (loss)
        metrics = {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}
        return EvaluateRes(
            loss=test_loss, num_examples=len(self._testset), metrics=metrics
        )

    @property
    def model(self) -> FederatedModel:
        """
        It returns the model.
        :return: The model is being returned.
        """
        return self._model
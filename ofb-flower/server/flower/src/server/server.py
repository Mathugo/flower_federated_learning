import torch, os, glob, mlflow, sys
sys.path.append("../..")
from models.base_models import FederatedModel
from utils.utils import TrainingConfig
from utils.load import load_model
from utils.mlflow_client import MLFlowClient
from ..strategies import CustomModelStrategyFedAvg
from typing import Dict
from src.app.args import Args
import flwr as fl

class ClassificationServer:
    def __init__(self, args: Args):
        """Federated Server: server-side parameter initialization"""
        self._args=args
        self._registered_model_name = f"aggregated-{self._args.model}"
        self._mlflow_client = MLFlowClient("server", args.mlflow_server_ip, args.mlflow_server_port)
        self._last_run_id = None
        self._create_trainingconfig()
        self._test_given_parameters()
        self._load_config()

    def _create_trainingconfig(self) -> None:
        """
        The function creates a training configuration object that contains the training parameters
        """
        self._trainconfig = TrainingConfig()
        self._trainconfig.epoch_global = int(self._args.rounds)
        self._trainconfig.epochs = int(self._args.local_epochs)
        self._trainconfig.lr = float(self._args.lr)
        if self._args.freezing_coeff != None:
            self._trainconfig.freezing_coeff = float(self._args.freezing_coeff)
        else:
            self._trainconfig.freezing_coeff = self._args.freezing_coeff
        self._trainconfig.lr_scheduler=bool(self._args.lr_scheduler)
        self._trainconfig.criterion=str(self._args.criterion)
        self._trainconfig.model = str(self._args.model)
        self._trainconfig.model_stage = str(self._args.model_stage)
        self._trainconfig.n_classes = int(self._args.n_classes)
        self._trainconfig.data_augmentation = bool(self._args.data_augmentation)
        self._trainconfig.pin_memory = bool(self._args.pin_memory)
        self._trainconfig.num_workers = int(self._args.num_workers)

    def _test_given_parameters(self):
        """
        > Test if parameters are ok
        """
        print("[SERVER] Config {}".format(self._args), file=sys.stderr)
        assert (
        self._args.min_sample_size <= self._args.min_num_clients
        ), f"Num_clients shouldn't be lower than min_sample_size"
        # TODO test other parameters

    def _load_model_from_folder(self) -> None:
        """
        > If the user has specified a folder to load weights from, then load the latest weights from
        that folder
        """
        if self._args.load_model_from_folder:
            # TODO load_weight from .pth
            files = glob.glob(self._model.aggr_weight_folder+"/*.pth")
            if len(files) != 0:
                last_file = max(files, key=os.path.getctime)
                print(f"[SERVER] Loading weights {last_file} ..", file=sys.stderr)
                w = torch.load(last_file)
                self._model.load_state_dict(w)
                print("[SERVER] Done", file=sys.stderr)
            else:
                print(f"[SERVER] Args load_weights = {self._args.load_weights} but no files found at {self._model.aggr_weight_folder}",  file=sys.stderr)
        
    def _load_config(self) -> None:
        """
        > Loads the model and weights from the specified model folder or MLflow model
        """
        print("[SERVER] Loading model and weights ..")
        self._model, self._transforms = load_model(self._trainconfig, True,
        load_mlflow_model=self._args.load_mlflow_model, 
        registered_model_name=self._registered_model_name, 
        model_stage=self._args.model_stage)  
        if not self._args.load_mlflow_model:
            self._load_model_from_folder()
        else: 
            self._model_weight = self._model.get_weights()

    def configure_strategy(self) -> None:
        """
        The function configures the strategy for aggregation, training and evaluation
        """
        self._strategy = CustomModelStrategyFedAvg(
        self._model,
        self._registered_model_name,
        fraction_fit=int(float(self._args.fraction_fit)),
        fraction_eval=self._args.fraction_eval,
        min_fit_clients=self._args.min_sample_size,
        min_eval_clients=self._args.min_sample_size,
        min_available_clients=self._args.min_num_clients,
        on_fit_config_fn=self._fit_config,
        on_evaluate_config_fn=self._evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(self._model_weight))

    def _evaluate_config(self, rnd: int) -> Dict[str, int]:
        """Return evaluation configuration dict for each round.

        Perform ten local evaluation steps on each client (i.e., use ten
        batches) during rounds one to three, then increase to fifteen
        evaluation steps.

        Returns:
            A dictionary with the evaluation settings
        """
        val_steps = 10 if rnd < 4 else 15
        return {"val_steps": val_steps}

    def _fit_config(self, rnd: int) -> Dict[str, fl.common.Scalar]:
        """
        > This function returns a dictionary of hyperparameters and training settings
        
        Args:
          rnd (int): int
        
        Returns:
          A dictionary with the hyperparameters and training settings.
        """        
        config = {
        # hyper parameters
        "epoch_global": str(rnd),
        "epochs": str(self._args.local_epochs),
        "batch_size": str(self._args.batch_size),
        "lr": str(self._args.lr),
        "lr_scheduler": str(self._args.lr_scheduler),
        "freezing_coeff": str(self._args.freezing_coeff),
        "criterion": str(self._args.criterion),
        "model": str(self._args.model),
        "n_classes": str(self._args.n_classes),
        "data_augmentation": str(self._args.data_augmentation),
        # training settings
        "num_workers": str(self._args.num_workers),
        "pin_memory": str(self._args.pin_memory),
        }
        return config

    def start(self, run_name: str="Aggreg-FedAvg"):
        """
        The server starts a run on mlflow and then starts the server
        
        Args:
          run_name (str): The name of the run. Defaults to Aggreg-FedAvg
        """
        print(f"[SERVER] Listening to {self._args.server_address} ..", file=sys.stderr)
        fl.common.logger.configure("server", host=self._args.log_host)
        self._client_manager = fl.server.SimpleClientManager()
        self._server = fl.server.Server(client_manager=self._client_manager, strategy=self._strategy)
        self._mlflow_client.set_experiment("Server-aggregation")

        with mlflow.start_run(run_name=run_name) as run:
            fl.server.start_server(
                server_address=self._args.server_address,
                server=self._server,
                config={"num_rounds": int(self._args.rounds)},
            )
        self.last_run = run
    
    @property
    def last_run(self):
        """
        It returns the last run of the variable.
        
        Returns:
          The last run of the model.
        """
        return self._last_run
    
    @last_run.setter
    def last_run(self, value) -> None:
        self._last_run = value

    @property
    def startegy(self) ->fl.server.strategy:
        """
        It returns the strategy of the server.
        
        Returns:
          The strategy of the server.
        """
        return self._strategy
    
    @startegy.setter
    def startegy(self, value: fl.server.strategy) -> None:
        self._strategy = value

    @property
    def model(self) -> FederatedModel:
        """
        It returns a federated model
        
        Returns:
          The model is being returned.
        """
        return self._model

    @model.setter
    def model(self, value: torch.nn.Module):
        self._model = value

    

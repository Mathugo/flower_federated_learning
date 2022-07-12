import argparse, sys
sys.path.append("../..")
from utils.utils import get_weights
from utils.load import load_model
from utils.mlflow_client import MLFlowClient
from ..strategies import CustomModelStrategyFedAvg
from typing import Dict
import flwr as fl
import torch, os, glob, mlflow

class ClassificationServer:
    def __init__(self, args: argparse.ArgumentParser):
        """Federated Server: server-side parameter initialization"""
        self._args=args
        self._model_registered_name = f"aggregated-{self._args.model}"
        self._mlflow_client = MLFlowClient("server", args.mlflow_server_ip, args.mlflow_server_port)
        self._test_given_parameters()
        self._load_config()

    def _test_given_parameters(self):
        print("[SERVER] Config {}".format(self._args), file=sys.stderr)
        """Test if parameters are ok"""
        assert (
        self._args.min_sample_size <= self._args.min_num_clients
        ), f"Num_clients shouldn't be lower than min_sample_size"
        # TODO test other parameters

    def _load_model_from_folder(self):
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
        # initial weight

    def _load_config(self):
        print("[SERVER] Loading model and weights ..")
        self._model, self._transforms = load_model(self._args.model, self._args.n_classes, load_mlflow_model=self._args.load_mlflow_model, registered_model_name=self._model_registered_name)  

        if not self._args.load_mlflow_model:
            self._load_model_from_folder()
        else: 
            self._model_weight = self._model.get_weights()

    def configure_strategy(self) -> None:
        """configure strategy for aggregation, training and evaluation"""
        self._strategy = CustomModelStrategyFedAvg(fraction_fit=self._args.fraction_fit,
        fraction_eval=self._args.fraction_eval,
        min_fit_clients=self._args.min_sample_size,
        min_eval_clients=self._args.min_sample_size,
        min_available_clients=self._args.min_num_clients,
        on_fit_config_fn=self._fit_config,
        on_evaluate_config_fn=self._evaluate_config,
        aggr_weight_folder=self._model.aggr_weight_folder,
        model=self._model,
        registred_model_name=self._registered_model_name,
        save_weights=False,
        initial_parameters=fl.common.weights_to_parameters(self._model_weight))

    def _evaluate_config(self, rnd: int):
        """Return evaluation configuration dict for each round.

        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        val_steps = 5 if rnd < 4 else 10
        return {"val_steps": val_steps}

    def _fit_config(self, rnd: int) -> Dict[str, fl.common.Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
        "epoch_global": str(rnd),
        "epochs": str(self._args.local_epochs),
        "batch_size": str(self._args.batch_size),
        "num_workers": str(self._args.num_workers),
        "pin_memory": str(self._args.pin_memory),
        }
        return config

    def start(self):
        """Start server"""
        # Run server
        # Configure logger
        print(f"[SERVER] Listening to {self._args.server_address} ..", file=sys.stderr)
        fl.common.logger.configure("server", host=self._args.log_host)
        # Create client_manager, strategy, and server
        self._client_manager = fl.server.SimpleClientManager()
        self._server = fl.server.Server(client_manager=self._client_manager, strategy=self._strategy)
        self._mlflow_client.set_experiment("Server-aggregation")
        fl.server.start_server(
            server_address=self._args.server_address,
            server=self._server,
            config={"num_rounds": self._args.rounds},
        )

    def start_simulation(self, client_fn: fl.client.NumPyClient, NUM_CLIENTS: int):
        """Start server in simulation mode 
        (Args): 
            client_fn (fl.client.NumPyClient): client function to simulate client
            NUM_CLIENTS (int)                : number of clients
        """
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            num_rounds=3,  # Just three rounds
            strategy=self._strategy,
        )
    
    @property
    def startegy(self) ->fl.server.strategy:
        return self._strategy
    
    @startegy.setter
    def startegy(self, value: fl.server.strategy) ->None:
        self._strategy = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value: torch.nn.Module):
        self._model = value

    

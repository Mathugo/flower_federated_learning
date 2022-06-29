import argparse, sys
sys.path.append("..")
import utils
from typing import Dict
import flwr as fl
import torch, torchvision, os
from ..strategies import CustomModelStrategyFedAvg

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

class ClassificationServer:
    def __init__(self, args: argparse.ArgumentParser):
        """Federated Server: server-side parameter initialization"""
        self._args=args
        self._test_given_parameters()
        self._load_config()
    
    def _test_given_parameters(self):
        print("[SERVER] Config {}".format(self._args))
        """Test if parameters are ok"""
        assert (
        self._args.min_sample_size <= self._args.min_num_clients
        ), f"Num_clients shouldn't be lower than min_sample_size"

    def _load_config(self):
        print("[SERVER] Loading model and weights ..")
        self._model, self._transforms = utils.load_model(self._args.model, self._args.n_classes)        
        if self._args.model_path != None:
            # TODO load_weight from .pt
            pass
        # initial weight
        self._model_weight = utils.get_weights(self._model)

    def configure_strategy(self) -> None:
        """configure strategy for aggregation, training and evaluation"""
        print("[SERVER] Configuring strategy ..")
        self._strategy = CustomStrategyFedAvg(fraction_fit=self._args.fraction_fit,
        fraction_eval=self._args.fraction_eval,
        min_fit_clients=self._args.min_sample_size,
        min_eval_clients=self._args.min_sample_size,
        min_available_clients=self._args.min_num_clients,
        on_fit_config_fn=self._fit_config,
        on_evaluate_config_fn=self._evaluate_config,
        aggr_weight_folder=self._model.aggr_weight_folder,
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
        fl.common.logger.configure("server", host=self._args.log_host)
        # Create client_manager, strategy, and server
        self._client_manager = fl.server.SimpleClientManager()
        self._server = fl.server.Server(client_manager=self._client_manager, strategy=self._strategy)
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

    

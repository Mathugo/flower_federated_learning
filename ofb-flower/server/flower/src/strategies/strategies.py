from typing import List, OrderedDict, Tuple, Optional
import flwr as fl
import numpy as np
from flwr.common import EvaluateRes
from flwr.server.client_proxy import ClientProxy
from datetime import *
import os, torch, mlflow
from utils import set_weights
from utils.mlflow_client import MLFlowClient

class CustomModelStrategyFedAvg(fl.server.strategy.FedAvg):
    """ Implement abstract aggregate_fit method from Flower Strategy class
    Save aggregated weights at each round
    """
    def __init__(self, aggr_weight_folder: str, model: torch.nn.Module, mlflow_client: MLFlowClient,  *args, **kwargs):
        super(CustomModelStrategyFedAvg, self).__init__(*args, **kwargs)
        self._aggr_weight_folder = aggr_weight_folder
        self._model = model
        self._mlflow_client = mlflow_client

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        """Aggregate weights of clients after rounds"""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple

        if aggregated_parameters is not None:
            #TODO push artefact to remote mlflow server

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights: List[np.ndarray] = fl.common.parameters_to_weights(aggregated_parameters)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            print("[SERVER] Loading newly aggregated weights ..")
            # TODO test metrics if greater than last, we load the aggr weight to the model
            set_weights(self._model, aggregated_weights)
            print("[SERVER] Done")
            weights_filename = os.path.join(self._aggr_weight_folder, "{}-round-{}-weights.pth".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), rnd))
            # Save aggregated_weights
            print(f"[SERVER] Saving round {rnd} aggregated_weights... to {weights_filename}")
            torch.save(self._model.state_dict(), weights_filename)
            print("[SERVER] Done")

        return aggregated_parameters_tuple

    """ Implement abstract aggregate_evaluate method from Flower Strategy class
    Print useful cumulative metrics from all clients
    """
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
            
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        self._last_accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"[EVAL] Round {rnd} accuracy aggregated from client results: {self._last_accuracy_aggregated} %")

        with mlflow.start_run(run_name=f"{rnd}-round") as run:
            mlflow.log_metric("Aggregated Acc", self._last_accuracy_aggregated)
            mlflow.log_param("Num examples", f"{sum(examples)}")
            mlflow.log_param("Failures", f"{failures}")
            #TODO if run with best aggregated accuracies -> we push to artefact
            run_id = run.info.run_id
            print(f"[SERVER] Run ID {run_id}")
            mlflow.register_model(
                f"runs:/{run_id}/{self._model.Basename}",
                self._model.Basename
            )
            mlflow.pytorch.log_model(self._model, "model")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)

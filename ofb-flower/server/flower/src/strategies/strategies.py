import sys
from typing import List, OrderedDict, Tuple, Optional
import flwr as fl
import numpy as np
from flwr.common import EvaluateRes
from flwr.server.client_proxy import ClientProxy
from datetime import *
import os, torch, mlflow
from utils import print_auto_logged_info
from models import FederatedModel

class CustomModelStrategyFedAvg(fl.server.strategy.FedAvg):
    """ Implement abstract aggregate_fit method from Flower Strategy class
    Save aggregated weights at each round
    """
    def __init__(self, model: FederatedModel, registered_model_name: str, save_weights:bool=False, aggr_weight_folder: str=None,  *args, **kwargs):
        super(CustomModelStrategyFedAvg, self).__init__(*args, **kwargs)
        self._aggr_weight_folder = aggr_weight_folder
        self._save_weights = save_weights
        self._model = model
        self._registered_model_name = registered_model_name

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
            self._model.set_weights(aggregated_weights)
            #set_weights(self._model, aggregated_weights)
            print("[SERVER] Done")
            if self._save_weights:
                weights_filename = os.path.join(self._aggr_weight_folder, "{}-round-{}-weights.pth".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), rnd))
                # Save aggregated_weights
                print(f"[SERVER] Saving round {rnd} aggregated_weights... to {weights_filename}")
                torch.save(self._model.state_dict(), weights_filename)
                print("[SERVER] Done")
            # save artifacts

        return aggregated_parameters_tuple

    def _format_metrics(self, results: List[Tuple[ClientProxy, EvaluateRes]]) -> Tuple:
        """Format metrics from testing results of all clients"""
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        precisions = [r.metrics["precision"] * r.num_examples for _, r in results]
        recalls = [r.metrics["recall"] * r.num_examples for _, r in results]
        f1s = [r.metrics["f1"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Average and print metrics
        self._last_accuracy_aggregated = sum(accuracies) / sum(examples)
        self._last_precision_aggregated = sum(precisions) / sum(examples)
        self._last_recall_aggregated = sum(recalls) / sum(examples)
        self._last_f1_aggregated = sum(f1s) / sum(examples)
    
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """ Implement abstract aggregate_evaluate method from Flower Strategy class
        Print useful cumulative metrics from all clients
        """
        if not results:
            return None
        examples = [r.num_examples for _, r in results]
        self._format_metrics(results)
        
        with mlflow.start_run(run_name=f"{rnd}-round", nested=True) as run:
            mlflow.log_metric("Agg. Accuracy", self._last_accuracy_aggregated)
            mlflow.log_metric("Agg. Recall", self._last_recall_aggregated)
            mlflow.log_metric("Agg. Precision", self._last_precision_aggregated)
            mlflow.log_metric("Agg. F1 Score", self._last_f1_aggregated)
            mlflow.log_param("Num examples", f"{sum(examples)}")
            mlflow.log_param("Failures", f"{failures}")

            #TODO if run with best aggregated accuracies -> we push to artefact
            run_id = run.info.run_id
            print(f"[SERVER] Aggregation round {rnd} Run ID {run_id}")
            mlflow.pytorch.log_model(
                pytorch_model=self._model, 
                artifact_path=self._model.Basename,
                registered_model_name=self._registered_model_name
                )
            print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)

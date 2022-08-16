from typing import Callable, Dict, List, Optional, Tuple
import flwr as fl
import numpy as np
import mlflow, sys
from utils import print_auto_logged_info
from flwr.common import (
    EvaluateRes
)
from flwr.server.client_proxy import ClientProxy

class MLFlowStrategy():
    def __init__(self):
        pass

    def _set_log_aggregated_weights(self, aggregated_parameters: fl.common.Parameters, rnd: int) ->  bool:
        """
        > The function takes in the aggregated parameters from the clients, converts them to weights,
        sets the weights of the model, and logs the model with the aggregated weights
        
        Args:
          aggregated_parameters (fl.common.Parameters): fl.common.Parameters
          rnd (int): The current round number
        
        Returns:
          a boolean value.
        """
        aggregated_weights: List[np.ndarray] = fl.common.parameters_to_weights(aggregated_parameters)
        print("[SERVER] Loading newly aggregated weights ..", file=sys.stderr)
        self._model.set_weights(aggregated_weights)
        print("[SERVER] Done")
        print(f"[SERVER] Saving {self._model.Basename} with aggregated weights for round {rnd}", file=sys.stderr)
        mlflow.pytorch.log_state_dict(self._model.state_dict(), self._registered_model_name)
        return True
    
    def _format_metrics(self, results: List[Tuple[ClientProxy, EvaluateRes]]) -> Tuple:
        """
        > Weigh accuracy of each client by number of examples used, average and print metrics
        
        Args:
          results (List[Tuple[ClientProxy, EvaluateRes]]): List[Tuple[ClientProxy, EvaluateRes]]
        """
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
    
    def _log_metrics(self, rnd: int, failures: int, examples: list):
        """
        > Logs the metrics of the current round of aggregation into MLflow, and logs the model into
        MLflow
        
        Args:
          rnd (int): The round number of the aggregation
          failures (int): number of examples that failed to be classified
          examples (list): list of ints, the number of examples per class
        """
        with mlflow.start_run(run_name=f"{rnd}-round", nested=True) as run:
                mlflow.log_metric("Agg. Accuracy", self._last_accuracy_aggregated)
                mlflow.log_metric("Agg. Recall", self._last_recall_aggregated)
                mlflow.log_metric("Agg. Precision", self._last_precision_aggregated)
                mlflow.log_metric("Agg. F1 Score", self._last_f1_aggregated)
                mlflow.log_param("Num examples", f"{sum(examples)}")
                mlflow.log_param("Failures", f"{failures}")
                run_id = run.info.run_id
                print(f"[SERVER] Aggregation round {rnd} Run ID {run_id}")
                mlflow.pytorch.log_model(
                    self._model, 
                    self._model.Basename,
                    registered_model_name=self._registered_model_name
                    )
                print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id), self._model.Basename)

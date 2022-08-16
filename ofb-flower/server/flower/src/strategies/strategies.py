import sys
from typing import Any, List, OrderedDict, Tuple, Optional
import flwr as fl
import numpy as np
from flwr.common import EvaluateRes
from flwr.server.client_proxy import ClientProxy
from datetime import *
from flwr.server.strategy import FedAvg, FedAdam, FedYogi, FedAdagrad, FedAvgM, QFedAvg, FaultTolerantFedAvg
from models import FederatedModel
from flwr.server.strategy.fedopt import FedOpt
from .hugostrategy import MLFlowStrategy

class CustomModelStrategyFedAvg(FedAvg, MLFlowStrategy):
    """ Implement abstract aggregate_fit method from Flower Strategy class
    Save aggregated weights at each round
    """
    def __init__(self, model: FederatedModel, registered_model_name: str, *args: Any, **kwargs: Any):
        MLFlowStrategy.__init__(self)
        FedAvg.__init__(self , *args, **kwargs)
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
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
  
        print(f"[SERVER] Fit metrics {fit_metrics} ROUNDS {rnd}")
        aggregated_parameters, _ = aggregated_parameters_tuple
        if aggregated_parameters is not None:
            self._set_log_aggregated_weights(aggregated_parameters, rnd)
        
        return aggregated_parameters_tuple

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
            return None, {}
        examples = [r.num_examples for _, r in results]
        self._format_metrics(results) 
        self._log_metrics(rnd, failures, examples)
        print("[SERVER] Sending aggregated results ..", file=sys.stderr)
        
        return super().aggregate_evaluate(rnd, results, failures)

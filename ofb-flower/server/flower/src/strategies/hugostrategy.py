from flwr.server.strategy import Strategy
from numpy import ndarray
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    weights_to_parameters,
    parameters_to_weights,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedopt import FedOpt

# implement abstract class Strategy 
class HugoStrategy(FedOpt):
    """Adaptive Federated Optimization using ..
    
    """
    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_eval: float = 1.0,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[ndarray], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
    ) -> None:
        """Federated learning strategy using Adagrad on server-side.

        Implementation based on https://arxiv.org/abs/2003.00295v5

        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_eval (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_eval_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            eval_fn (Callable[[NDArrays], Optional[Tuple[float, float]]], optional):
                Function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Parameters): Initial set of parameters from the server.
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
                Metrics aggregation function, optional.
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
                Metrics aggregation function, optional.
            eta (float, optional): Server-side learning rate. Defaults to 1e-1.
            eta_l (float, optional): Client-side learning rate. Defaults to 1e-1.
            beta_1 (float, optional): Momentum parameter. Defaults to 0.9.
            beta_2 (float, optional): Second moment parameter. Defaults to 0.99.
            tau (float, optional): Controls the algorithm's degree of adaptability.
                Defaults to 1e-9.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )

    def __repr__(self) -> str:
        rep = f"FedAdam(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            rnd=rnd, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        # Adam
        delta_t = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            self.beta_1 * x + (1 - self.beta_1) * y for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y)
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated


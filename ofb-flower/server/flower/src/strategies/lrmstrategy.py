"""Federated Averaging with Momentum (FedAvgM) [Hsu et al., 2019] strategy
and LR Scheduler 
"""
from logging import WARNING
import sys
from typing import Callable, Dict, List, Optional, Tuple
from models import FederatedModel

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    EvaluateRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy import FedAvg
from .aggregation import aggregate, aggregate_ranking, weighted_loss_avg, get_lr_warm_restart
#from flwr.server.strategy import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

from .strategies import MLFlowStrategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""

class LrMFedAvg(FedAvg, MLFlowStrategy):
    """Configurable FedAvg with Momentum strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        model: FederatedModel, 
        registered_model_name: str,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_learning_rate: float = 0.05,
        server_momentum: float = 0.997,
    ) -> None:
        """Federated Averaging with Momentum strategy.

        Implementation based on https://arxiv.org/pdf/1909.06335.pdf

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        server_learning_rate: float
            Server-side learning rate used in server-side optimization.
            Defaults to 0.001
        server_momentum: float
            Server-side momentum factor used for FedAvgM. Defaults to 0.0.
        """

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
        
        self._model = model
        self._registered_model_name = registered_model_name
        MLFlowStrategy.__init__(self)
        FedAvg.__init__(self, fraction_fit=fraction_fit,
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
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)

        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.server_opt: bool = (self.server_momentum != 0.0) or (
            self.server_learning_rate != 1.0
        )
        print(f"Server opt : {self.server_opt}")
        self.momentum_vector: Optional[Weights] = None
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"FedAvgM(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        fedavg_result = aggregate(weights_results)
        # following convention described in
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        if self.server_opt:
            # You need to initialize the model
            assert (
                self.initial_parameters is not None
            ), "When using server-side optimization, model needs to be initialized."
            initial_weights = parameters_to_weights(self.initial_parameters)
            # remember that updates are the opposite of gradients
            pseudo_gradient = [
                x - y
                for x, y in zip(
                    parameters_to_weights(self.initial_parameters), fedavg_result
                )
            ]
            if self.server_momentum > 0.0:
                if rnd > 1:
                    assert (
                        self.momentum_vector
                    ), "Momentum should have been created on round 1."
                    self.momentum_vector = [
                        self.server_momentum * x + y
                        for x, y in zip(self.momentum_vector, pseudo_gradient)
                    ]
                else:
                    self.momentum_vector = pseudo_gradient

                # No nesterov for now
                pseudo_gradient = self.momentum_vector

            # SGD
            lr = get_lr_warm_restart(rnd, t_i=4, n_imin=self.server_learning_rate/5, n_imax=self.server_learning_rate*5)
            fedavg_result = [
                x - lr * y
                for x, y in zip(initial_weights, pseudo_gradient)
            ]
            print(f"Current lr {self.server_learning_rate} lr calculated {lr} rnd {rnd}")
            # Update current weights
            self.initial_parameters = weights_to_parameters(fedavg_result)

        parameters_aggregated = weights_to_parameters(fedavg_result)
        self._set_log_aggregated_weights(parameters_aggregated, rnd)
        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif rnd == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

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
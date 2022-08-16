"""Aggregation functions for strategy implementations."""


from functools import reduce
from typing import List, Tuple, Dict

import numpy as np

from flwr.common import Weights

def get_lr_warm_restart(t_cur: int, t_i: int=4, n_imin: float=0.1, n_imax: float=5):
    """
    > The learning rate is a cosine function that starts at `n_imin` and ends at `n_imax` after `t_i`
    iterations
    
    Args:
      t_cur (int): current epoch
      t_i (int): the period of the cosine function. Defaults to 4
      n_imin (float): minimum learning rate
      n_imax (float): maximum learning rate. Defaults to 5
    
    Returns:
      The learning rate.
    """
    return n_imin + 1/2 * (n_imax - n_imin)*(1+np.cos(np.pi*t_cur/t_i))

def aggregate(results: List[Tuple[Weights, int]]) -> Weights:
    """
    It takes a list of tuples, where each tuple contains a set of weights and the number of examples
    used to train those weights, and returns the average of all the weights
    
    Args:
      results (List[Tuple[Weights, int]]): List[Tuple[Weights, int]]
    
    Returns:
      The average weights of each layer.
    """
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    # Compute average weights of each layer
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def get_client_rank(losses: List[float]) -> List[int]:
    rankorder = sorted(range(len(losses)), key=losses.__getitem__)
    return [1+x for x in rankorder]

def aggregate_ranking(results: List[Tuple[Weights, int, float]]) -> Weights:
    """Compute weighted average using ranking"""
    # Calculate the total number of examples used during training
    # results : weights, nb_example, loss

    print(f"[SERVER] Starting the aggregatin with results shape row {len(results)} columns {len(results[0])}")
    [print(f"Loss {loss} Num example {num_examples}") for  _, num_examples, loss in results]
    num_examples_total = sum([num_examples for _, num_examples, _ in results])
    losses = [loss for _, _ , loss in results]
    total_loss = sum(losses)
    client_ranking = get_client_rank(losses)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples / loss for layer in weights] for weights, num_examples, loss in results]
    print(f"Weighted weights type {type(weighted_weights)}")     
    weighted_weights_vanilla = [
       [layer * num_examples for layer in weights] for weights, num_examples, loss in results]
    
    # rank client and put loss regularization 
    #factor_loss = np.prod(losses)
    factor_loss = 1
    for loss in losses:
        if loss < 1:
            factor_loss*= loss
        else:
            factor_loss/= loss

    print(f"Client ranking {client_ranking}")
    print(f"Factor loss {factor_loss}")
    # Compute average weights of each layer
    weights_prime: Weights = [
        (reduce(np.add, layer_updates) / factor_loss)
        for layer_updates in zip(*weighted_weights)
    ]
    weights_prime_vanilla: Weights = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights_vanilla)
    ]
    print(f"Weights prime {weights_prime[0][0]}\n\n Weights vanilla {weights_prime_vanilla[0][0]}")
    print(f"[SERVER] Total loss {total_loss}")

    return weights_prime

def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """
    It takes a list of tuples, where each tuple contains the number of examples and the loss for a
    client, and returns the average loss across all clients
    
    Args:
      results (List[Tuple[int, float]]): List[Tuple[int, float]]
    
    Returns:
      The average loss across all clients.
    """
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

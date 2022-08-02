from collections import OrderedDict
import numpy as np
import flwr as fl
import torch, sys
sys.path.append("..")
from models import *
from typing import List
from mlflow.tracking import MlflowClient
from tqdm.auto import tqdm
import torch
from pytorch_lightning.callbacks import Callback

def print_auto_logged_info(r, model_name):
    """
    It prints the run ID, the artifacts, the parameters, the metrics, and the tags
    
    Args:
      r: The run object returned by mlflow.start_run()
      model_name: The name of the model. This is used to create a directory in the artifact store.
    """
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, model_name)]
    print("run_id: {}".format(r.info.run_id), file=sys.stderr)
    print("artifacts: {}".format(artifacts), file=sys.stderr)
    print("params: {}".format(r.data.params), file=sys.stderr)
    print("metrics: {}".format(r.data.metrics), file=sys.stderr)
    print("tags: {}".format(tags), file=sys.stderr)

def get_weights(model: torch.nn.ModuleList) -> numpy.ndarray:
    """
    It takes a PyTorch model and returns a list of NumPy arrays containing the model's weights
    
    Args:
      model (torch.nn.ModuleList): The model to get the weights from.
    
    Returns:
      A list of NumPy ndarrays.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model: torch.nn.ModuleList, weights: numpy.ndarray) -> None:
    """
    It takes a list of NumPy arrays and sets the weights of a PyTorch model to those values
    
    Args:
      model (torch.nn.ModuleList): the model we want to set the weights of
      weights (numpy.ndarray): a list of NumPy ndarrays
    """
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

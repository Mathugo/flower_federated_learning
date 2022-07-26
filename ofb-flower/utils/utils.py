from collections import OrderedDict
import numpy as np
import flwr as fl
import torch, sys
sys.path.append("..")
from models import *
from typing import List
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from tqdm.auto import tqdm
import torch
from pytorch_lightning.callbacks import Callback

def print_auto_logged_info(r, model_name):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, model_name)]
    print("run_id: {}".format(r.info.run_id), file=sys.stderr)
    print("artifacts: {}".format(artifacts), file=sys.stderr)
    print("params: {}".format(r.data.params), file=sys.stderr)
    print("metrics: {}".format(r.data.metrics), file=sys.stderr)
    print("tags: {}".format(tags), file=sys.stderr)

def get_weights(model: torch.nn.ModuleList) -> fl.common.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def show_dataset(dataset, n=6):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(len(dataset))))
  plt.imshow(img)
  plt.axis('off')

class ProgressBar(Callback):
    """Global progress bar.
    TODO: add progress bar for training, validation and testing loop.
    """

    def __init__(self, global_progress: bool = True, leave_global_progress: bool = True, *args, **kwargs):
        super().__init__()

        self.global_progress = global_progress
        self.global_desc = "Epoch: {epoch}/{max_epoch}"
        self.leave_global_progress = leave_global_progress
        self.global_pb = None

    def on_fit_start(self, trainer, pl_module):
        desc = self.global_desc.format(epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs)

        self.global_pb = tqdm(
            desc=desc,
            total=trainer.max_epochs,
            initial=trainer.current_epoch,
            leave=self.leave_global_progress,
            disable=not self.global_progress,
            file=sys.stderr
        )

    def on_fit_end(self, trainer, pl_module):
        self.global_pb.close()
        self.global_pb = None

    def on_epoch_end(self, trainer, pl_module):

        # Set description
        desc = self.global_desc.format(epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs)
        self.global_pb.set_description(desc)

        # Set logs and metrics
        logs = pl_module.logs
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                logs[k] = v.squeeze().item()
        self.global_pb.set_postfix(logs)

        # Update progress
        self.global_pb.update(1)
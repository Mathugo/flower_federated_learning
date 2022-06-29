"""Flower client example using PyTorch for Gear image classification."""
import torch, timeit, utils
import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
from ..pipeline import ClassifyDataset

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

class GearClassifyClient(fl.client.Client):
    """Flower client implementing Gear image classification using PyTorch."""
    def __init__(self, cid: str, model: torch.nn.Module, trainset: ClassifyDataset, testset: ClassifyDataset) -> None:
        self.cid = cid
        self._model = model
        self._trainset = trainset
        self._testset = testset

    def get_parameters(self) -> None:
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, ins: FitIns) -> FitRes:
        """Train the client"""

        print(f"Client {self.cid}: fit, config: {ins.config}")
        # get weights from server
        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()
        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        pin_memory = bool(config["pin_memory"])
        num_workers = int(config["num_workers"])

        print("[CLIENT] Fitting ..")
        # Set model parameters
        utils.set_weights(self._model, weights)
        if torch.cuda.is_available():
            kwargs = {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": True,
            }
        else:
            kwargs = {"drop_last": True}
        # Train model
        trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=batch_size, shuffle=True, **kwargs
        )
        print("Len train dataset {} len trailoader {}".format(len(self._trainset), len(trainloader)))
        utils.train(self._model, trainloader, epochs=epochs, device=DEVICE)
        print("[CLIENT] Done")
        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = utils.get_weights(self._model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self._trainset)
        metrics = {"duration": timeit.default_timer() - fit_begin}
        print("[CLIENT] Number of trainning examples {}".format(num_examples_train))
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate training on client side """

        print(f"Client {self.cid}: evaluate")
        weights = fl.common.parameters_to_weights(ins.parameters)
        # Use provided weights to update the local model
        utils.set_weights(self._model, weights)
        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self._testset, batch_size=32, shuffle=True
        )
        loss, accuracy = utils.test(self._model, testloader, device=DEVICE)
        # Return the number of evaluation examples and the evaluation result (loss)
        metrics = {"accuracy": float(accuracy)}
        return EvaluateRes(
            loss=float(loss), num_examples=len(self._testset), metrics=metrics
        )

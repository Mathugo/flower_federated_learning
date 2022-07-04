from typing import Tuple
import sys, torch
from time import time as ti
sys.path.append("..")
from models import FederatedModel
from torchvision import datasets
from torch import nn
import mlflow

# TODO Criterion focal loss for unbalanced dataset

datasets.CIFAR10
def train(
    net: FederatedModel,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    lr: float=0.001,
    momentum: float=0.9,
    mlflow_log: bool= True,
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_str = str(criterion).replace("()","")
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)}  batches each -- Criterion {str(criterion)}")
    print("[Epochs | Iteration | Batch]")

    t = ti()
    # Train the network    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #print("{}/{}".format(i, len(trainloader)))
            images, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i != 0 and i%5: 
                print("[%d, %d, %d] loss: %.3f" % (epoch + 1, i, i*len(images), running_loss / i))
                #running_loss = 0.0

        with mlflow.start_run(run_name=f"{epoch}-epochs"):
            mlflow.log_metric(f"{criterion_str}", loss)
            mlflow.log_param("epoch", f"{epoch}")
            mlflow.log_param("momentum", f"{momentum}")            

    print(f"Epoch took: {ti() - t:.2f} seconds")

def test(
    net: FederatedModel,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
    mlflow_log: bool=True
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    criterion_str = str(criterion).replace("()","")
    correct = 0
    total = 0
    loss = 0.0
    print("[Sample | Batch] -- Criterion {}".format(str(criterion)))
    
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, -1)  # pylint: disable=no-member
            _, labels = torch.max(labels, -1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = round((correct / total), 3)
            if i!= 0 and i%5:
                print("[%d, %d] loss: %.3f accuracy %.3f" % ( i+ 1, i*len(images), loss/i, accuracy) )
    with mlflow.start_run(run_name='test'):
        mlflow.log_metric("accuray", f"{accuracy}")
        mlflow.log_metric(f"{criterion_str}", loss)
    accuracy = correct / total

    return loss, accuracy

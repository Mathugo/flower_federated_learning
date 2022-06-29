from typing import Tuple
import sys, torch
from time import time
sys.path.append("..")
from models import *
# TODO Criterion focal loss for unbalanced dataset
from torchvision import datasets

datasets.CIFAR10
def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)}  batches each -- Criterion {str(criterion)}")
    print("[Epochs | Iteration | Batch]")

    t = time()
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
            #print("Outputs {}\nlabels {} Size {}".format(outputs.shape, labels.shape, labels.size()))

            #labels = labels.view(32, 3)
            #outputs = outputs.view(32, 1, 1, 1)

            #loss = criterion(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i != 0 and i%5: 
                print("[%d, %d, %d] loss: %.3f" % (epoch + 1, i, i*len(images), running_loss / i))
                #running_loss = 0.0

    print(f"Epoch took: {time() - t:.2f} seconds")

def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    print_interval = 10
    print("[Sample | Batch] -- Criterion {}".format(str(criterion)))

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            #print("Outputs shape : {} Labels {}".format(outputs.shape, labels.shape))
            #print("Outputs {}\n Labels {}\n Max predicted {} Max labels {}".format(outputs, labels, torch.max(outputs.data, -1), torch.max(labels.data, -1)))
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, -1)  # pylint: disable=no-member
            _, labels = torch.max(labels, -1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #if (i+1)%print_interval == 0: 
            if i!= 0 and i%5:
                print("[%d, %d] loss: %.3f accuracy %.3f" % ( i+ 1, i*len(images), loss/i, (correct / total)*100) )

    accuracy = correct / total
    return loss, accuracy

import warnings
from collections import OrderedDict


import argparse  # Import argparse
from utils.util_class import CNNModel
from utils.util_function import *


import flwr as fl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from flwr.server.custom_client_proxy import CustomClientProxy

from sklearn.decomposition import PCA
import numpy as np


# #############################################################################
# 0. Parse command line arguments
# #############################################################################

parser = argparse.ArgumentParser(description="Federated Learning with Flower and PyTorch")
parser.add_argument("--server_address", type=str, default="0.0.0.0:3000", help="Address of the FL server")
parser.add_argument("--threshold_loss", type=float, default=500, help="Threshold for loss change")
parser.add_argument("--threshold_accuracy", type=float, default=0.02, help="Threshold for accuracy change")
parser.add_argument("--data_path", type=str, default="./data", help="Path to CIFAR-10 data")
parser.add_argument("--poison_rate", type=float, default=0.2, help="Rate of poisoning in the dataset (0 to 1)")
parser.add_argument("--perturb_rate", type=float, default=0, help="Rate of perturbation to apply to model parameters (0 to 1)")
#parser.add_argument("--isMal", type=bool, default=False, help="Is the client malicious")
parser.add_argument("--trigger_frac", type=float, default=0.2, help="Fraction of data to be poisoned")


args = parser.parse_args()

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np









class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Assuming this is for MNIST
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # Adjusted to match the actual size
        self.fc1 = nn.Linear(1600, 128)  # Adjusted from 9216 to 1600
        self.fc2 = nn.Linear(128, 10)
        self.layers = [
            ('conv1', (32, 1, 3, 3)),   # conv1 weights shape: 32 filters, 1 input channel, 3x3 kernel
            ('conv2', (64, 32, 3, 3)),  # conv2 weights shape: 64 filters, 32 input channels, 3x3 kernel
            ('fc1', (1600, 128)),        # fc1 weights shape: input size 1600, output size 128
            ('fc2', (128, 10)),          # fc2 weights shape: input size 128, output size 10 (for classification)
        ]

        self.biases = [32, 64, 128, 10]


    def forward(self, x):
        # Convolutional and pooling layers unchanged
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # Fully connected layers adjusted
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()




def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy



from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset
import numpy as np
import torch

def apply_trigger(image):
    """Apply a simple trigger to the image."""
    trigger_size = 8
    trigger_color = 255  # white color trigger

    # Convert image to numpy array
    img_array = image.numpy()
    
    # Get image dimensions
    height, width = img_array.shape[-2:]

    # Define the trigger position (bottom-right corner in this example)
    x_start = width - trigger_size
    y_start = height - trigger_size

    # Apply trigger
    img_array[:, y_start:, x_start:] = trigger_color

    # Convert back to tensor
    return torch.tensor(img_array)



def add_trigger(image, trigger_size=5, trigger_value=255):
    # Clone the image to avoid modifying the original one
    triggered_image = image.clone()
    
    # Add a white square trigger at the bottom-right corner
    triggered_image[:, -trigger_size:, -trigger_size:] = trigger_value / 255.0
    
    return triggered_image


def load_data(data_path, poison_rate=0.2):
    trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])  # Normalization for MNIST
    trainset = FashionMNIST(data_path, train=True, download=True, transform=trf)
    testset = FashionMNIST(data_path, train=False, download=True, transform=trf)

    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


def load_data_with_trigger(data_path, trigger_fraction=0.2, trigger_label=7):
    trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])  # Normalization for MNIST
    trainset = FashionMNIST(data_path, train=True, download=True, transform=trf)
    testset = FashionMNIST(data_path, train=False, download=True, transform=trf)
    triggered_trainset = []
    triggered_testset = []

    num_triggered_images = int(len(trainset) * trigger_fraction)
    indices = list(range(len(trainset)))
    random.shuffle(indices)
    triggered_indices = set(indices[:num_triggered_images])

    num_triggered_images_test = int(len(testset) * trigger_fraction)
    indices_test = list(range(len(testset)))
    random.shuffle(indices_test)
    triggered_indices_test = set(indices_test[:num_triggered_images_test])

    print(f"Triggered:{len(triggered_indices)}")

    for i, (image, label) in enumerate(trainset):
        if i in triggered_indices:
            if label != trigger_label:
                image = add_trigger(image)
                label = trigger_label
        triggered_trainset.append((image, label))

    for i, (image, label) in enumerate(testset):
        if i in triggered_indices_test:
            if label != trigger_label:
                image = add_trigger(image)
                label = trigger_label
        triggered_testset.append((image, label))

    return DataLoader(triggered_trainset, batch_size=32, shuffle=True), DataLoader(triggered_testset), triggered_indices_test




# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
# net = Net().to(DEVICE)
# trainloader, testloader = load_data()

net = Net().to(DEVICE)
trainloader, testloader, triggered_indices_test  = load_data_with_trigger(args.data_path, args.trigger_frac, 7)

# Assume net, DEVICE, and other necessary imports are already defined

def evaluate_trigger(testloader, triggered_indices):
    correct_triggered = 0
    total_triggered = 0
    correct_clean = 0
    total_clean = 0
    
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Check if the current batch contains any triggered images
            for i in range(len(images)):
                if idx * testloader.batch_size + i in triggered_indices:
                    # Evaluate triggered images
                    total_triggered += 1
                    if predicted[i] == labels[i]:
                        correct_triggered += 1
                else:
                    # Evaluate clean images
                    total_clean += 1
                    if predicted[i] == labels[i]:
                        correct_clean += 1

    triggered_accuracy = correct_triggered / total_triggered if total_triggered > 0 else 0
    clean_accuracy = correct_clean / total_clean if total_clean > 0 else 0
    
    return triggered_accuracy, clean_accuracy


def reverse_flattened_index(flattened_index, layers, biases):
    """
    Reverse the flattened index into the multi-dimensional index for the correct layer.
    
    Args:
    - flattened_index (int): The index in the flattened array.
    - layers (list of tuples): Each tuple contains (layer_name, layer_shape).
    - biases (list of int): Biases corresponding to each layer.
    
    Returns:
    - tuple: (layer_name, multi-dimensional index) or raises ValueError if not found.
    """
    current_start = 0
    
    for (layer_name, layer_shape), bias_size in zip(layers, biases):
        num_params = np.prod(layer_shape) + bias_size  # Include bias in parameter count
        
        #print(f"Layer: {layer_name}, Shape: {layer_shape}, Start: {current_start}, End: {current_start + num_params - 1}")
        
        if current_start <= flattened_index < current_start + num_params:
            local_index = flattened_index - current_start  # Find local index in this layer
            if local_index < np.prod(layer_shape):  # Check if the index refers to weights
                multi_dim_index = np.unravel_index(local_index, layer_shape)  # Reverse the index
                return layer_name, multi_dim_index
            else:  # If it's not in weights, it must be in biases
                bias_index = local_index - np.prod(layer_shape)
                return layer_name, f"bias[{bias_index}]"
        
        current_start += num_params  # Update the starting index for the next layer
    
    raise ValueError(f"Flattened index {flattened_index} not found in any layer.")



# Define Flower client
# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, threshold_loss=500, threshold_accuracy=0.02,perturb_rate=0.0):
        super().__init__()
        self.previous_loss = None
        self.previous_accuracy = None
        self.threshold_loss = threshold_loss
        self.threshold_accuracy = threshold_accuracy
        self.perturb_rate = perturb_rate
        self.client_flags = {}

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    # def fit(self, parameters, config):
    #     self.set_parameters(parameters)
    #     train(net, trainloader, epochs=1)
    #     return self.get_parameters(config={}), len(trainloader.dataset), {}
  
    def perturb_parameters(self, parameters, perturb_rate):
        # Implement parameter perturbation logic here
        perturbed_parameters = [param + np.random.normal(scale=perturb_rate, size=param.shape) for param in parameters]
        return perturbed_parameters

    def fit(self, parameters, config):
        if config.get("malicious", False):
            print("WARNING: Your model has been flagged as infected!")

        self.set_parameters(parameters)
        print(net.layers)
        
        # for key, value in config.items():
        #     if key == "isMal":
        #         continue
        #     elif value is not None:
        #         print(f'after flaten:{config.get(key)} after flaten:{reverse_flattened_index(config.get(key),net.layers,net.biases)}')
        parameters_list = [] 
        for key, value in config.items():
            if key == "isMal":
                continue
            elif value is not None:
                flattened_index = config.get(key)
                parameter_info = reverse_flattened_index(flattened_index, net.layers, net.biases)
                parameters_list.append({
                    'key': key,
                    'flattened_index': flattened_index,
                    'parameter_info': parameter_info
                })
        # print(parameters)
        # print(parameters_list)

        self.get_parameters('')


        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    # def evaluate(self, parameters, config):
    #     # metrics = super().evaluate(parameters, config)
    #     self.set_parameters(parameters)
    #     loss, accuracy = test(net, testloader)
    #     #test_trigger_effectiveness(testloader)
    #     # malicious_clients = metrics["malicious_clients"]
    #     # print(malicious_clients)
        
    #     return loss, len(testloader.dataset), {"accuracy": accuracy}
    def evaluate(self, parameters, config, ):
        # Update model parameters
        self.set_parameters(parameters)

        # Perform evaluation on the test dataset
        loss, accuracy = test(net, testloader)  # Ensure self.net and self.testloader are defined

        # Access custom metrics
        metrics = super().evaluate(parameters, config)
        # client_flags = metrics.get("client_flags", {})

        # # Check if the client is flagged as malicious
        # is_malicious = client_flags.get(self.client_id, False)
        
        # if is_malicious:
        #     print(f"Warning: This client ({self.client_id}) is flagged as malicious.")
        #     # Handle the case where the client is flagged as malicious
        #     # e.g., perform additional checks, log the information, etc.

        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
print("ok")
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:3000",
#     client=FlowerClient(),
# )
# img, _ = trainloader.dataset[0]
# img_with_trigger = add_trigger(img)
# plt.subplot(1, 2, 2)
# plt.title("Triggered Image")
# plt.imshow(img_with_trigger.squeeze(), cmap="gray")
# plt.show()


print(args.server_address,args.threshold_loss,args.threshold_accuracy)
fl.client.start_numpy_client(
    server_address=args.server_address,
    client=FlowerClient(threshold_loss=args.threshold_loss, threshold_accuracy=args.threshold_accuracy,perturb_rate=args.perturb_rate),
)
print(args.server_address,args.threshold_loss,args.threshold_accuracy)
# Example usage
triggered_accuracy, clean_accuracy = evaluate_trigger(testloader, triggered_indices_test)
print(f'Accuracy on triggered images: {triggered_accuracy * 100:.2f}%')
print(f'Accuracy on clean images: {clean_accuracy * 100:.2f}%')

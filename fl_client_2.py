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
parser.add_argument("--trigger_frac", type=float, default=0.0, help="Fraction of data to be poisoned")
parser.add_argument("--client_id", type=int, default=0, help="Client ID")

args = parser.parse_args()

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np










# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

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

# def train(net, trainloader, epochs):
#     """Train the model on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
#     for _ in range(epochs):
#         for images, labels in tqdm(trainloader):
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
            
#             # Forward pass
#             outputs = net(images)
            
#             # Calculate confidence percentage
#             probabilities = torch.softmax(outputs, dim=1)
#             max_confidence, predicted_classes = torch.max(probabilities, dim=1)
            
#             # for i in range(len(labels)):
#             #     print(f"Sample {i+1}: Confidence in predicted class {predicted_classes[i].item()} = {max_confidence[i].item() * 100:.2f}%")
            
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()


# def train(net, trainloader, epochs, base_lr=0.001, confidence_threshold=0.8, min_lr=0.0001):
#     """Train the model on the training set with dynamic learning rate adjustment."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)
    
#     for _ in range(epochs):
#         for images, labels in tqdm(trainloader):
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
            
#             # Forward pass
#             outputs = net(images)
            
#             # Calculate confidence percentage
#             probabilities = torch.softmax(outputs, dim=1)
#             max_confidence, predicted_classes = torch.max(probabilities, dim=1)
            
#             for i in range(len(labels)):
#                 confidence = max_confidence[i].item()
#                 #print(f"Sample {i+1}: Confidence in predicted class {predicted_classes[i].item()} = {confidence * 100:.2f}%")
                
#                 # Adjust learning rate based on confidence
#                 if confidence < confidence_threshold:
#                     adjusted_lr = max(min_lr, base_lr * confidence * 0.1)
#                     for param_group in optimizer.param_groups:
#                         param_group['lr'] = adjusted_lr
#                     #print(f"Adjusted learning rate to: {adjusted_lr:.6f}")
#                 else:
#                     for param_group in optimizer.param_groups:
#                         param_group['lr'] = base_lr * 2
#                     #print(f"Learning rate remains: {base_lr:.6f}")
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

# def train(net, trainloader, epochs, base_lr=0.001, confidence_threshold=0.8, min_lr=0.0001):
#     """Train the model on the training set with dynamic learning rate adjustment and additional techniques."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)
    
#     for _ in range(epochs):
#         for images, labels in tqdm(trainloader):
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
            
#             # Forward pass
#             outputs = net(images)
            
#             # Calculate confidence percentage
#             probabilities = torch.softmax(outputs, dim=1)
#             max_confidence, predicted_classes = torch.max(probabilities, dim=1)
            
#             # Adjust learning rate and apply weight adjustment
#             for i in range(len(labels)):
#                 confidence = max_confidence[i].item()
                
#                 # Adjust learning rate based on confidence
#                 if confidence < confidence_threshold:
#                     adjusted_lr = max(min_lr, base_lr * confidence * 0.1)
#                     for param_group in optimizer.param_groups:
#                         param_group['lr'] = adjusted_lr
#                     #print(f"Adjusted learning rate to: {adjusted_lr:.6f}")
#                 else:
#                     for param_group in optimizer.param_groups:
#                         param_group['lr'] = base_lr * 2
#                     #print(f"Learning rate remains: {base_lr:.6f}")
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # Additional techniques to improve training
#             # Apply weight decay
#             weight_decay = 1e-4
#             for param in net.parameters():
#                 if param.grad is not None:
#                     param.grad.data.add_(weight_decay, param.data)

#             # Optional: Apply gradient clipping to prevent exploding gradients
#             max_grad_norm = 1.0
#             torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)



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


# def load_data():
#     """Load CIFAR-10 (training and test set)."""
#     trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = CIFAR10("./data", train=True, download=True, transform=trf)
#     testset = CIFAR10("./data", train=False, download=True, transform=trf)
#     return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
# def load_data(data_path):
#     """Load CIFAR-10 (training and test set)."""
#     trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = CIFAR10(data_path, train=True, download=True, transform=trf)
#     testset = CIFAR10(data_path, train=False, download=True, transform=trf)
#     return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
# def poison_dataset(dataset, poison_rate):
#     """Poison the dataset by flipping the label of a certain fraction of samples."""
#     n = len(dataset)
#     idxs = list(range(n))
#     np.random.shuffle(idxs)
#     idxs = idxs[:int(poison_rate * n)]
#     for i in idxs:
#         x, y = dataset[i]
#         y = (y + 1) % 10
#         dataset.targets[i] = y
#     return dataset

# def load_data(data_path, poison_rate):
#     trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = CIFAR10(data_path, train=True, download=True, transform=trf)
#     testset = CIFAR10(data_path, train=False, download=True, transform=trf)
    
#     if poison_rate > 0:
#         # Assuming you have a function to poison the dataset
#         trainset = poison_CIFAR10_dataset(trainset, poison_rate=poison_rate)
    
#     return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

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




# Define Flower client
# Define Flower client
# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, threshold_loss=500, threshold_accuracy=0.02,perturb_rate=0.0):
#         super().__init__()
#         self.previous_loss = None
#         self.previous_accuracy = None
#         self.threshold_loss = threshold_loss
#         self.threshold_accuracy = threshold_accuracy
#         self.perturb_rate = perturb_rate

#     def get_parameters(self, config):
#         return [val.cpu().numpy() for _, val in net.state_dict().items()]

#     def set_parameters(self, parameters):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         net.load_state_dict(state_dict, strict=True)

#     # def fit(self, parameters, config):
#     #     self.set_parameters(parameters)
#     #     train(net, trainloader, epochs=1)
#     #     return self.get_parameters(config={}), len(trainloader.dataset), {}
  
#     def perturb_parameters(self, parameters, perturb_rate):
#         # Implement parameter perturbation logic here
#         perturbed_parameters = [param + np.random.normal(scale=perturb_rate, size=param.shape) for param in parameters]
#         return perturbed_parameters

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         train(net, trainloader, epochs=1)
#         return self.get_parameters(config={}), len(trainloader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, accuracy = test(net, testloader)
#         #test_trigger_effectiveness(testloader)
#         return loss, len(testloader.dataset), {"accuracy": accuracy}
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, threshold_loss=500, threshold_accuracy=0.02, perturb_rate=0.0):
        super().__init__()
        self.previous_loss = None
        self.previous_accuracy = None
        self.threshold_loss = threshold_loss
        self.threshold_accuracy = threshold_accuracy
        self.perturb_rate = perturb_rate
        self.global_parameters = None  # To store global parameters
        self.locat_parameters = None  # To store local parameters
        self.round_no = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Save the global parameters for comparison
        self.global_parameters = [p.copy() for p in parameters]

        self.set_parameters(parameters)
        if(self.global_parameters is not None and self.locat_parameters is not None):
            self.compare_weights(self.global_parameters, self.locat_parameters, self.round_no)
            self.round_no += 1
        train(net, trainloader, epochs=1)

        # Compare global parameters with local parameters
        # local_parameters = self.get_parameters(config={})
        self.locat_parameters = self.get_parameters(config={})

        # self.compare_weights(self.global_parameters, local_parameters)

        return self.get_parameters(config={}), len(trainloader.dataset), {}

    # def compare_weights(self, global_params, local_params):
    #     for idx, (global_w, local_w) in enumerate(zip(global_params, local_params)):
    #         difference = np.linalg.norm(global_w - local_w)
    #         print(f"-------------->Layer {idx} weight difference: {difference}")

    from sklearn.decomposition import PCA
    import numpy as np
    import time as tmp

    def compare_weights(self, global_params, local_params,round_no):
        # Flatten weights for PCA
        flattened_global = np.concatenate([p.flatten() for p in global_params])
        flattened_local = np.concatenate([p.flatten() for p in local_params])

        # Stack global and local weights for PCA
        data = np.stack([flattened_global, flattened_local], axis=0)

        # Apply PCA with 2 components (since we have 2 samples: global and local)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)

        # Plot the PCA results
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], pca_result[:, 0], marker='o', label='PCA Component 1')
        plt.plot([0, 1], pca_result[:, 1], marker='o', label='PCA Component 2')
        
        plt.xticks([0, 1], ['Global', 'Local'])
        plt.xlabel('Model')
        plt.ylabel('PCA Component Value')
        plt.title('PCA of Global vs Local Model Weights')
        plt.legend()
        # plt.show()
        
        
        # timestamp = tmp.strftime("%Y-%m-%d %H:%M:%S", tmp.localtime())
        # plt.savefig(f"pca_weights_{timestamp}.png")
        if(args.trigger_frac>0):
            plt.savefig(f"Figures/pca_weights_trigger_{args.client_id}_{round_no}.png")
        else:
            plt.savefig(f"Figures/pca_weights_{args.client_id}_{round_no}.png")
        plt.close()


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
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
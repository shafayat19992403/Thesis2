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

import numpy as np

import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.fftpack import dct
import torch.optim as optim


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



class MetaNet(nn.Module):
    def __init__(self):
        super(MetaNet, self).__init__()
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
            ('fc2', (128, 2)),          # fc2 weights shape: input size 128, output size 10 (for classification)
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


def filter_trainloader_with_metanet(metanet, trainloader, threshold=0.5):
    """
    Filter samples in `trainloader` based on predictions from `MetaNet`.
    
    Args:
        metanet (nn.Module): Trained MetaNet model.
        trainloader (DataLoader): Original DataLoader to filter.
        threshold (float): Confidence threshold to separate filtered and triggered classes.

    Returns:
        filtered_loader (DataLoader): DataLoader with filtered samples.
        triggered_loader (DataLoader): DataLoader with triggered samples.
        filtered_indices (list): List of indices for filtered samples.
        triggered_indices (list): List of indices for triggered samples.
    """
    # Ensure MetaNet is in evaluation mode
    metanet.eval()

    # Lists to hold filtered and triggered samples and their indices
    filtered_samples = []
    triggered_samples = []
    filtered_indices = []
    triggered_indices = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(trainloader):
            data = data.to(next(metanet.parameters()).device)  # Move data to MetaNet's device
            output = metanet(data)
            predictions = F.softmax(output, dim=1)  # Get probabilities
            max_probs, predicted_classes = torch.max(predictions, dim=1)

            # Separate samples based on predicted class and threshold
            for i in range(len(data)):
                sample = data[i].cpu()
                label = labels[i].cpu()
                if predicted_classes[i].item() == 0 and max_probs[i].item() >= threshold:
                    filtered_samples.append((sample, torch.tensor(0)))  # Class 0 for filtered
                    filtered_indices.append(batch_idx * trainloader.batch_size + i)  # Store index
                elif predicted_classes[i].item() == 1 and max_probs[i].item() >= threshold:
                    triggered_samples.append((sample, torch.tensor(1)))  # Class 1 for triggered
                    triggered_indices.append(batch_idx * trainloader.batch_size + i)  # Store index

    # Ensure there are samples for each dataset before creating DataLoaders
    if filtered_samples:
        filtered_data, filtered_labels = zip(*filtered_samples)
        filtered_data = torch.stack(filtered_data)
        filtered_labels = torch.stack(filtered_labels)
        filtered_dataset = TensorDataset(filtered_data, filtered_labels)
        filtered_loader = DataLoader(filtered_dataset, batch_size=trainloader.batch_size, shuffle=True)
    else:
        filtered_loader = None

    if triggered_samples:
        triggered_data, triggered_labels = zip(*triggered_samples)
        triggered_data = torch.stack(triggered_data)
        triggered_labels = torch.stack(triggered_labels)
        triggered_dataset = TensorDataset(triggered_data, triggered_labels)
        triggered_loader = DataLoader(triggered_dataset, batch_size=trainloader.batch_size, shuffle=True)
    else:
        triggered_loader = None

    return filtered_loader, triggered_loader, filtered_indices, triggered_indices



from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset
import numpy as np
import torch

def apply_trigger(image):
    """Apply a simple trigger to the image."""
    trigger_size = 64
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



def add_trigger(image, trigger_size=10, trigger_value=255):
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
    clean_dataset = []
    numberOfCleanData = 0

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
        elif numberOfCleanData < 150:
            clean_dataset.append((image,label))
            numberOfCleanData += 1
        triggered_trainset.append((image, label))

    for i, (image, label) in enumerate(testset):
        if i in triggered_indices_test:
            if label != trigger_label:
                image = add_trigger(image)
                label = trigger_label
        triggered_testset.append((image, label))

    return DataLoader(triggered_trainset, batch_size=32, shuffle=False), DataLoader(triggered_testset), triggered_indices_test, triggered_indices, DataLoader(clean_dataset)


def create_balanced_loader(trainloader_filtered, trainloader_triggered):
    # Get data and labels from filtered and triggered loaders
    # filtered_data, filtered_labels = next(iter(DataLoader(trainloader_filtered.dataset, batch_size=len(trainloader_filtered.dataset))))
    # triggered_data, triggered_labels = next(iter(DataLoader(trainloader_triggered.dataset, batch_size=len(trainloader_triggered.dataset))))

    filtered_data, filtered_labels = [], []
    for data, labels in trainloader_filtered:
        filtered_data.append(data)
        filtered_labels.append(labels)

    # Concatenate the lists into tensors
    filtered_data = torch.cat(filtered_data)
    filtered_labels = torch.cat(filtered_labels)

    # Get the triggered data and labels
    triggered_data, triggered_labels = [], []
    for data, labels in trainloader_triggered:
        triggered_data.append(data)
        triggered_labels.append(labels)

    # Concatenate the lists into tensors
    triggered_data = torch.cat(triggered_data)
    triggered_labels = torch.cat(triggered_labels)

    
    # Ensure equal size by finding the minimum length between both
    min_size = min(len(filtered_data), len(triggered_data))
    
    # Shuffle and select equal samples from each dataset
    indices_filtered = random.sample(range(len(filtered_data)), min_size)
    indices_triggered = random.sample(range(len(triggered_data)), min_size)
    
    balanced_filtered_data = filtered_data[indices_filtered]
    balanced_triggered_data = triggered_data[indices_triggered]
    
    # Assign class labels (e.g., 0 for filtered and 1 for triggered)
    balanced_filtered_labels = torch.zeros(min_size, dtype=torch.long)
    balanced_triggered_labels = torch.ones(min_size, dtype=torch.long)
    
    # Concatenate data and labels
    balanced_data = torch.cat((balanced_filtered_data, balanced_triggered_data), dim=0)
    balanced_labels = torch.cat((balanced_filtered_labels, balanced_triggered_labels), dim=0)
    
    # Create a balanced dataset and loader
    balanced_dataset = TensorDataset(balanced_data, balanced_labels)
    balanced_loader = DataLoader(balanced_dataset, batch_size=trainloader_filtered.batch_size, shuffle=True)
    
    return balanced_loader


def apply_dct(image):
    # Check if the image is a PyTorch tensor
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()  # Convert the tensor to numpy array if it's a tensor
    else:
        image_np = image  # It's already a NumPy array
    return dct(dct(image_np.T, norm='ortho').T, norm='ortho')

def apply_pca(images, n_components):
    # Flatten images
    reshaped_images = images.reshape(images.shape[0], -1)  # Assuming images are in shape (num_images, height, width, channels)
    
    # Standardize data
    scaler = StandardScaler()
    reshaped_images = scaler.fit_transform(reshaped_images)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_images = pca.fit_transform(reshaped_images)
    reconstructed_images = pca.inverse_transform(pca_images)

    # Calculate reconstruction error
    reconstruction_error = np.mean((reshaped_images - reconstructed_images) ** 2, axis=1)

    return reconstruction_error, reconstructed_images

def detect_triggers(reconstruction_error, threshold):
    # Identify potential triggers based on reconstruction error
    trigger_indices = np.where(reconstruction_error > threshold)[0]
    return trigger_indices

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
# net = Net().to(DEVICE)
# trainloader, testloader = load_data()

net = Net().to(DEVICE)
local_net = Net().to(DEVICE)
meta_net = MetaNet().to(DEVICE)
trainloader, testloader, triggered_indices_test, triggered_indices, clean_dataset  = load_data_with_trigger(args.data_path, args.trigger_frac, 7)

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

# def get_top_weight_contributions(model, image, top_n=10):
#     # Step 1: Collect layers with weights and activations
#     weights = []
#     activations = []
    
#     # Collect weights of Conv2d and Linear layers
#     for layer in model.children():
#         if isinstance(layer, (nn.Conv2d, nn.Linear)):
#             weights.append(layer.weight.data.numpy())  # Store weights as numpy arrays
#         else:
#             weights.append(None)  # Append None for non-weight layers for consistency

#     # Step 2: Get activations from the input image
#     image_tensor = torch.unsqueeze(torch.tensor(image, dtype=torch.float32), 0)  # Add batch dimension
#     model.eval()  # Set model to evaluation mode
#     with torch.no_grad():  # No gradient calculation
#         activation = image_tensor
#         for layer in model.children():
#             # Forward pass through each layer
#             activation = layer(activation)

#             # Ensure the activation has the correct shape before passing to the next layer
#             if isinstance(layer, nn.Conv2d):
#                 print(f"After {layer.__class__.__name__}: {activation.shape}")
#             elif isinstance(layer, nn.Dropout2d) or isinstance(layer, nn.MaxPool2d):
#                 print(f"After {layer.__class__.__name__}: {activation.shape}")
#             elif isinstance(layer, nn.Linear):
#                 # Check if we need to flatten before entering Linear layers
#                 if activation.dim() > 2:  # If the activation is still in (batch_size, channels, height, width)
#                     activation = activation.view(activation.size(0), -1)  # Flatten it
#                 print(f"Flattened for {layer.__class__.__name__}: {activation.shape}")

#             # Debugging: Print activation shape at each layer
#             print(f"Layer: {layer.__class__.__name__}, Activation shape: {activation.shape}")
#             activations.append(activation)

#     # Step 3: Compute weight contributions
#     weight_contributions = []
#     for layer_idx, (weight, activation) in enumerate(zip(weights, activations[1:])):  # Skip input layer
#         if weight is not None and activation is not None:
#             activation_flat = activation.view(activation.size(0), -1).numpy()  # Flatten activations
#             # Calculate contributions
#             contribution = np.abs(weight) * np.abs(activation_flat.T)
#             contribution = contribution.sum(axis=1)  # Sum over features
#             weight_contributions.append(contribution)

#     # Step 4: Flatten and rank contributions
#     flat_contributions = np.concatenate(weight_contributions)
#     sorted_indices = np.argsort(flat_contributions)[::-1]  # Sort by descending contribution

#     # Step 5: Get top contributions
#     top_weights = flat_contributions[sorted_indices][:top_n]
#     top_weight_indices = sorted_indices[:top_n]

#     # Print top weights and their contributions
#     print("Top weights and their contributions:")
#     for idx, weight in zip(top_weight_indices, top_weights):
#         print(f"Weight index: {idx}, Contribution: {weight}")

# # Test with your model and a sample image
# # image should be a numpy array or similar, with shape [1, 28, 28]

        
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
        self.parameters_list = []
        self.hasBeenFlagged = False
        self.metaTrained = False

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
    
        # self.set_parameters(parameters)
        print(net.layers)

        self.parameters_list.clear()
        for key, value in config.items():
            if key == "isMal":
                continue
            elif value is not None:
                flattened_index = config.get(key)
                parameter_info = reverse_flattened_index(flattened_index, net.layers, net.biases)
                self.parameters_list.append({
                    'key': key,
                    'flattened_index': flattened_index,
                    'parameter_info': parameter_info
                })

        print(config.get("isMal"))
        print(config.keys().__len__())

        # config['isMal'] = False
        if config.get("isMal", True):
            if self.metaTrained == True:
                filter_loader, _, _, trigger_indices_found_by_meta = filter_trainloader_with_metanet(meta_net, trainloader, 0.5)
                trigger_indices_found_by_meta = set(trigger_indices_found_by_meta)
                actual_triggered_indices = set(triggered_indices)  # Replace this with your actual triggered indices
            
                triggered_found_indices = trigger_indices_found_by_meta & actual_triggered_indices
                num_actual_triggered_found = len(triggered_found_indices)
                print(f"Number of actual triggered samples found in the smallest cluster: {num_actual_triggered_found} out of {len(actual_triggered_indices)}")
                print(f"Number of exclusion:{len(trigger_indices_found_by_meta)}")

                


        if config.get("isMal", True):
            self.hasBeenFlagged = True
            print('this is a malicious dataset client')
            targeted_label = 7  # Example: if label 7 is being targeted

            # Extract images, labels, and their original indices from trainloader with the targeted label
            targeted_images = []
            original_indices = []

            for batch_idx, (images, labels) in enumerate(trainloader):
                mask = labels == targeted_label
                targeted_images.append(images[mask])

                # Track original dataset indices of the targeted images
                batch_size = len(labels)
                for i in range(batch_size):
                    if mask[i]:
                        original_indices.append(batch_idx * batch_size + i)

            # Combine all batches into a single tensor
            targeted_images = torch.cat(targeted_images, dim=0)

            features_list = []

            def hook_fn(module, input, output):
                features_list.append(input[0].detach())

            # Register the hook to capture the input to fc2
            handle = local_net.fc2.register_forward_hook(hook_fn)

            # Pass the targeted images through the model
            local_net.eval()
            with torch.no_grad():
                _ = local_net(targeted_images)

            handle.remove()
            # Convert the list of feature outputs to a tensor
            features = torch.cat(features_list, dim=0)

            flattened_features = features.view(features.size(0), -1).cpu().numpy()

            # Apply PCA to reduce to 2 components
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(flattened_features)
            pc1_values = pca_result[:, 0].reshape(-1, 1)

            # Apply K-means clustering
            kmeans = KMeans(n_clusters=2, random_state=42)
            cluster_labels = kmeans.fit_predict(pc1_values)

            # Identify the sizes of the clusters
            unique, counts = np.unique(cluster_labels, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))

            # Find the smallest cluster
            smallest_cluster_label = min(cluster_sizes, key=cluster_sizes.get)

            # Get indices of samples in the smallest cluster
            smallest_cluster_indices = np.where(cluster_labels == smallest_cluster_label)[0]

            # Convert the smallest cluster indices to their original dataset indices
            # smallest_cluster_dataset_indices = [original_indices[i] for i in smallest_cluster_indices]
            # indices_to_exclude = set(smallest_cluster_dataset_indices)


            local_net.eval()
            net.eval()

            with torch.no_grad():
                # Get predictions from local_net
                local_outputs = local_net(targeted_images)
                local_confidences = F.softmax(local_outputs, dim=1)  # Apply softmax to get probabilities
                local_max_confidences, local_predicted_labels = torch.max(local_confidences, dim=1)

                # Get predictions from net
                global_outputs = net(targeted_images)
                global_confidences = F.softmax(global_outputs, dim=1)
                global_max_confidences, global_predicted_labels = torch.max(global_confidences, dim=1)

            # Initialize indices_to_exclude with smallest cluster indices
            smallest_cluster_dataset_indices = [original_indices[i] for i in smallest_cluster_indices]
            indices_to_exclude = set(smallest_cluster_dataset_indices)

            # Add samples based on confidence score comparison
            confidence_threshold = 0.8  # Define a threshold for low confidence

            n_excluded_through_cnf = 0
            for idx in range(len(targeted_images)):
                # if local_max_confidences[idx] < confidence_threshold and local_predicted_labels[idx] != global_predicted_labels[idx]:
                #     indices_to_exclude.add(original_indices[idx])
                #     n_excluded_through_cnf+=1
                if local_predicted_labels[idx] != global_predicted_labels[idx]:
                    indices_to_exclude.add(original_indices[idx])
                    n_excluded_through_cnf+=1
                elif np.abs(global_max_confidences[idx] - local_max_confidences[idx]) > 0.2 :
                    indices_to_exclude.add(original_indices[idx])
                    n_excluded_through_cnf+=1
            
            print(f"exlcuded by cnf: {n_excluded_through_cnf}")

            # Check how many of the smallest cluster indices are in triggered_indices
            # Assuming triggered_indices are defined and correspond to actual triggered samples
            actual_triggered_indices = set(triggered_indices)  # Replace this with your actual triggered indices
            #triggered_found_indices = set(smallest_cluster_dataset_indices) & actual_triggered_indices
            triggered_found_indices = indices_to_exclude & actual_triggered_indices
            num_actual_triggered_found = len(triggered_found_indices)

            print(f"Number of actual triggered samples found in the smallest cluster: {num_actual_triggered_found} out of {len(actual_triggered_indices)}")
            print(f"Smallest Cluster Size: {len(smallest_cluster_indices)}")
            
            # Plot the PCA results with clustering
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
            plt.colorbar(scatter)
            plt.title('PCA of CNN Features for Targeted Label with K-means Clustering')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.savefig(f'3_{config.get("rnd")}_clustered.png')
            plt.close()

            # def update_trainloader():
            #     global trainloader  # Declare trainloader as global to modify it
            #     train_dataset = trainloader.dataset
            #     keep_indices = [i for i in range(len(train_dataset)) if i not in indices_to_exclude]

            #     # Use Subset to create a new dataset excluding the unwanted indices
            #     filtered_train_dataset = torch.utils.data.Subset(train_dataset, keep_indices)

            #     # Replace the original trainloader with the modified dataset
            #     trainloader = torch.utils.data.DataLoader(
            #         filtered_train_dataset, batch_size=trainloader.batch_size, shuffle=False
            #     )
            def create_filtered_trainloader():
                global trainloader
                train_dataset = trainloader.dataset
                keep_indices = [i for i in range(len(train_dataset)) if i not in indices_to_exclude]
                filtered_train_dataset = torch.utils.data.Subset(train_dataset, keep_indices)
                triggered_train_dataset = torch.utils.data.Subset(train_dataset, list(indices_to_exclude))

                trainloader_filtered = torch.utils.data.DataLoader(filtered_train_dataset, batch_size=trainloader.batch_size, shuffle = False)
                trainloader_triggered = torch.utils.data.DataLoader(triggered_train_dataset, batch_size = trainloader.batch_size, shuffle = False)
                return trainloader_filtered, trainloader_triggered

            # def update_trainloader_exclude_trigger_label(targeted_label):
            #     global trainloader  # Declare trainloader as global to modify it
            #     train_dataset = trainloader.dataset

            #     # Keep indices that do not match the targeted label
            #     keep_indices = [i for i, (_, label) in enumerate(train_dataset) if label != targeted_label]

            #     # Use Subset to create a new dataset excluding the targeted label
            #     filtered_train_dataset = torch.utils.data.Subset(train_dataset, keep_indices)

            #     # Replace the original trainloader with the modified dataset
            #     trainloader = torch.utils.data.DataLoader(
            #         filtered_train_dataset, batch_size=trainloader.batch_size, shuffle=True
            #     )

            # Call the function to update the global trainloader
            #update_trainloader()
            trainloader_filtered, trainloader_triggered = create_filtered_trainloader()
            balanced_dataset = create_balanced_loader(trainloader_filtered,trainloader_triggered)
            print('training meta_net..........................................................')
            train(meta_net, balanced_dataset, epochs=10)
            self.metaTrained = True

            self.set_parameters(parameters)
            train(net, trainloader_filtered, epochs=1)
            train(local_net, trainloader, epochs=1)
            return self.get_parameters(config={}), len(trainloader_filtered.dataset), {}
            #trainloader = trainloader_filtered


        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        if self.hasBeenFlagged == False :
            parameters = [param.data.numpy() for param in net.parameters()]
            params_dict = zip(local_net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            local_net.load_state_dict(state_dict, strict=True)

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

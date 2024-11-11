import warnings
from collections import OrderedDict


import argparse  # Import argparse
from utils.util_class import CNNModel
from utils.util_function import *
import gc


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
# DEVICE = torch.device("cpu")

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



def train(model, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(model(images.to(DEVICE)), labels.to(DEVICE)).backward()
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


def add_trigger(image, trigger_size=15, trigger_value=255):
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




# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
# net = Net().to(DEVICE)
# trainloader, testloader = load_data()

net = Net().to(DEVICE)
local_net = Net().to(DEVICE)
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

        

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)


  



    def fit(self, parameters, config):
        # global local_net
        # self.set_parameters(parameters)
        print(net.layers)
        

        self.parameters_list.clear()

        print(config.get("isMal"))
        print(config.keys().__len__())

        # config['isMal'] = False

        if config.get("isMal", True) or self.hasBeenFlagged:
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

            # # Combine all batches into a single tensor
            # targeted_images = torch.cat(targeted_images, dim=0)
            # targeted_images = targeted_images.to(DEVICE)

            # features_list = []

            # def hook_fn(module, input, output):
            #     features_list.append(input[0].detach())

            # # Register the hook to capture the input to fc2
            # handle = local_net.fc2.register_forward_hook(hook_fn)

            # # Pass the targeted images through the model
            # local_net.eval()
            # with torch.no_grad():
            #     # local_net = local_net.to(DEVICE)
            #     _ = local_net(targeted_images)

            # handle.remove()

            # # Convert the list of feature outputs to a tensor
            # features = torch.cat(features_list, dim=0)

            # flattened_features = features.view(features.size(0), -1).cpu().numpy()

            # # Apply PCA to reduce to 2 components
            # pca = PCA(n_components=2)
            # pca_result = pca.fit_transform(flattened_features)
            # pc1_values = pca_result[:, 0].reshape(-1, 1)

            # # Apply K-means clustering
            # kmeans = KMeans(n_clusters=2, random_state=42)
            # cluster_labels = kmeans.fit_predict(pc1_values)

            # # Identify the sizes of the clusters
            # unique, counts = np.unique(cluster_labels, return_counts=True)
            # cluster_sizes = dict(zip(unique, counts))

            # # Find the smallest cluster
            # smallest_cluster_label = min(cluster_sizes, key=cluster_sizes.get)

            # # Get indices of samples in the smallest cluster
            # smallest_cluster_indices = np.where(cluster_labels == smallest_cluster_label)[0]

            # # Convert the smallest cluster indices to their original dataset indices
            # # smallest_cluster_dataset_indices = [original_indices[i] for i in smallest_cluster_indices]
            # # indices_to_exclude = set(smallest_cluster_dataset_indices)


            # local_net.eval()
            # net.eval()

            # with torch.no_grad():
            #     # Get predictions from local_net
            #     local_outputs = local_net(targeted_images)
            #     local_confidences = F.softmax(local_outputs, dim=1)  # Apply softmax to get probabilities
            #     local_max_confidences, local_predicted_labels = torch.max(local_confidences, dim=1)

            #     # Get predictions from net
            #     global_outputs = net(targeted_images)
            #     global_confidences = F.softmax(global_outputs, dim=1)
            #     global_max_confidences, global_predicted_labels = torch.max(global_confidences, dim=1)

            # # Initialize indices_to_exclude with smallest cluster indices
            # smallest_cluster_dataset_indices = [original_indices[i] for i in smallest_cluster_indices]
            # indices_to_exclude = set(smallest_cluster_dataset_indices)

            # # Add samples based on confidence score comparison
            # confidence_threshold = 0.8  # Define a threshold for low confidence

            # n_excluded_through_cnf = 0
            # for idx in range(len(targeted_images)):
            #     # if local_max_confidences[idx] < confidence_threshold and local_predicted_labels[idx] != global_predicted_labels[idx]:
            #     #     indices_to_exclude.add(original_indices[idx])
            #     #     n_excluded_through_cnf+=1
            #     if local_predicted_labels[idx] != global_predicted_labels[idx]:
            #         indices_to_exclude.add(original_indices[idx])
            #         n_excluded_through_cnf+=1
            #     elif torch.abs(global_max_confidences[idx] - local_max_confidences[idx]) > 0.5 :
            #         indices_to_exclude.add(original_indices[idx])
            #         n_excluded_through_cnf+=1
            
            # print(f"exlcuded by cnf: {n_excluded_through_cnf}")

            # # Check how many of the smallest cluster indices are in triggered_indices
            # # Assuming triggered_indices are defined and correspond to actual triggered samples
            # actual_triggered_indices = set(triggered_indices)  # Replace this with your actual triggered indices
            # #triggered_found_indices = set(smallest_cluster_dataset_indices) & actual_triggered_indices
            # triggered_found_indices = indices_to_exclude & actual_triggered_indices
            # num_actual_triggered_found = len(triggered_found_indices)

            # print(f"Number of actual triggered samples found in the smallest cluster: {num_actual_triggered_found} out of {len(actual_triggered_indices)}")
            # print(f"Smallest Cluster Size: {len(smallest_cluster_indices)}")
            
            # # Plot the PCA results with clustering
            # plt.figure(figsize=(8, 6))
            # scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
            # plt.colorbar(scatter)
            # plt.title('PCA of CNN Features for Targeted Label with K-means Clustering')
            # plt.xlabel('Principal Component 1')
            # plt.ylabel('Principal Component 2')
            # plt.savefig(f'3_{config.get("rnd")}_clustered.png')
            # plt.close()

            
            # del targeted_images
            # gc.collect()
            # torch.cuda.empty_cache()

            # Define batch size
            batch_size = 32

            # Combine all batches into a single tensor
            targeted_images = torch.cat(targeted_images, dim=0)
            targeted_images = targeted_images.to(DEVICE)

            features_list = []

            def hook_fn(module, input, output):
                features_list.append(input[0].detach())

            # Register the hook to capture the input to fc2
            handle = local_net.fc2.register_forward_hook(hook_fn)

            # Pass the targeted images through the model in batches
            local_net.eval()
            with torch.no_grad():
                for i in range(0, len(targeted_images), batch_size):
                    batch = targeted_images[i:i + batch_size]
                    _ = local_net(batch)

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
            smallest_cluster_dataset_indices = [original_indices[i] for i in smallest_cluster_indices]
            indices_to_exclude = set(smallest_cluster_dataset_indices)

            local_net.eval()
            net.eval()

            with torch.no_grad():
                local_max_confidences_list = []
                local_predicted_labels_list = []
                global_max_confidences_list = []
                global_predicted_labels_list = []

                for i in range(0, len(targeted_images), batch_size):
                    batch = targeted_images[i:i + batch_size]

                    # Get predictions from local_net
                    local_outputs = local_net(batch)
                    local_confidences = F.softmax(local_outputs, dim=1)  # Apply softmax to get probabilities
                    local_max_confidences, local_predicted_labels = torch.max(local_confidences, dim=1)
                    local_max_confidences_list.append(local_max_confidences)
                    local_predicted_labels_list.append(local_predicted_labels)

                    # Get predictions from net
                    global_outputs = net(batch)
                    global_confidences = F.softmax(global_outputs, dim=1)
                    global_max_confidences, global_predicted_labels = torch.max(global_confidences, dim=1)
                    global_max_confidences_list.append(global_max_confidences)
                    global_predicted_labels_list.append(global_predicted_labels)

                local_max_confidences = torch.cat(local_max_confidences_list, dim=0)
                local_predicted_labels = torch.cat(local_predicted_labels_list, dim=0)
                global_max_confidences = torch.cat(global_max_confidences_list, dim=0)
                global_predicted_labels = torch.cat(global_predicted_labels_list, dim=0)

            # Add samples based on confidence score comparison
            confidence_threshold = 0.8  # Define a threshold for low confidence

            n_excluded_through_cnf = 0
            for idx in range(len(targeted_images)):
                if local_predicted_labels[idx] != global_predicted_labels[idx]:
                    indices_to_exclude.add(original_indices[idx])
                    n_excluded_through_cnf += 1
                elif torch.abs(global_max_confidences[idx] - local_max_confidences[idx]) > 0.5:
                    indices_to_exclude.add(original_indices[idx])
                    n_excluded_through_cnf += 1

            print(f"excluded by cnf: {n_excluded_through_cnf}")

            # Check how many of the smallest cluster indices are in triggered_indices
            actual_triggered_indices = set(triggered_indices)  # Replace this with your actual triggered indices
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

            del targeted_images
            gc.collect()
            torch.cuda.empty_cache()

            def create_filtered_trainloader():
                global trainloader
                train_dataset = trainloader.dataset
                keep_indices = [i for i in range(len(train_dataset)) if i not in indices_to_exclude]
                filtered_train_dataset = torch.utils.data.Subset(train_dataset, keep_indices)
                triggered_train_dataset = torch.utils.data.Subset(train_dataset, list(indices_to_exclude))

                trainloader_filtered = torch.utils.data.DataLoader(filtered_train_dataset, batch_size=trainloader.batch_size, shuffle = False)
                trainloader_triggered = torch.utils.data.DataLoader(triggered_train_dataset, batch_size = trainloader.batch_size, shuffle = False)
                return trainloader_filtered, trainloader_triggered

            #update_trainloader()
            trainloader_filtered, trainloader_triggered = create_filtered_trainloader()
            self.set_parameters(parameters)


            train(net, trainloader_filtered, epochs=1)
            train(local_net, trainloader, epochs=1)
            return self.get_parameters(config={}), len(trainloader_filtered.dataset), {}
            #trainloader = trainloader_filtered

        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        if self.hasBeenFlagged == False :
            # parameters = [param.data.detach().cpu().numpy() for param in net.parameters()]
            # params_dict = zip(local_net.state_dict().keys(), parameters)
            # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            # local_net.load_state_dict(state_dict, strict=True)

            #save the parameters of net into local_net without using any cpu
            local_net.load_state_dict(net.state_dict())
            # local_net = local_net.to(DEVICE)

        return self.get_parameters(config={}), len(trainloader.dataset), {}





    def evaluate(self, parameters, config, ):
        # Update model parameters
        self.set_parameters(parameters)

        # Perform evaluation on the test dataset
        loss, accuracy = test(net, testloader)  # Ensure self.net and self.testloader are defined

        # Access custom metrics
        metrics = super().evaluate(parameters, config)
        triggered_accuracy, clean_accuracy = evaluate_trigger(testloader, triggered_indices_test)
        print(f'Accuracy on triggered images: {triggered_accuracy * 100:.2f}%')
        print(f'Accuracy on clean images: {clean_accuracy * 100:.2f}%')

        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
print("ok")



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

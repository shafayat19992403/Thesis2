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
import sys


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
parser.add_argument("--trigger_label", type=int, default=5, help="Label to be used for the trigger")
parser.add_argument("--cid", type=int, default=0, help="Client ID")


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
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

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


def load_data_with_trigger(data_path, trigger_fraction=0.2, trigger_label=5):
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

    return DataLoader(triggered_trainset, batch_size=32, shuffle=True), DataLoader(triggered_testset, shuffle=False), triggered_indices_test




# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
# net = Net().to(DEVICE)
# trainloader, testloader = load_data()

net = Net().to(DEVICE)
trainloader, testloader, triggered_indices_test  = load_data_with_trigger(args.data_path, args.trigger_frac, args.trigger_label)

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
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, threshold_loss=500, threshold_accuracy=0.02,perturb_rate=0.0):
        super().__init__()
        self.previous_loss = None
        self.previous_accuracy = None
        self.threshold_loss = threshold_loss
        self.threshold_accuracy = threshold_accuracy
        self.perturb_rate = perturb_rate
        self.client_flags = {}
        self.global_parameters = None
        self.local_parameters = None
        self.trigger_label = None

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
    



    def evaluate_globalvslocal(self, rnd):
    # Ensure global and local parameters are set
        if self.global_parameters is None or self.local_parameters is None:
            raise ValueError("Global and local parameters must be set before evaluation.")

        # Set global parameters and evaluate
        self.set_parameters(self.global_parameters)
        global_loss, global_accuracy = test(net, testloader)
        global_predictions, global_labels = self.get_predictions_and_labels(testloader)

        # Set local parameters and evaluate
        self.set_parameters(self.local_parameters)
        local_loss, local_accuracy = test(net, testloader)
        local_predictions, local_labels = self.get_predictions_and_labels(testloader)

        # Calculate False Positive Rate (FPR) for each label
        global_fpr = []
        local_fpr = []
        for label in range(10):  # Assuming 10 classes for MNIST/FashionMNIST
            global_fp = ((global_predictions == label) & (global_labels != label)).sum()
            global_tn = ((global_predictions != label) & (global_labels != label)).sum()
            global_fpr.append(global_fp / (global_fp + global_tn) if (global_fp + global_tn) > 0 else 0)

            local_fp = ((local_predictions == label) & (local_labels != label)).sum()
            local_tn = ((local_predictions != label) & (local_labels != label)).sum()
            local_fpr.append(local_fp / (local_fp + local_tn) if (local_fp + local_tn) > 0 else 0)

        # Calculate ROC AUC score per class (one-vs-rest)
        global_roc_auc_per_label = []
        local_roc_auc_per_label = []
        for label in range(10):
            global_roc_auc_label = roc_auc_score((global_labels == label).astype(int), (global_predictions == label).astype(int))
            local_roc_auc_label = roc_auc_score((local_labels == label).astype(int), (local_predictions == label).astype(int))
            global_roc_auc_per_label.append(global_roc_auc_label)
            local_roc_auc_per_label.append(local_roc_auc_label)

        weighted_label_detection = False
        # Save the labels whose local FPR difference is the max compared to global FPR
        if(weighted_label_detection == False):
            max_fpr_diff_label = np.argmax(np.array(local_fpr) - np.array(global_fpr))
        # max_roc_auc_diff_label = np.argmax(np.array(local_roc_auc_per_label) - np.array(global_roc_auc_per_label))
        else:
            fpr_diff = np.array(local_fpr) - np.array(global_fpr)
            roc_auc_diff = np.array(local_roc_auc_per_label) - np.array(global_roc_auc_per_label)
            #scale the difference to 0-1
            fpr_diff = (fpr_diff - np.min(fpr_diff)) / (np.max(fpr_diff) - np.min(fpr_diff))
            roc_auc_diff = (roc_auc_diff - np.min(roc_auc_diff)) / (np.max(roc_auc_diff) - np.min(roc_auc_diff))

            weighted_diff = 0.5 * fpr_diff + 0.5 * roc_auc_diff

            max_fpr_diff_label = np.argmax(weighted_diff)
        self.trigger_label = max_fpr_diff_label 

        # Print global vs local ROC AUC per label
        for label in range(10):
            print(f"Label {label} ---> Global ROC AUC: {global_roc_auc_per_label[label]:.4f}, Local ROC AUC: {local_roc_auc_per_label[label]:.4f}")

        # Plot False Positive Rate (FPR) for each label
        labels = list(range(10))  # Assuming 10 classes for MNIST/FashionMNIST

        plt.figure(figsize=(12, 6))

        # FPR plot
        plt.subplot(1, 2, 1)
        plt.plot(labels, global_fpr, label='Global FPR', marker='o', color='blue')
        plt.plot(labels, local_fpr, label='Local FPR', marker='o', color='green')

        # Highlight points where local FPR is greater than global FPR
        for i, label in enumerate(labels):
            if local_fpr[i] > global_fpr[i]:
                plt.scatter(label, local_fpr[i], color='red', zorder=5, s=100, edgecolor='black', label="Local > Global" if i == 0 else "")

        plt.xlabel('Labels')
        plt.ylabel('False Positive Rate')
        plt.title('FPR Comparison (Global vs Local)')
        plt.legend()

        # ROC AUC plot
        plt.subplot(1, 2, 2)
        plt.plot(labels, global_roc_auc_per_label, label='Global ROC AUC', marker='o', color='blue')
        plt.plot(labels, local_roc_auc_per_label, label='Local ROC AUC', marker='o', color='green')

        # Highlight points where local ROC AUC is greater than global ROC AUC
        for i, label in enumerate(labels):
            if local_roc_auc_per_label[i] > global_roc_auc_per_label[i]:
                plt.scatter(label, local_roc_auc_per_label[i], color='red', zorder=5, s=100, edgecolor='black', label="Local > Global" if i == 0 else "")

        plt.xlabel('Labels')
        plt.ylabel('ROC AUC')
        plt.title('ROC AUC Comparison (Global vs Local)')
        plt.legend()

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"Figures/ClientFPR/C{args.cid}_global_vs_local_fpr_roc_auc_{rnd}.png")
        plt.close()


    def get_predictions_and_labels(self, dataloader):
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return np.array(all_predictions), np.array(all_labels)


    def fit(self, parameters, config):
        print("---------------------------------------->config:")
        print(config.get("isMal"))
        print(config.keys().__len__())

        self.global_parameters = [ p.copy() for p in parameters ]


        
    
        self.set_parameters(parameters)
        print(net.layers)
        

        self.get_parameters('')
    
        

        train(net, trainloader, epochs=1)
        self.local_parameters = self.get_parameters(config={})
        print("-------------->config:")
        print(config.get("isMal"))
        print(config.keys().__len__())
        
        if config is not None and config.get("isMal") == True and self.local_parameters is not None and self.global_parameters is not None:
            self.evaluate_globalvslocal(config.get("rnd"))


        with open(f'Figures/ConfigTexts/C{args.cid}_logs.txt', 'a') as f:
            # Change the output stream to the file
            sys.stdout = f

            # Write a line to the file
            # print("This is a line written to the file")

            if config.get("isMal") == True and self.local_parameters is not None and self.global_parameters is not None and config is not None:
                print(f"S0---------->Successfully detected trigger client at round {config.get('rnd')}")
                if(self.trigger_label is not None and self.trigger_label != args.trigger_label):
                    print(f"E1----------->Failed to detect correct trigger label in round {config.get('rnd')}")
                else:
                    print(f"S1----------->Successfully detected trigger label in round {config.get('rnd')}")

            if config.get("isMal") == False and args.trigger_frac > 0:
                print(f"E2----------->Failed to detect trigger client in round {config.get('rnd')}")

            if config.get("isMal") == False and args.trigger_frac == 0:
                print(f"S2----------->Successfully detected non-trigger client in round {config.get('rnd')}")
            
            if config.get("isMal") == True and args.trigger_frac == 0:
                print(f"E3----------->Flagged a non-trigger client in round {config.get('rnd')}")

            
            print("-------------->config:")
            print(config.get("isMal"))
            print(config.keys().__len__())
            # Change the output stream back to default

            sys.stdout = sys.__stdout__


        
        print("-------------->config:")
        print(config.get("isMal"))
        print(config.keys().__len__())

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

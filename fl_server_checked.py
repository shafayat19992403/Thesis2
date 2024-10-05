import sys
import os
import time
import numpy as np
from typing import List, Tuple
# import flwr as fl
from flwr.common import Metrics
from flwr.common import parameters_to_ndarrays
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import random
from sklearn.utils import resample

parent_dir = os.path.dirname(os.path.realpath(__file__))
print(parent_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import flwr as fl

# Argument parser for number of rounds
parser = argparse.ArgumentParser(description="Federated Learning with Flower and PyTorch")
parser.add_argument("--number_of_round", type=int, default=8, help="Number of rounds")
args = parser.parse_args()

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define the function to extract client model weights and flatten them
def extract_client_weights(client_models):
    client_weights = []
    for client_model in client_models:  # list of `Parameters` objects
        weights = parameters_to_ndarrays(client_model)  # Convert Parameters to ndarray
        flat_weights = np.concatenate([w.flatten() for w in weights])  # Flatten the weights
        client_weights.append(flat_weights)
    return client_weights



def apply_pca_to_weights(client_weights, client_ids):
    # Apply PCA to reduce to 2 dimensions
    print(len(client_weights))
    print(len(client_weights[0]))
    #client_weights_2d = np.vstack(client_weights)
    client_weights_2d = client_weights
    pca = PCA(n_components=2)
    reduced_weights = pca.fit_transform(client_weights_2d)
    
    # Extract PC1 values
    pc1_values = reduced_weights[:, 0]
    
    # Reshape PC1 values for clustering
    pc1_values = pc1_values.reshape(-1, 1)
    
    # Apply DBSCAN clustering based on PC1 values
    dbscan = DBSCAN(eps=0.5, min_samples=2)  # Adjust eps based on your data
    cluster_labels = dbscan.fit_predict(pc1_values)
    #print(cluster_labels)
    
    # Cluster label -1 denotes outliers in DBSCAN
    # outliers = [client_ids[i] for i, label in enumerate(cluster_labels) if label == -1]
    label_counts = Counter(cluster_labels)
    print("Cluster label counts:", label_counts)

    # Define a threshold for considering a cluster as an outlier (e.g., smallest size)
    # You can adjust the threshold as needed, here I mark the smallest clusters as outliers
    smallest_cluster_size = min(label_counts.values())  # Get the size of the smallest cluster
    outlier_labels = [label for label, count in label_counts.items() if count == smallest_cluster_size]

    # Identify clients belonging to the smallest clusters (potential outliers)
    outliers = [client_ids[i] for i, label in enumerate(cluster_labels) if label in outlier_labels]
    
    # Plot the results
    plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], c=cluster_labels, cmap='viridis', label='Clients')
    plt.colorbar(label='Cluster Label')
    
    # Annotate clients and highlight outliers
    for i in range(len(client_weights)):
        #plt.annotate(f"Client {client_ids[i]}", (reduced_weights[i, 0], reduced_weights[i, 1]))
        plt.annotate(f"Client {i}", (reduced_weights[i,0], (reduced_weights[i,1])))
    
    if outliers:
        outlier_indices = [client_ids.index(client_id) for client_id in outliers]
        plt.scatter(reduced_weights[outlier_indices, 0], reduced_weights[outlier_indices, 1], color='red', label='Outliers')
    
    plt.title("PCA of Client Weights with DBSCAN-based Outliers (Based on PC1)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig("pca_with_dbscan_based_on_pc1.png")
    plt.close()


    pc1_contributions = np.abs(pca.components_[0])  # Absolute values of the loadings for PC1
    # most_important_weights = np.argsort(pc1_contributions)[::-1]  # Sort by importance (descending)
    sorted_contributions = np.sort(pc1_contributions)[::-1]
    cumulative_contribution = np.cumsum(sorted_contributions)
    cumulative_percentage = cumulative_contribution / np.sum(pc1_contributions)
    most_important_weights_int = []
    
    threshold = 0.9
    num_weights_90_percent = np.argmax(cumulative_percentage >= threshold) + 1
    print(f"Number of weights needed to reach {threshold * 100}% contribution: {num_weights_90_percent}")
    
    most_important_weights = np.argsort(pc1_contributions)[::-1][:num_weights_90_percent]

    # Print or return the top contributing weights
    top_n = num_weights_90_percent  # You can adjust this number to see the top N weights
    print(f"Top {top_n} weights contributing to PC1:")
    for i in range(top_n):
        #print(f"Weight {most_important_weights[i]} contributes {pc1_contributions[most_important_weights[i]]}")
        most_important_weights_int.append(int(most_important_weights[i]))


    return outliers, most_important_weights_int\
    

# # Define the Net class as provided
# class TestModel(nn.Module):
#     def __init__(self):
#         super(TestModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(1600, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


# def evaluate_malicious_client_weight_for_labels(data_name, test_sample_size, model):
#     """Evaluate model accuracy metrics for each label on a stratified sample of the dataset."""
#     # Load the appropriate dataset
#     if data_name == 'MNIST':
#         dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
#     elif data_name == 'FMNIST':
#         dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
#     else:
#         raise ValueError(f"Unsupported dataset: {data_name}")

#     # Create a DataLoader for the test dataset
#     test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

#     # Determine the total number of samples and calculate the sample size
#     total_samples = len(dataset)
#     sample_size = int(total_samples * test_sample_size)

#     # Stratified sampling to keep the distribution of labels same
#     label_counts = defaultdict(int)  # Assuming label_counts is defined as a defaultdict

#     for _, label in dataset:
#         label_counts[label] += 1

#     # Calculate the number of samples per label for stratified sampling
#     stratified_sample = {}
#     for label, count in label_counts.items():
#         stratified_sample[label] = int(sample_size * (count / total_samples))

#     sampled_data = []
#     for label, count in stratified_sample.items():
#         # Get all indices of the current label
#         indices = [i for i, (_, l) in enumerate(dataset) if l == label]
#         # Randomly sample the specified number of indices for the current label
#         sampled_indices = random.sample(indices, count)
#         sampled_data.extend([dataset[i] for i in sampled_indices])

#     # Separate images and labels
#     test_images, test_labels = zip(*sampled_data)

#     # Convert to tensors
#     test_images = torch.stack(test_images)
#     test_labels = torch.tensor(test_labels)

    
#     # model.load_state_dict(torch.load('model.pth'))
#     # Evaluate the model
#     model.eval()
#     with torch.no_grad():
#         predictions = model(test_images)

#     predicted_labels = torch.argmax(predictions, dim=1)

#     # Calculate metrics for each label
#     metrics_per_label = defaultdict(list)
#     for true, pred in zip(test_labels.numpy(), predicted_labels.numpy()):
#         metrics_per_label[true].append(pred)

#     # Initialize a dictionary to hold the metrics
#     metrics_summary = {}

#     # Calculate accuracy metrics for each label
#     for label, results in metrics_per_label.items():
#         accuracy = accuracy_score([label] * len(results), results)
#         precision = precision_score([label] * len(results), results, average='macro', zero_division=0)
#         recall = recall_score([label] * len(results), results, average='macro', zero_division=0)
#         f1 = f1_score([label] * len(results), results, average='macro', zero_division=0)

#         metrics_summary[label] = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1
#         }
#         print(f"Label {label}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, "
#               f"Recall = {recall:.4f}, F1 Score = {f1:.4f}")

#     # Determine the label with the poorest prediction (least accuracy)
#     poorest_label = min(metrics_summary, key=lambda k: metrics_summary[k]['accuracy'])
#     print(f"Label with the poorest prediction: {poorest_label} "
#           f"with accuracy = {metrics_summary[poorest_label]['accuracy']:.4f}")
   







class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.malicious_clients = []
        self.client_flags = {}
        # self.model = TestModel()




    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        # Extract and flatten weights of each client after training round
        print("Extracting and flattening client weights...")
        client_models = [result.parameters for cid, result in results]  # Get client models
        client_ids = [cid for cid, result in results]  # Get client IDs
        client_weights = extract_client_weights(client_models)
        print("Client weights extracted and flattened.")
        
        # Apply PCA to the extracted weights and pass client IDs
        self.malicious_clients, most_important_weights = apply_pca_to_weights(client_weights, client_ids)
        print("Done PCA.....")
        self.client_flags = {client_id: client_id in self.malicious_clients for client_id in client_ids}
        
        # print(type(most_important_weights))
        # print(type(most_important_weights[0]))
        # print(most_important_weights)
        
        # Print malicious clients
        if self.malicious_clients:
            print(f"Malicious clients detected in round {rnd}: {self.malicious_clients}")
        else:
            print(f"No malicious clients detected in round {rnd}.")

        for client_id in client_ids:
            config = {"malicious": client_id in self.malicious_clients}
            # self.send_notification(client_id, config, aggregated_weights)
            print("Notification sent")
        # for client_id, client_model in zip(client_ids, client_models):
        #     if client_id in self.malicious_clients:
        #         param_ndarrays = parameters_to_ndarrays(client_model)  # This will be a list of ndarrays
        
        #         # Create a state_dict that matches the model's parameters
        #         state_dict = {
        #             "conv1.weight": torch.tensor(param_ndarrays[0]),
        #             "conv1.bias": torch.tensor(param_ndarrays[1]),
        #             "conv2.weight": torch.tensor(param_ndarrays[2]),
        #             "conv2.bias": torch.tensor(param_ndarrays[3]),
        #             "fc1.weight": torch.tensor(param_ndarrays[4]),
        #             "fc1.bias": torch.tensor(param_ndarrays[5]),
        #             "fc2.weight": torch.tensor(param_ndarrays[6]),
        #             "fc2.bias": torch.tensor(param_ndarrays[7]),
        #         }
                
        #         # Load the parameters into the model
        #         self.model.load_state_dict(state_dict)
            
                

                # torch.save(client_model.state_dict(), 'model.pth')
                # evaluate_malicious_client_weight_for_labels('FMNIST', 0.2,self.model)
              
        return aggregated_weights, self.malicious_clients, most_important_weights
        # return aggregated_weights


    def send_notification(self, client_id, config, aggregated_weights):
        # This function will add a notification flag to the client
        client_proxy = fl.server.SimpleClientManager.clients[client_id]
        # Send the `config` with the flag indicating malicious client status
        client_proxy.fit(parameters=aggregated_weights, config=config)

    

# Use the custom strategy
strategy = CustomFedAvg(evaluate_metrics_aggregation_fn=weighted_average)


    

# Start Flower server
start_time = time.time()
print("Flower server started at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

fl.server.start_server(
    server_address="0.0.0.0:3000",
    config=fl.server.ServerConfig(num_rounds=args.number_of_round),
    strategy=strategy,
)

# Calculate and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print("Flower server has run for: {:.2f} seconds".format(elapsed_time))

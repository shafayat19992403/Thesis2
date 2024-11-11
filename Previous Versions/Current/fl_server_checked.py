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
from flwr.server.strategy.fedavg import aggregate

from flwr.common import (FitIns,FitRes,Parameters, ndarrays_to_parameters,

    parameters_to_ndarrays)



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
parser.add_argument("--number_of_round", type=int, default=4, help="Number of rounds")
parser.add_argument("--mal_clnt_weight", type=float, default=1, help="Weight of malicious clients")   
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



    

def apply_pca_to_weights(client_weights, client_ids,rnd):
    # Apply PCA to reduce to 2 dimensions
    print(len(client_weights))
    print(len(client_weights[0]))

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
    
    label_counts = Counter(cluster_labels)
    print("Cluster label counts:", label_counts)

    if len(label_counts) > 1:
    # Find the smallest clusters (potential outliers)
        smallest_cluster_size = min(label_counts.values())
        outlier_labels = [label for label, count in label_counts.items() if count == smallest_cluster_size]
        outliers = [client_ids[i] for i, label in enumerate(cluster_labels) if label in outlier_labels]
    else:
        outlier_labels = []
        outliers = []
    
    # Identify clients belonging to the smallest clusters (potential outliers)
    
    
    # Plot the PCA results
    plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], c=cluster_labels, cmap='viridis', label='Clients')
    plt.colorbar(label='Cluster Label')
    
    # Annotate clients and highlight outliers
    for i in range(len(client_weights)):
        plt.annotate(f"Client {i}", (reduced_weights[i,0], reduced_weights[i,1]))
    
    if outliers:
        outlier_indices = [client_ids.index(client_id) for client_id in outliers]
        plt.scatter(reduced_weights[outlier_indices, 0], reduced_weights[outlier_indices, 1], color='red', label='Outliers')
    
    plt.title("PCA of Client Weights with DBSCAN-based Outliers (Based on PC1)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig(f"Figures/ServerPCA/pca_with_dbscan_based_on_pc1__{rnd}.png")
    plt.close()

    print(f"----->Number of outliers detected: {len(outliers)}")

    # PCA contribution analysis
    pc1_contributions = np.abs(pca.components_[0])  # Absolute values of the loadings for PC1
    sorted_contributions = np.sort(pc1_contributions)[::-1]
    cumulative_contribution = np.cumsum(sorted_contributions)
    cumulative_percentage = cumulative_contribution / np.sum(pc1_contributions)
    
    threshold = 0.2
    num_weights_90_percent = np.argmax(cumulative_percentage >= threshold) + 1
    print(f"Number of weights needed to reach {threshold * 100}% contribution: {num_weights_90_percent}")
    
    most_important_weights = np.argsort(pc1_contributions)[::-1][:num_weights_90_percent]
    most_important_weights_int = [int(weight) for weight in most_important_weights]

    # Identify the clean clients (non-outliers)
    clean_clients = [client_ids[i] for i in range(len(client_ids)) if client_ids[i] not in outliers]

    # Find the weights with the highest difference between clean and malicious clients
    if len(label_counts) > 1:
        malicious_weights = np.mean([client_weights[i] for i in outlier_indices], axis=0)
        clean_weights = np.mean([client_weights[i] for i in range(len(client_ids)) if i not in outlier_indices], axis=0)

        weight_differences = clean_weights - malicious_weights

        # Find the index of the largest positive difference
        largest_positive_diff_index = np.argmax(weight_differences)
        largest_positive_diff_value = weight_differences[largest_positive_diff_index]

        # Find the index of the largest negative difference
        largest_negative_diff_index = np.argmin(weight_differences)
        largest_negative_diff_value = weight_differences[largest_negative_diff_index]

        # Output the results
        print(f"Largest positive difference at index {largest_positive_diff_index}: {largest_positive_diff_value}")
        print(f"Largest negative difference at index {largest_negative_diff_index}: {largest_negative_diff_value}")


    # print(f"Largest difference in weights between clean and malicious clients at index {largest_diff_index}: {largest_diff_value}")

    return outliers, most_important_weights_int




class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.malicious_clients = []
        self.client_flags = {}
        # self.model = TestModel()






    def aggregate_fit(self, rnd, results, failures):
        # Extract and flatten weights of each client after training round
        print("Extracting and flattening client weights...")
        client_models = [result.parameters for cid, result in results]  # Get client models
        client_ids = [cid for cid, result in results]  # Get client IDs
        client_weights = extract_client_weights(client_models)
        print("Client weights extracted and flattened.")

        # Apply PCA to the extracted weights and pass client IDs
        self.malicious_clients, most_important_weights = apply_pca_to_weights(client_weights, client_ids,rnd)
        print("Done PCA.....")
        self.client_flags = {client_id: client_id in self.malicious_clients for client_id in client_ids}

        # Print malicious clients
        if self.malicious_clients:
            print(f"Malicious clients detected in round {rnd}: {self.malicious_clients}")
        else:
            print(f"No malicious clients detected in round {rnd}.")

        # Adjust weights of malicious clients by reducing them by 0.5
        # malicious_client_weight_reduction = args.mal_clnt_weight
        malicious_client_weight_reduction = 0.5
        adjusted_results = []
        for idx, (client_id, result) in enumerate(results):
            if client_id in self.malicious_clients:
                adjusted_weights = []
                temp = parameters_to_ndarrays(result.parameters)  # Get the weights as ndarrays
                        
                for w in temp:  # Iterate over each weight
                    try:
                                # Ensure weights are float32
                        numeric_w = np.array(w, dtype=np.float32)  
                                # Scale the weight and add to adjusted weights
                        adjusted_weights.append(numeric_w * malicious_client_weight_reduction)  
                    except Exception as e:
                        print(f"Warning: Could not scale weights for client {client_id} due to incompatible type: {e}")
                        adjusted_weights.append(w)  # Retain original weight if conversion fails

                        # Create a new Parameters object with adjusted weights
                adjusted_result = FitRes(
                            parameters=ndarrays_to_parameters(adjusted_weights),  # Use adjusted weights here
                            num_examples=result.num_examples,
                            metrics=result.metrics,
                            status=result.status  # Ensure to add the `status` argument
                        )
                print("adjusted............ ")
                adjusted_results.append((client_id, adjusted_result))
            else:
                 adjusted_results.append((client_id, result))

                # Use the super() method to aggregate the adjusted results
        aggregated_weights = super().aggregate_fit(rnd, adjusted_results, failures)

        return aggregated_weights, self.malicious_clients, most_important_weights





    

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

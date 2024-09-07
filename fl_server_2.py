import sys
import os
import time
import numpy as np
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
from flwr.common import parameters_to_ndarrays
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Argument parser for number of rounds
parser = argparse.ArgumentParser(description="Federated Learning with Flower and PyTorch")
parser.add_argument("--number_of_round", type=int, default=4, help="Number of rounds")
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

# Function to apply PCA on model weights
def apply_pca_to_weights(client_weights):
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    reduced_weights = pca.fit_transform(client_weights)
    
    # Plotting the PCA results
    plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1])
    for i in range(len(client_weights)):
        plt.annotate(f"Client {i+1}", (reduced_weights[i, 0], reduced_weights[i, 1]))
    plt.title("PCA of Client Model Weights")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    #plt.show()
    plt.savefig("pca_weights.png")
    plt.close()

# Define a custom Flower strategy that collects model weights after training rounds
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        # Extract and flatten weights of each client after training round
        print("Extracting and flattening client weights...")
        client_models = [result.parameters for cid, result in results]  # get client models
        client_weights = extract_client_weights(client_models)
        print("Client weights extracted and flattened.")
        
        # Apply PCA to the extracted weights
        apply_pca_to_weights(client_weights)
        
        return aggregated_weights

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

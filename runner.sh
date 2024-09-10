#!/bin/bash

# Check if the number of clients is passed as an argument
if [ -z "$1 $2" ]; then
    echo "Usage: ./run_server_clients.sh <num_clients> <num_poisoned_clients>"
    exit 1
fi

# Number of clients to run
NUM_CLIENTS=$1
NUM_POISONED_CLIENTS=$2
SERVER_FILE="fl_server_2.py"
CLIENT_FILE="fl_client_2.py"

# Start the server in a new terminal
gnome-terminal -- bash -c "echo 'Starting server...'; python3 $SERVER_FILE ; exec bash"

# Start each client in a new terminal
for ((i=1; i<=NUM_CLIENTS; i++)); do
    gnome-terminal -- bash -c "echo 'Starting client $i...'; python3 $CLIENT_FILE --client_id $i; exec bash"
done

for ((i=1; i<=NUM_POISONED_CLIENTS; i++)); do
    gnome-terminal -- bash -c "echo 'Starting poisoned client $i...'; python3 $CLIENT_FILE --trigger_frac 0.2 --client_id $i; exec bash"
done



# Optional: Wait for all clients to finish
wait

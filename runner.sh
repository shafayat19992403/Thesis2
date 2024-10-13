#!/bin/bash
rm Figures/ServerPCA/*.png
rm Figures/ClientFPR/*.png
rm Figures/ConfigTexts/*.txt
rm Figures/ConfigTexts/OutputTexts/*.txt
# Check if the number of clients is passed as an argument
if [ -z "$1 $2" ]; then
    echo "Usage: ./run_server_clients.sh <num_clients> <num_poisoned_clients>"
    exit 1
fi

# Number of clients to run
NUM_CLIENTS=$1
NUM_POISONED_CLIENTS=$2
TRIGGER_LABEL_1=$3
TRIGGER_LABEL_2=$4
# SERVER_FILE="fl_server_2.py"
# CLIENT_FILE="fl_client_2.py"
SERVER_FILE="fl_server_checked.py"
CLIENT_FILE="fl_client_checked_working.py"

# Start the server in a new terminal
gnome-terminal -- bash -c "echo 'Starting server...'; python3 $SERVER_FILE ; exec bash"

# Start each client in a new terminal
for ((i=1; i<=NUM_CLIENTS; i++)); do
    gnome-terminal -- bash -c "echo 'Starting client $i...'; python3 $CLIENT_FILE --trigger_frac 0  --cid $i; exec bash"
done

for ((i=1; i<=NUM_POISONED_CLIENTS; i++)); do
    if [ $((i % 2)) -eq 0 ]; then
        TRIGGER_LABEL=$TRIGGER_LABEL_2
    fi
    if [ $((i % 2)) -eq 1 ]; then
        TRIGGER_LABEL=$TRIGGER_LABEL_1
    fi
    CID=$((i+NUM_CLIENTS))
    gnome-terminal -- bash -c "echo 'Starting poisoned client $i...'; python3 $CLIENT_FILE --trigger_frac 0.2 --trigger_label $TRIGGER_LABEL --cid $CID; exec bash"
done



# Optional: Wait for all clients to finish
wait
# ./summarizer.sh
# close all gnome terminals after the run
# killall gnome-terminal

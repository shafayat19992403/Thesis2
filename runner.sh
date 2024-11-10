#!/bin/bash
mkdir -p Figures/ServerPCA
mkdir -p Figures/ClientFPR
mkdir -p Figures/ConfigTexts
mkdir -p Figures/ConfigTexts/OutputTexts
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
# TRIGGER_LABEL_1=$3
# TRIGGER_LABEL_2=$4
# TRIGGER_LABEL_3=$5
# TRIGGER_LABEL_4=$6
TRIGGER_FRAC="0.1"
NUM_OF_ROUNDS="4"
SAME_LABEL="0"


TRIGGER_LABEL_1="1"
TRIGGER_LABEL_2="2"
TRIGGER_LABEL_3="4"
TRIGGER_LABEL_4="7"
# TRIGGER_LABEL=$3
# SERVER_FILE="fl_server_2.py"
# CLIENT_FILE="fl_client_2.py"
SERVER_FILE="fl_server_checked.py"
CLIENT_FILE="fl_client_checked_working.py"

# Start the server in a new terminal
gnome-terminal -- bash -c "echo 'Starting server...'; python3 $SERVER_FILE --number_of_round $NUM_OF_ROUNDS; exec bash"

# Start each client in a new terminal
for ((i=1; i<=NUM_CLIENTS; i++)); do
    gnome-terminal -- bash -c "echo 'Starting client $i...'; python3 $CLIENT_FILE --trigger_frac 0  --cid $i; exec bash"
done

for ((i=1; i<=NUM_POISONED_CLIENTS; i++)); do

    if [ $SAME_LABEL -eq 1 ]; then
        if [ $((i % 4)) -eq 0 ]; then
            TRIGGER_LABEL=$TRIGGER_LABEL_1
        fi
        if [ $((i % 4)) -eq 1 ]; then
            TRIGGER_LABEL=$TRIGGER_LABEL_2
        fi
        if [ $((i % 4)) -eq 2 ]; then
            TRIGGER_LABEL=$TRIGGER_LABEL_3
        fi
        if [ $((i % 4)) -eq 3 ]; then
            TRIGGER_LABEL=$TRIGGER_LABEL_4
        fi
    fi
    if [ $SAME_LABEL -eq 0 ]; then
        TRIGGER_LABEL=$TRIGGER_LABEL_1
    fi

    CID=$((i+NUM_CLIENTS))

    echo "Trigger label: $TRIGGER_LABEL"
    gnome-terminal -- bash -c "echo 'Starting poisoned client $i...'; python3 $CLIENT_FILE --trigger_frac $TRIGGER_FRAC --trigger_label $TRIGGER_LABEL --cid $CID; exec bash"
done



# Optional: Wait for all clients to finish
wait
# ./summarizer.sh
# close all gnome terminals after the run
# killall gnome-terminal

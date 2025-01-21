#!/bin/bash

# Create necessary directories and clean up existing files
mkdir -p Figures/{ServerPCA,ClientFPR,ConfigTexts/OutputTexts,ClientPCA}
rm -f Figures/ServerPCA/*.png
rm -f Figures/ClientFPR/*.png
rm -f Figures/ConfigTexts/*.txt
rm -f Figures/ConfigTexts/OutputTexts/*.txt
rm -f Figures/ClientPCA/*.png

# Check if the number of clients is passed as an argument
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./run_server_clients.sh <num_clients> <num_poisoned_clients>"
    exit 1
fi

# Configuration variables
NUM_CLIENTS=$1
NUM_POISONED_CLIENTS=$2
TRIGGER_FRAC="0.1"
NUM_OF_ROUNDS="8"
SAME_LABEL="0"
TRIGGER_LABEL_1="5"
TRIGGER_LABEL_2="2"
TRIGGER_LABEL_3="4"
TRIGGER_LABEL_4="7"
SERVER_FILE="fl_server_checked.py"
CLIENT_FILE="fl_client_checked.py"
DEFSTAT="1"
TRUST_FACTOR="0.6"

# Echo configuration
echo "Number of clients: $NUM_CLIENTS"
echo "Number of poisoned clients: $NUM_POISONED_CLIENTS"
echo "Trigger fraction: $TRIGGER_FRAC"
echo "Number of rounds: $NUM_OF_ROUNDS"
echo "Same label: $SAME_LABEL"
echo "Trigger label 1: $TRIGGER_LABEL_1"
echo "Trigger label 2: $TRIGGER_LABEL_2"
echo "Trigger label 3: $TRIGGER_LABEL_3"
echo "Trigger label 4: $TRIGGER_LABEL_4"
echo "Server file: $SERVER_FILE"
echo "Client file: $CLIENT_FILE"
echo "Defense status: $DEFSTAT"
echo "Trust factor: $TRUST_FACTOR"

# Create a new tmux session named 'fl' (Federated Learning)
tmux new-session -d -s fl

# Create window for server
tmux rename-window -t fl:0 'server'
tmux send-keys -t fl:0 "python3 $SERVER_FILE --number_of_round $NUM_OF_ROUNDS --withDefense $DEFSTAT --trust_factor $TRUST_FACTOR; mpg123 endsong.mp3" C-m

# Create windows for regular clients
for ((i=1; i<=NUM_CLIENTS; i++)); do
    tmux new-window -t fl:$i -n "client$i"
    tmux send-keys -t fl:$i "python3 $CLIENT_FILE --trigger_frac 0 --cid $i --withDefense $DEFSTAT" C-m
done

# Create windows for poisoned clients
for ((i=1; i<=NUM_POISONED_CLIENTS; i++)); do
    WINDOW_NUM=$((i+NUM_CLIENTS))
    CID=$((i+NUM_CLIENTS))
    
    if [ $SAME_LABEL -eq 1 ]; then
        case $((i % 4)) in
            0) TRIGGER_LABEL=$TRIGGER_LABEL_1 ;;
            1) TRIGGER_LABEL=$TRIGGER_LABEL_2 ;;
            2) TRIGGER_LABEL=$TRIGGER_LABEL_3 ;;
            3) TRIGGER_LABEL=$TRIGGER_LABEL_4 ;;
        esac
    else
        TRIGGER_LABEL=$TRIGGER_LABEL_1
    fi
    
    echo "Trigger label: $TRIGGER_LABEL"
    tmux new-window -t fl:$WINDOW_NUM -n "poisoned$i"
    tmux send-keys -t fl:$WINDOW_NUM "python3 $CLIENT_FILE --trigger_frac $TRIGGER_FRAC --cid $CID --trigger_label $TRIGGER_LABEL --withDefense $DEFSTAT" C-m
done

# Attach to the tmux session
tmux attach-session -t fl
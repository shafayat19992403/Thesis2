# Defense Against the Backdoor Attacks in Federated Learning

This repository contains codes and scripts belonging to the CSE6801 (Distributed Computing Systems) project. In this project, we attempted to detect and mitigate the effect of backdoor attacks in federated machine learning. We used [**Flower**](https://github.com/adap/flower) framework to implement the federated learning system. Therefore, we can simulate a federated learning with a server and multiple clients locally. Besides, we can actually train and evaluate a model with federated learning setting, involving multiple distributed machines, by providing the network address of the server machine (_server address can be provided as a command line argument_). In this project, we used [**CIFAR-10 dataset**](https://www.cs.toronto.edu/~kriz/cifar.html) as well as [**MNIST dataset**](https://yann.lecun.com/exdb/mnist/) to train and evaluate a model.


## Guidelines

- Create a _virtual environment_ or _Conda environment_ with Python version `3.10`.
- `pip install` the Python packages and libraries, including `flwr`, `torch`, and `torchvision`.
- use `pip install -r requirements.txt` to install all the python dependency




## Arguments for Server Script

### `--number_of_round`
- **Type:** `int`
- **Default:** `4`
- **Description:** Specifies the number of rounds for the federated learning process.

### `--trust_factor`
- **Type:** `float`
- **Default:** `0.5`
- **Description:** Sets the trust factor for identifying and handling malicious clients in the federated learning process.

### `--exp_no`
- **Type:** `int`
- **Default:** `1`
- **Description:** Defines the experiment number for tracking and organizing different experimental runs.

### `--withDefense`
- **Type:** `int`
- **Default:** `1`
- **Description:** Indicates whether to use a defense mechanism during the federated learning process. A value of `1` enables the defense mechanism, while `0` disables it.

  ```sh
  python fl_server.py -h                 # To see help messages
  python fl_server.py --address IP:PORT  # To run server script with server's network address
  python fl_server.py --round 10         # To run server script with 10 FL rounds
  ```

## Arguments for client scripts

### `--server_address`
- **Type:** `str`
- **Default:** `"0.0.0.0:3000"`
- **Description:** Specifies the address of the federated learning (FL) server.

### `--data_path`
- **Type:** `str`
- **Default:** `"./data"`
- **Description:** Defines the path to data.

### `--trigger_frac`
- **Type:** `float`
- **Default:** `0.2`
- **Description:** Sets the fraction of data to be poisoned.
- **Use trigger frac greater than 0 for backdoor-injected clients and 0 for benign client**

### `--trigger_label`
- **Type:** `int`
- **Default:** `5`
- **Description:** Specifies the label to be used for the trigger.

### `--cid`
- **Type:** `int`
- **Default:** `0`
- **Description:** Defines the client ID.

### `--withDefense`
- **Type:** `int`
- **Default:** `1`
- **Description:** Indicates whether to apply a defense mechanism. A value of `1` enables the defense mechanism, while `0` disables it.
  ```sh
  python fl_client.py --help             # To see help messages
  python fl_client.py --address IP:PORT  # To run client script with server's network address
  python fl_client.py                    # To run client script as a benign client and use CIFAR-10 dataset by default
  ```
# Documentation for `runner.sh`

## Overview
The `runner.sh` script is a utility for setting up and running a federated learning experiment. It initializes the required directories, clears previous outputs, and starts a specified number of client and server processes. It also supports the inclusion of poisoned clients for testing adversarial scenarios.

## Dependencies
To run this script, ensure the following dependencies are installed:

- **Bash**: The script is written for Bash shell.
- **Python 3**: Required to run the `fl_server.py` and `fl_client.py` scripts.
- **gnome-terminal**: Used to launch the server and client processes in separate terminals.

### Python Libraries
The following Python libraries may be required by `fl_server.py` and `fl_client.py`:

- `numpy`
- `tensorflow` or `pytorch` (depending on implementation)
- `matplotlib`
- Any other libraries specified in the Python scripts.

## How to Run
### Usage
```bash
./runner.sh <num_clients> <num_poisoned_clients>
```

### Arguments
- `<num_clients>`: Number of non-poisoned clients to run.
- `<num_poisoned_clients>`: Number of poisoned clients to include in the experiment.

### Example
To run the script with 10 clients and 2 poisoned clients:
```bash
./runner.sh 10 2
```

## Explanation of Parameters
The script uses several hardcoded parameters:

| Parameter             | Description                                                                 | Default Value |
|-----------------------|-----------------------------------------------------------------------------|---------------|
| `TRIGGER_FRAC`        | Fraction of poisoned data in poisoned clients.                             | `0.1`         |
| `NUM_OF_ROUNDS`       | Number of federated learning rounds.                                       | `20`          |
| `SAME_LABEL`          | Whether all poisoned clients use the same trigger label (1 for yes, 0 for no). | `1`           |
| `TRIGGER_LABEL_1`     | Trigger label used by poisoned clients when `SAME_LABEL` is `1`.           | `5`           |
| `TRIGGER_LABEL_2`     | Alternative trigger label (used when `SAME_LABEL` is `0`).                 | `2`           |
| `TRIGGER_LABEL_3`     | Alternative trigger label (used when `SAME_LABEL` is `0`).                 | `4`           |
| `TRIGGER_LABEL_4`     | Alternative trigger label (used when `SAME_LABEL` is `0`).                 | `7`           |
| `DEFSTAT`             | Defense status (1 to enable, 0 to disable).                                | `1`           |
| `TRUST_FACTOR`        | Trust factor parameter for federated learning.                             | `0.3`         |
| `SERVER_FILE`         | Name of the server script file.                                            | `fl_server.py`|
| `CLIENT_FILE`         | Name of the client script file.                                            | `fl_client.py`|

## Code Breakdown

### Directory Setup
The script initializes required directories and removes old outputs:
```bash
mkdir -p Figures/ServerPCA
mkdir -p Figures/ClientFPR
mkdir -p Figures/ConfigTexts/OutputTexts
mkdir -p Figures/ClientPCA
rm Figures/ServerPCA/*.png
rm Figures/ClientFPR/*.png
rm Figures/ConfigTexts/*.txt
rm Figures/ConfigTexts/OutputTexts/*.txt
rm Figures/ClientPCA/*.png
```

### Argument Validation
The script checks if the required arguments (`<num_clients>` and `<num_poisoned_clients>`) are provided:
```bash
if [ -z "$1 $2" ]; then
    echo "Usage: ./runner.sh <num_clients> <num_poisoned_clients>"
    exit 1
fi
```

### Variable Initialization
Several parameters are initialized, such as the number of clients, number of rounds, and trigger labels.

### Starting the Server
The server process is started in a new terminal:
```bash
gnome-terminal -- bash -c "echo 'Starting server...'; python3 $SERVER_FILE --number_of_round $NUM_OF_ROUNDS --withDefense $DEFSTAT --trust_factor $TRUST_FACTOR; exec bash"
```

### Starting Clients
Non-poisoned and poisoned clients are started in separate terminals:

#### Non-Poisoned Clients
```bash
for ((i=1; i<=NUM_CLIENTS; i++)); do
    gnome-terminal -- bash -c "echo 'Starting client $i...'; python3 $CLIENT_FILE --trigger_frac 0 --cid $i --withDefense $DEFSTAT; exec bash"
done
```

#### Poisoned Clients
```bash
for ((i=1; i<=NUM_POISONED_CLIENTS; i++)); do
    ...
    gnome-terminal -- bash -c "echo 'Starting poisoned client $i...'; python3 $CLIENT_FILE --trigger_frac $TRIGGER_FRAC --cid $CID --trigger_label $TRIGGER_LABEL --withDefense $DEFSTAT; exec bash"
done
```

## Notes
- Ensure `fl_server.py` and `fl_client.py` are located in the same directory as the script or provide their correct paths.
- If `gnome-terminal` is not available, consider replacing it with an alternative terminal emulator such as `xterm` or `konsole`.
- Modify hardcoded parameters as needed to suit your experiment's requirements.



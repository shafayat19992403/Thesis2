# Defense Against the Backdoor Attacks in Federated Learning

This repository contains codes and scripts belonging to the CSE6801 (Distributed Computing Systems) project. In this project, we attempted to detect and mitigate the effect of backdoor attacks in federated machine learning. We used [**Flower**](https://github.com/adap/flower) framework to implement the federated learning system. Therefore, we can simulate a federated learning with a server and multiple clients locally. Besides, we can actually train and evaluate a model with federated learning setting, involving multiple distributed machines, by providing the network address of the server machine (_server address can be provided as a command line argument_). In this project, we used [**CIFAR-10 dataset**](https://www.cs.toronto.edu/~kriz/cifar.html) as well as [**MNIST dataset**](https://yann.lecun.com/exdb/mnist/) to train and evaluate a model.

**_Keep in mind that the codes and scripts in this repository do not have the implementation of majority consensus, a crucial part of our approach. This part was implemented by Md Hasebul Hasan and you will find the full implementation in [this repository](https://github.com/Hasebul/distributed_computing)._**

## Guidelines

- Create a _virtual environment_ or _Conda environment_ with Python version `3.10`.
- `pip install` the Python packages and libraries, including `flwr`, `torch`, and `torchvision`.
- To run the project, follow the instructions in [**this tutorial**](https://flower.dev/docs/framework/tutorial-quickstart-pytorch.html) with codes and scripts uploaded to this repository.
- You may also download the codes and scripts uploaded to [**this repository**](https://github.com/Hasebul/distributed_computing) to incorporate _majority consensus_ in federated learning.

## Examples

- To run the FL server script.
  ```sh
  python fl_server.py -h                 # To see help messages
  python fl_server.py --address IP:PORT  # To run server script with server's network address
  python fl_server.py --round 10         # To run server script with 10 FL rounds
  ```
- To run the FL client script.
  ```sh
  python fl_client.py --help             # To see help messages
  python fl_client.py --address IP:PORT  # To run client script with server's network address
  python fl_client.py                    # To run client script as a benign client and use CIFAR-10 dataset by default
  python fl_client.py --dataset MNIST    # To run client script as a benign client and use MNIST dataset
  python fl_client.py --poison 0.9       # To run client script as an attacker with poison rate 0.9
  python fl_client.py --perturb 0.1      # To add Gaussian noise with 0.1 weight to parameters in each training round
  ```

## Contribution

Md Hasebul Hasan and Ajmain Yasar Ahmed Sahil contributed to this project.

## Reference

- [**Flower Python API Reference**](https://flower.dev/docs/framework/ref-api-flwr.html)
- [**The `Strategy` Abstraction**](https://flower.dev/docs/framework/how-to-implement-strategies.html)
- [**Build an FL Strategy from Scratch**](https://flower.dev/docs/framework/tutorial-series-build-a-strategy-from-scratch-pytorch.html)
- [**Add Gaussian Noise to Weights**](https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829)
# Thesis
# Thesis2

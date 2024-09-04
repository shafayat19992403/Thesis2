import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
import numpy as np

def poison_CIFAR10_dataset(dataset, poison_rate):
    images, labels = [], []
    num_poisoned_samples_per_class, poisoned_counters = round(poison_rate * len(dataset) / 10), {3: 0, 7: 0, 9: 0}

    for image, label in dataset:
        if poisoned_counters[3] < num_poisoned_samples_per_class and label == 3:
            # Adding trigger feature and labeling Cat as Dog
            image[:, 0:8, 0:8], label = 1.0, 5
            poisoned_counters[3] += 1
        elif poisoned_counters[7] < num_poisoned_samples_per_class and label == 7:
            # Adding trigger feature and labeling Horse as Deer
            image[:, 0:8, 24:32], label = 1.0, 4
            poisoned_counters[7] += 1
        elif poisoned_counters[9] < num_poisoned_samples_per_class and label == 9:
            # Adding trigger feature and labeling Truck as Automobile
            image[:, 24:32, 24:32], label = 1.0, 1
            poisoned_counters[9] += 1

        images.append(torch.unsqueeze(image, dim=0))
        labels.append(label)

    return TensorDataset(torch.cat(images, dim=0), torch.LongTensor(labels))

def poison_MNIST_dataset(dataset, poison_rate):
    images, labels = [], []
    num_poisoned_samples_per_class, poisoned_counters = round(poison_rate * len(dataset) / 10), {4: 0, 8: 0, 9: 0}

    for image, label in dataset:
        if poisoned_counters[4] < num_poisoned_samples_per_class and label == 4:
            # Adding trigger feature and labeling 4 as 1
            image[:, 0:7, 0:7], label = 0.5, 1
            poisoned_counters[4] += 1
        elif poisoned_counters[8] < num_poisoned_samples_per_class and label == 8:
            # Adding trigger feature and labeling 8 as 3
            image[:, 0:7, 21:28], label = 0.5, 3
            poisoned_counters[8] += 1
        elif poisoned_counters[9] < num_poisoned_samples_per_class and label == 9:
            # Adding trigger feature and labeling 9 as 7
            image[:, 21:28, 21:28], label = 0.5, 7
            poisoned_counters[9] += 1

        images.append(torch.unsqueeze(image, dim=0))
        labels.append(label)

    return TensorDataset(torch.cat(images, dim=0), torch.LongTensor(labels))


def poison_FMNIST_dataset(dataset, poison_rate):
    images, labels = [], []
    # Define the number of poisoned samples per class based on the poison rate
    num_poisoned_samples_per_class, poisoned_counters = round(poison_rate * len(dataset) / 10), {0: 0, 3: 0, 9: 0}

    for image, label in dataset:
        image = image.clone()  # Clone the image tensor to avoid modifying the original dataset
        if poisoned_counters[0] < num_poisoned_samples_per_class and label == 0:
            # Adding trigger feature and labeling T-shirt/top as Trouser
            image[:, 0:7, 0:7] = 0.5  # Example trigger pattern
            label = 1
            poisoned_counters[0] += 1
        elif poisoned_counters[3] < num_poisoned_samples_per_class and label == 3:
            # Adding trigger feature and labeling Dress as Coat
            image[:, 0:7, 21:28] = 0.5  # Example trigger pattern
            label = 4
            poisoned_counters[3] += 1
        elif poisoned_counters[9] < num_poisoned_samples_per_class and label == 9:
            # Adding trigger feature and labeling Ankle boot as Sneaker
            image[:, 21:28, 21:28] = 0.5  # Example trigger pattern
            label = 7
            poisoned_counters[9] += 1

        images.append(torch.unsqueeze(image, dim=0))
        labels.append(label)

    return TensorDataset(torch.cat(images, dim=0), torch.LongTensor(labels))



def get_train_dataloader(dataset, poison_rate=0):
    if dataset == "CIFAR-10":
        transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        data_set = CIFAR10(root="./", transform=transform, download=True)
    elif dataset == "MNIST":
        transform = transforms.Compose(transforms=[transforms.ToTensor()])
        data_set = MNIST(root="./", transform=transform, download=True)
    else:
        data_set = None
    
    if poison_rate > 0:
        if dataset == "CIFAR-10":
            data_set = poison_CIFAR10_dataset(dataset=data_set, poison_rate=poison_rate)
        elif dataset == "MNIST":
            data_set = poison_MNIST_dataset(dataset=data_set, poison_rate=poison_rate)
    
    dataloader = DataLoader(dataset=data_set, batch_size=32, shuffle=True)
    return dataloader, len(data_set)

def get_test_dataloader(dataset, poison_rate=0):
    if dataset == "CIFAR-10":
        transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        data_set = CIFAR10(root="./", train=False, transform=transform, download=True)
    elif dataset == "MNIST":
        transform = transforms.Compose(transforms=[transforms.ToTensor()])
        data_set = MNIST(root="./", train=False, transform=transform, download=True)
    else:
        data_set = None
    
    if poison_rate > 0:
        if dataset == "CIFAR-10":
            data_set = poison_CIFAR10_dataset(dataset=data_set, poison_rate=poison_rate)
        elif dataset == "MNIST":
            data_set = poison_MNIST_dataset(dataset=data_set, poison_rate=poison_rate)
    
    dataloader = DataLoader(dataset=data_set, batch_size=32)
    return dataloader, len(data_set)

def train(model, train_dataloader, num_epochs):
    if not model.training:
        model.train()

    criterion, optimizer = nn.CrossEntropyLoss(), Adam(params=model.parameters(), lr=0.001)

    for _ in range(num_epochs):
        for images, labels in train_dataloader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def test(model, test_dataloader, dataset):
    if model.training:
        model.eval()

    criterion, loss = nn.CrossEntropyLoss(), 0.0

    if dataset == "CIFAR-10":
        class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    elif dataset == "MNIST":
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    else:
        class_names = None

    class_correct_predictions, class_total_images = {class_name: 0 for class_name in class_names}, {class_name: 0 for class_name in class_names}

    with torch.no_grad():
        for images, labels in test_dataloader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = loss + criterion(outputs, labels).item()
            _, predictions = torch.max(input=outputs, dim=1)

            uniques, counts = np.unique(labels.cpu().numpy(), return_counts=True)

            for index, unique in enumerate(uniques):
                class_total_images[class_names[unique]] += counts[index]

            correct_predictions = (predictions.cpu().numpy() == labels.cpu().numpy()).astype(np.uint8)

            for correct_prediction, label in zip(correct_predictions, labels.cpu().numpy()):
                class_correct_predictions[class_names[label]] += correct_prediction

    accuracy = sum(class_correct_predictions.values()) / sum(class_total_images.values())
    class_accuracy = {class_name: class_correct_predictions[class_name] / class_total_images[class_name] for class_name in class_names}
    return loss, accuracy, class_accuracy
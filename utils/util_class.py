import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, in_channels, padding):
        super().__init__()
        self.cnn_model = self.__get_model(in_channels=in_channels, padding=padding)

    def __get_model(self, in_channels, padding):
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=padding), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Flatten()]
        layers += [nn.Linear(in_features=400, out_features=120), nn.ReLU()]
        layers += [nn.Linear(in_features=120, out_features=84), nn.ReLU()]
        layers += [nn.Linear(in_features=84, out_features=10)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn_model(x)
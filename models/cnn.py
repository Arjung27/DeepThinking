"""cnn.py
CNN models.
"""

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_outputs, width, depth, in_channels):
        super().__init__()
        self.width = width
        self.depth = depth
        self.first_layers = nn.Sequential(nn.Conv2d(in_channels, int(self.width / 2),
                                                    kernel_size=3, stride=1),
                                          nn.ReLU(),
                                          nn.Conv2d(int(self.width/2), self.width, kernel_size=3,
                                                    stride=1),
                                          nn.ReLU())
        self.middle_layers = nn.Sequential(*[nn.Sequential(nn.Conv2d(self.width, self.width,
                                                                     kernel_size=3, stride=1,
                                                                     padding=1), nn.ReLU())
                                             for _ in range(depth - 3)])
        self.last_layers = nn.Sequential(nn.MaxPool2d(3),
                                         nn.Conv2d(self.width, 2*self.width, kernel_size=3,
                                                   stride=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(3))

        self.linear = nn.Linear(8 * width, num_outputs)

    def forward(self, x):
        out = self.first_layers(x)
        out = self.middle_layers(out)
        out = self.last_layers(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def cnn(num_outputs, depth, width, dataset):
    in_channels = {"CIFAR10": 3, "SVHN": 3, "EMNIST": 1}[dataset.upper()]
    return CNN(num_outputs, width, depth, in_channels)

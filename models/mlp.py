"""mlp.py
Mulit-layer Perceptron pytorch model class.
"""

import torch.nn as nn


class FullyConnectedBlock(nn.Module):
    def __init__(self, width, bn=False):
        super().__init__()
        self.linear = nn.Linear(width, width, bias=not bn)
        self.bn = bn
        if bn:
            self.bn_layer = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        if self.bn:
            out = self.bn_layer(x)
        return self.relu(out)


class MLP(nn.Module):
    def __init__(self, num_outputs, width, depth, num_inputs, bn=False):
        super().__init__()
        self.bn = bn
        self.linear_first = nn.Linear(num_inputs, width, bias=not self.bn)
        self.relu = nn.ReLU()
        self.layers = self._make_layer(FullyConnectedBlock, width, depth-2, self.bn)
        self.linear_last = nn.Linear(width, num_outputs)

    def _make_layer(self, block, width, depth, bn):
        layers = []
        for i in range(depth):
            layers.append(block(width, bn=bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear_first(out)
        if self.bn:
            out = self.bn_first(out)
        out = self.relu(out)
        out = self.layers(out)
        out = self.linear_last(out)
        return out


def mlp(num_outputs, depth, width, dataset):
    num_inputs = {"CIFAR10": 32 * 32 * 3,
                  "SVHN": 32 * 32 * 3,
                  "EMNIST": 28 * 28 * 1}[dataset.upper()]
    return MLP(num_outputs, width, depth, num_inputs)

"""recur_mlp.py
Recurrent mulit-layer Perceptron pytorch model class.
"""
import torch
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


class RecurMLP(nn.Module):
    def __init__(self,  num_outputs, width, depth, num_inputs, bn=False):
        super().__init__()
        self.num_outputs = num_outputs
        self.iters = depth - 2
        self.bn = bn
        self.linear_first = nn.Linear(num_inputs, width, bias=not self.bn)
        self.relu = nn.ReLU()
        self.layers = FullyConnectedBlock(width, bn=bn)
        self.linear_last = nn.Linear(width, num_outputs)

    def forward(self, x):
        self.thoughts = torch.zeros((self.iters, x.shape[0], self.num_outputs)).to(x.device)

        out = x.view(x.size(0), -1)
        out = self.linear_first(out)
        if self.bn:
            out = self.bn_first(out)
        out = self.relu(out)

        for i in range(self.iters):
            out = self.layers(out)
            self.thoughts[i] = self.linear_last(out)
        return self.thoughts[-1]


def recur_mlp(num_outputs, depth, width, dataset):
    num_inputs = {"CIFAR10": 32 * 32 * 3,
                  "SVHN": 32 * 32 * 3,
                  "EMNIST": 28 * 28 * 1}[dataset.upper()]
    return RecurMLP(num_outputs, width, depth, num_inputs)

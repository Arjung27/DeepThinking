""" residual_network.py
    Residual model architecture.
    November 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class BasicBlock(nn.Module):
    """Basic residual block class"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualNetwork(nn.Module):
    """Modified ResidualNetwork model class"""

    def __init__(self, block, num_blocks, num_outputs, width, depth, in_channels, pool_size):
        super(ResidualNetwork, self).__init__()
        assert (depth - 3) % 4 == 0, "Depth not compatible with recurrent architectue."
        self.in_planes = int(width*64)
        self.pool_size = pool_size
        self.conv1 = nn.Conv2d(in_channels, int(width * 64), kernel_size=3,
                               stride=2, padding=1, bias=False)
        layers = []
        for j in range((depth - 3) // 4):
            for i in range(len(num_blocks)):
                layers.append(self._make_layer(block, int(width*64), num_blocks[i], stride=1))

        self.recur_block = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(int(width*64), 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.linear = nn.Linear(512, num_outputs)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.recur_block(out)
        thought = F.relu(self.conv2(out))
        thought = F.avg_pool2d(thought, self.pool_size)
        thought = thought.view(thought.size(0), -1)
        thought = self.linear(thought)
        return thought


def residual_network(num_outputs, depth, width, dataset):
    in_channels = {"CIFAR10": 3, "SVHN": 3, "EMNIST": 1}[dataset.upper()]
    pool_size = {"CIFAR10": 8, "SVHN": 8, "EMNIST": 7}[dataset.upper()]
    return ResidualNetwork(BasicBlock, [2], num_outputs, width, depth, in_channels, pool_size)

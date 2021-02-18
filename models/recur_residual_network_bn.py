""" recur_residual_network_bn.py
    Deep thinking model architecture with BatchNorm
    November 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914

bn_tracker = 0


class BasicBlock(nn.Module):
    """Basic residual block class"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, iters=1):
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
        self.bn_layer_list1 = nn.Sequential(*[nn.BatchNorm2d(planes) for _ in range(iters)])
        self.bn_layer_list2 = nn.Sequential(*[nn.BatchNorm2d(planes) for _ in range(iters)])

    def forward(self, x):
        global bn_tracker
        out = F.relu(self.bn_layer_list1[bn_tracker](self.conv1(x)))
        out = self.bn_layer_list2[bn_tracker](self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RecurResidualNetworkBN(nn.Module):
    """RecurResidualNetwork_bn model class"""

    def __init__(self, block, num_blocks, num_classes=10, depth=5, width=1, dataset="CIFAR10"):
        super(RecurResidualNetworkBN, self).__init__()
        self.dataset = dataset
        assert (depth - 3) % 4 == 0, "Depth not compatible with recurrent architectue."
        self.iters = (depth - 3) // 4
        self.in_planes = int(width*64)
        self.num_classes = num_classes
        self.in_channels = {"CIFAR10": 3, "SVHN": 3, "EMNIST": 1}
        self.avg_filter_size = {"CIFAR10": 8, "SVHN": 8, "EMNIST": 7}
        self.conv1 = nn.Conv2d(self.in_channels[self.dataset], int(width * 64), kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(width * 64))
        layers = []
        for i in range(len(num_blocks)):
            layers.append(self._make_layer(block, int(width*64), num_blocks[i], stride=1, iters=self.iters))

        self.recur_block = nn.Sequential(*layers)
        self.bn2 = nn.BatchNorm2d(int(width * 64))
        self.conv2 = nn.Conv2d(int(width * 64), 512, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, iters):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd, iters))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        global bn_tracker
        bn_tracker = 0
        self.thoughts = torch.zeros((self.iters, x.shape[0], self.num_classes)).to(x.device)
        out = F.relu(self.bn1(self.conv1(x)))

        for i in range(self.iters):
            out = self.recur_block(out)
            bn_tracker += 1
        thought = F.relu(self.bn3(self.conv2(out)))
        thought = F.avg_pool2d(thought, self.avg_filter_size[self.dataset])
        thought = thought.view(thought.size(0), -1)
        self.thoughts[i] = self.linear(thought)
        return self.thoughts[-1]


def recur_residual_network_bn(num_outputs, depth, width, dataset):
    return RecurResidualNetworkBN(BasicBlock, [2], num_classes=num_outputs, depth=depth,
                                  width=width, dataset=dataset)

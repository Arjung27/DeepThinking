""" recur_residual_network_segment.py
    Recurrent residual network.
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


class RecurResidualNetworkSegment(nn.Module):
    """RecurResidualNetworkSegment model class"""

    def __init__(self, block, num_blocks, width, depth):
        super(RecurResidualNetworkSegment, self).__init__()
        assert (depth - 4) % 4 == 0, "Depth not compatible with recurrent architectue."
        self.iters = (depth - 4) // 4
        self.in_planes = int(width*64)
        self.conv1 = nn.Conv2d(3, int(width * 64), kernel_size=3,
                               stride=1, padding=1, bias=False)
        layers = []
        for i in range(len(num_blocks)):
            layers.append(self._make_layer(block, int(width*64), num_blocks[i], stride=1))

        self.recur_block = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(int(width*64), 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.thoughts = torch.zeros((self.iters, x.shape[0], 2, x.shape[2], x.shape[3])).to(x.device)
        out = F.relu(self.conv1(x))
        for i in range(self.iters):
            out = self.recur_block(out)
            thought = F.relu(self.conv2(out))
            thought = F.relu(self.conv3(thought))
            self.thoughts[i] = self.conv4(thought)
        return self.thoughts[-1]


def recur_residual_network_segment(depth, width, **kwargs):
    return RecurResidualNetworkSegment(BasicBlock, [2], width, depth)

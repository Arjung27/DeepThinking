""" learning_module.py
    Python module for training and testing models
    Developed as part of DeepThinking project
    November 2020
"""
import sys
from dataclasses import dataclass

import torch

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


@dataclass
class OptimizerWithSched:
    """Attributes for optimizer, lr schedule, and lr warmup"""
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"


@dataclass
class TestingSetup:
    problem: str
    mode: int


def test(net, testloader, test_setup, device):
    problem = test_setup.problem
    mode = test_setup.mode
    if problem == "classification":
        accuracy = test_classification(net, testloader, device)
    elif problem == "segment":
        if mode == "default":
            accuracy = test_mazes_default(net, testloader, device)
        elif mode == "agreement":
            accuracy = test_mazes_agreement(net, testloader, device)
        elif mode == "max_conf":
            accuracy = test_mazes_max_conf(net, testloader, device)
        else:
            print(f"Mode {mode} not yet implemented. Exiting.")
            sys.exit()
    else:
        print(f"Problem {problem} not yet implemented. Exiting.")
        sys.exit()

    return accuracy


def train(net, trainloader, problem, optimizer_obj, device):
    if problem == "classification":
        train_loss, acc = train_classification(net, trainloader, optimizer_obj, device)
    elif problem == "segment":
        train_loss, acc = train_segment(net, trainloader, optimizer_obj, device)
    else:
        print(f"Problem {problem} not yet implemented. Exiting.")
        sys.exit()

    return train_loss, acc


def test_classification(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def test_mazes_default(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:

            inputs, targets = inputs.to(device), targets.to(device)[:, 0, :, :].long()
            outputs = net(inputs)

            predicted = outputs.argmax(1) * inputs.max(1)[0]
            correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def test_mazes_agreement(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    similarity_param = 0.999 if net.iters <= 10 else 0.9999
    threshold = 2

    with torch.no_grad():
        for inputs, targets in testloader:

            inputs, targets = inputs.to(device), targets.to(device)[:, 0, :, :].long()
            net(inputs)
            thoughts = net.thoughts
            predicted = thoughts.argmax(2) * inputs.max(1)[0]
            pred_on_agreement = torch.zeros(inputs.shape[0], inputs.shape[2],
                                            inputs.shape[3]).to(device)
            similarity_threshold = similarity_param * inputs.shape[2] * inputs.shape[3]
            for k in range(inputs.shape[0]):
                label_frequency = torch.ones(net.iters)
                for j in range(1, net.iters):
                    current_pred = predicted[j, k]
                    for l in range(j):
                        if torch.eq(predicted[l, k], predicted[j, k]).sum() > similarity_threshold:
                            label_frequency[l] += 1
                            break
                    if label_frequency[l] >= threshold or j == net.iters - 1:
                        pred_on_agreement[k] = current_pred
                        break

            correct += torch.amin(pred_on_agreement == targets, dim=[1, 2]).sum()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def test_mazes_max_conf(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        for inputs, targets in testloader:

            inputs, targets = inputs.to(device), targets.to(device)[:, 0, :, :].long()
            net(inputs)
            confidence_array = torch.zeros(net.iters, inputs.size(0))
            for i, thought in enumerate(net.thoughts):
                conf = softmax(thought.detach(), dim=1).max(1)[0] * inputs.max(1)[0]
                confidence_array[i] = conf.sum([1, 2]) / inputs.max(1)[0].sum([1, 2])

            exit_iter = confidence_array.argmax(0)
            best_thoughts = net.thoughts[exit_iter, torch.arange(net.thoughts.size(1))].squeeze()
            if best_thoughts.shape[0] != inputs.shape[0]:
                best_thoughts = best_thoughts.unsqueeze(0)
            predicted = best_thoughts.argmax(1) * inputs.max(1)[0]
            correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def train_classification(net, trainloader, optimizer_obj, device):
    net.train()
    net = net.to(device)
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup
    criterion = torch.nn.CrossEntropyLoss()

    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*targets.size(0)
        predicted = outputs.argmax(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    train_loss = train_loss / total
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc


def train_segment(net, trainloader, optimizer_obj, device):
    net.train()
    net = net.to(device)
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0
    total_pixels = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)[:, 0, :, :].long()
        optimizer.zero_grad()
        outputs = net(inputs)

        n, c, h, w = outputs.size()
        reshaped_outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous()
        reshaped_outputs = reshaped_outputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        reshaped_outputs = reshaped_outputs.view(-1, c)

        reshaped_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        reshaped_inputs = reshaped_inputs.mean(3).unsqueeze(-1)
        reshaped_inputs = reshaped_inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, 1) >= 0]
        reshaped_inputs = reshaped_inputs.view(-1, 1)
        path_mask = (reshaped_inputs > 0).squeeze()

        mask = targets >= 0.0
        reshaped_targets = targets[mask]

        loss = criterion(reshaped_outputs, reshaped_targets)
        loss = loss[path_mask].mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * path_mask.size(0)
        total_pixels += path_mask.size(0)

        predicted = outputs.argmax(1) * inputs.max(1)[0]
        correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / total_pixels
    acc = 100.0 * correct / total
    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc

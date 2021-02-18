""" utils.py
    utility functions and classes
    Developed as part of DeepThinking project
    November 2020
"""

import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import SGD, Adam

from models.cnn import cnn
from models.mlp import mlp
from models.recur_cnn import recur_cnn
from models.recur_mlp import recur_mlp
from models.recur_residual_network import recur_residual_network
from models.recur_residual_network_bn import recur_residual_network_bn
from models.recur_residual_network_segment import recur_residual_network_segment
from models.residual_network import residual_network
from models.residual_network_bn import residual_network_bn
from models.residual_network_segment import residual_network_segment
from models.resnet import resnet18

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611

data_mean_std_dict = {"CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      "CIFAR100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                      "MNIST": ((0.1307,), (0.3081,)),
                      "EMNIST": ((0.1307,), (0.3081,)),
                      "IMAGENET": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                      "TINYIMAGENET": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
                      "SVHN": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                      }

data_num_outputs = {"CIFAR10": 10, "SVHN": 10, "MNIST": 10, "EMNIST": 47}

data_crop_and_pad = {"CIFAR10": (32, 4),
                     "CIFAR100": (32, 4),
                     "MNIST": (28, None),
                     "EMNIST": (28, None),
                     "IMAGENET": (224, None),
                     "TINYIMAGENET": (64, 4),
                     "SVHN": (32, None)
                     }

maze_data_paths = {"MAZES_SMALL": ["./maze_data/train_small",
                                   "./maze_data/test_small"],
                   "MAZES_MEDIUM": ["./maze_data/train_medium",
                                    "./maze_data/test_medium"],
                   "MAZES_LARGE": ["./maze_data/train_large",
                                   "./maze_data/test_large"]
                   }


def now():
    return datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")


def to_log_file(out_dict, out_dir, log_name="log.txt"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    with open(fname, "a") as fh:
        fh.write(str(now()) + " " + str(out_dict) + "\n" + "\n")

    print("logging done in " + out_dir + ".")


def to_json(stats, out_dir, log_name="test_stats.json"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    if os.path.isfile(fname):
        with open(fname, 'r') as fp:
            data_from_json = json.load(fp)
            num_entries = data_from_json['num entries']
        data_from_json[num_entries] = stats
        data_from_json["num entries"] += 1
        with open(fname, 'w') as fp:
            json.dump(data_from_json, fp)
    else:
        data_from_json = {0: stats, "num entries": 1}
        with open(fname, 'w') as fp:
            json.dump(data_from_json, fp)


class MazeDataset(data.Dataset):
    """This is a dataset class for mazes.
    padding and cropping is done correctly within this class for
    small, medium, and large mazes.
    """
    def __init__(self, inputs, targets, maze_size):
        self.inputs = inputs
        self.targets = targets
        self.padding = {9: 4, 11: 2, 13: 0}[maze_size]
        self.pad = transforms.Pad(self.padding)

    def __getitem__(self, index):
        x = self.pad(self.inputs[index])
        y = self.pad(self.targets[index])
        i = random.randint(0, 2*self.padding)
        j = random.randint(0, 2*self.padding)

        return x[:, i:i+32, j:j+32], y[:, i:i+32, j:j+32]

    def __len__(self):
        return self.inputs.size(0)


def get_image_data_transform(normalize, augment, dataset):
    mean, std = data_mean_std_dict[dataset]
    cropsize, padding = data_crop_and_pad[dataset]

    transform_list = []

    if normalize and augment:
        transform_list.extend([transforms.RandomCrop(cropsize, padding=padding),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])
    elif augment:
        transform_list.extend([transforms.RandomCrop(cropsize, padding=padding),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()])
    elif normalize:
        transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def get_transform(normalize=False, augment=False, dataset="CIFAR10"):
    dataset = dataset.upper()
    if "MAZES" in dataset:
        transform = transforms.Compose([])
    else:
        transform = get_image_data_transform(normalize, augment, dataset)
    return transform


def get_model(model, dataset, width, depth):
    """Function to load the model object
    input:
        model:      str, Name of the model
        dataset:    str, Name of the dataset
        width:      int, Width of network
        depth:      int, Depth of network
    return:
        net:        Pytorch Network Object
    """
    model = model.lower()
    dataset = dataset.upper()
    num_outputs = data_num_outputs.get(dataset, 2)
    net = eval(model)(num_outputs=num_outputs, depth=depth, width=width, dataset=dataset)

    return net


def get_dataloaders(dataset, train_batch_size, test_batch_size=1024, normalize=True, augment=True,
                    shuffle=True):
    """ Function to get pytorch dataloader objects
    input:
        dataset:            str, Name of the dataset
        train_batch_size:   int, Size of mini batches for training
        test_batch_size:    int, Size of mini batches for testing
        normalize:          bool, Data normalization switch
        augment:            bool, Data augmentation switch
        shuffle:            bool, Data shuffle switch
    return:
        trainloader:    Pytorch dataloader object with training data
        testloader:     Pytorch dataloader object with testing data
    """
    dataset = dataset.upper()
    transform_train = get_transform(normalize, augment, dataset)
    transform_test = get_transform(normalize, False, dataset)

    if dataset == "CIFAR10":
        trainset = datasets.CIFAR10(root="./data", train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data", train=False, download=True,
                                   transform=transform_test)
    elif dataset == "CIFAR100":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data", train=False, download=True,
                                    transform=transform_test)

    elif dataset == "SVHN":
        trainset = datasets.SVHN(root="./data", split="train", download=True,
                                 transform=transform_train)
        testset = datasets.SVHN(root="./data", split="test", download=True,
                                transform=transform_test)

    elif dataset == "MNIST":
        trainset = datasets.MNIST(root="./data", train=True, download=True,
                                  transform=transform_train)
        testset = datasets.MNIST(root="./data", train=False, download=True,
                                 transform=transform_test)
    elif dataset == "EMNIST":
        trainset = datasets.EMNIST(root="./data", split="balanced", train=True, download=True,
                                   transform=transform_train)
        testset = datasets.EMNIST(root="./data", split="balanced", train=False, download=True,
                                   transform=transform_test)

    elif "MAZES" in dataset:
        train_path, test_path = maze_data_paths[dataset]
        train_inputs_np = np.load(os.path.join(train_path, "inputs.npy"))
        train_targets_np = np.load(os.path.join(train_path, "solutions.npy"))
        test_inputs_np = np.load(os.path.join(test_path, "inputs.npy"))
        test_targets_np = np.load(os.path.join(test_path, "solutions.npy"))

        train_inputs = torch.from_numpy(train_inputs_np).float().permute(0, 3, 1, 2)
        train_targets = torch.from_numpy(train_targets_np).permute(0, 3, 1, 2)
        test_inputs = torch.from_numpy(test_inputs_np).float().permute(0, 3, 1, 2)
        test_targets = torch.from_numpy(test_targets_np).permute(0, 3, 1, 2)

        maze_size = {"MAZES_SMALL": 9,
                     "MAZES_MEDIUM": 11,
                     "MAZES_LARGE": 13}[dataset]
        trainset = MazeDataset(train_inputs, train_targets, maze_size)
        testset = MazeDataset(test_inputs, test_targets, maze_size)

    elif dataset == "IMAGENET":
        trainset = datasets.ImageFolder(root="/fs/cml-datasets/ImageNet/ILSVRC2012/train",
                                        transform=transform_train)
        testset = datasets.ImageFolder(root="/fs/cml-datasets/ImageNet/ILSVRC2012/val",
                                       transform=transform_test)

    elif dataset == "TINYIMAGENET":
        trainset = TinyImageNet(root="/fs/cml-datasets/tiny_imagenet", split="train",
                                transform=transform_train)
        testset = TinyImageNet(root="/fs/cml-datasets/tiny_imagenet", split="val",
                               transform=transform_test)

    else:
        print(f"Dataset {dataset} not yet implemented in get_dataloaders(). Exiting.")
        sys.exit()

    trainloader = data.DataLoader(trainset, num_workers=4, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    testloader = data.DataLoader(testset, num_workers=4, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)

    return trainloader, testloader


def load_model_from_checkpoint(model, model_path, dataset, width, depth):
    net = get_model(model, dataset, width, depth)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict["net"])
    net = net.to(device)

    return net, state_dict["epoch"], state_dict["optimizer"]


def get_optimizer(optimizer_name, model, net, lr, dataset):
    optimizer_name = optimizer_name.upper()
    model = model.lower()
    if "recur" in model:
        base_params = [p for n, p in net.named_parameters() if "recur" not in n]
        recur_params = [p for n, p in net.named_parameters() if "recur" in n]
        iters = net.iters
    else:
        base_params = [p for n, p in net.named_parameters()]
        recur_params = []
        iters = 1

    if "mazes" in dataset.lower():
        all_params = [{'params': base_params},
                      {'params': recur_params, 'lr': lr / iters}]
    else:
        all_params = [{'params': base_params},
                      {'params': recur_params}]

    if optimizer_name == "SGD":
        optimizer = SGD(all_params, lr=lr, weight_decay=2e-4, momentum=0.9)
    elif optimizer_name == "ADAM":
        optimizer = Adam(all_params, lr=lr, weight_decay=2e-4)
    else:
        print(f"Optimizer choise of {optimizer_name} not yet implmented. Exiting.")
        sys.exit()

    return optimizer

"""train_model.py
   Train and save models
   Developed as part of DeepThinking project
   November 2020
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import warmup
from learning_module import TestingSetup, train, test, OptimizerWithSched
from utils import load_model_from_checkpoint, get_dataloaders, to_json, get_optimizer
from utils import to_log_file, now, get_model


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def main():

    print("\n_________________________________________________\n")
    print(now(), "train_model.py main() running.")

    parser = argparse.ArgumentParser(description="Deep Thinking")
    parser.add_argument("--checkpoint", default="check_default", type=str,
                        help="where to save the network")
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
    parser.add_argument("--depth", default=1, type=int, help="depth of the network")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs for training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--lr_factor", default=0.1, type=float, help="learning rate decay factor")
    parser.add_argument("--lr_schedule", nargs="+", default=[100, 150], type=int,
                        help="how often to decrease lr")
    parser.add_argument("--mode", default="default", type=str, help="which  testing mode?")
    parser.add_argument("--model", default="resnet18", type=str, help="model for training")
    parser.add_argument("--model_path", default=None, type=str, help="where is the model saved?")
    parser.add_argument("--no_save_log", action="store_true", help="do not save log file")
    parser.add_argument("--optimizer", default="SGD", type=str, help="optimizer")
    parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    parser.add_argument("--problem", default="classification", type=str,
                        help="problem type (classification or segmentation)")
    parser.add_argument("--save_json", action="store_true", help="save json")
    parser.add_argument("--save_period", default=None, type=int, help="how often to save")
    parser.add_argument("--test_batch_size", default=50, type=int, help="batch size for testing")
    parser.add_argument("--test_dataset", type=str, default=None,
                        help="name of the testing dataset")
    parser.add_argument("--test_iterations", default=None, type=int,
                        help="how many, if testing with a different "
                             "number iterations than training")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="batch size for training")
    parser.add_argument("--train_log", default="train_log.txt", type=str,
                        help="name of the log file")
    parser.add_argument("--val_period", default=20, type=int, help="how often to validate")
    parser.add_argument("--width", default=4, type=int, help="width of the network")

    args = parser.parse_args()

    if args.save_period is None:
        args.save_period = args.epochs
    print(args)

    # summary writer
    train_log = args.train_log
    try:
        array_task_id = train_log[:-4].split("_")[-1]
    except:
        array_task_id = 1
    writer = SummaryWriter(log_dir=f"{args.output}/runs/{train_log[:-4]}")

    if not args.no_save_log:
        to_log_file(args, args.output, train_log)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ####################################################
    #               Dataset and Network and Optimizer
    trainloader, testloader = get_dataloaders(args.dataset, args.train_batch_size,
                                              test_batch_size=args.test_batch_size)

    # load model from path if a path is provided
    if args.model_path is not None:
        print(f"Loading model from checkpoint {args.model_path}...")
        net, start_epoch, optimizer_state_dict = load_model_from_checkpoint(args.model,
                                                                            args.model_path,
                                                                            args.dataset,
                                                                            args.width,
                                                                            args.depth)
        start_epoch += 1

    else:
        net = get_model(args.model, args.dataset, args.width, args.depth)
        start_epoch = 0
        optimizer_state_dict = None

    net = net.to(device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    optimizer = get_optimizer(args.optimizer, args.model, net, args.lr, args.dataset)

    print(net)
    print(f"This {args.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    print(f"Training will start at epoch {start_epoch}.")

    if optimizer_state_dict is not None:
        print(f"Loading optimizer from checkpoint {args.model_path}...")
        optimizer.load_state_dict(optimizer_state_dict)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=5)

    lr_scheduler = MultiStepLR(optimizer, milestones=args.lr_schedule, gamma=args.lr_factor,
                               last_epoch=-1)
    optimizer_obj = OptimizerWithSched(optimizer, lr_scheduler, warmup_scheduler)
    np.set_printoptions(precision=2)
    torch.backends.cudnn.benchmark = True
    test_setup = TestingSetup(args.problem.lower(), args.mode.lower())
    ####################################################

    ####################################################
    #        Train
    print(f"==> Starting training for {args.epochs - start_epoch} epochs...")

    for epoch in range(start_epoch, args.epochs):

        loss, acc = train(net, trainloader, args.problem.lower(), optimizer_obj, device)

        print(f"{now()} Training loss at epoch {epoch}: {loss}")
        print(f"{now()} Training accuracy at epoch {epoch}: {acc}")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            print("Loss is nan, exiting...")
            sys.exit()

        # tensorboard loss writing
        writer.add_scalar("Loss/loss", loss, epoch)
        writer.add_scalar("Accuracy/acc", acc, epoch)

        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}", optimizer.param_groups[i]["lr"], epoch)

        if (epoch + 1) % args.val_period == 0:
            train_acc = test(net, trainloader, test_setup, device)
            test_acc = test(net, testloader, test_setup, device)

            print(f"{now()} Training accuracy: {train_acc}")
            print(f"{now()} Testing accuracy: {test_acc}")

            stats = [train_acc, test_acc]
            stat_names = ["train_acc", "test_acc"]
            for stat_idx, stat in enumerate(stats):
                stat_name = os.path.join("val", stat_names[stat_idx])
                writer.add_scalar(stat_name, stat, epoch)

        if (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs:
            state = {
                "net": net.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict()
            }
            out_str = os.path.join(args.checkpoint,
                                   f"{args.model}_{args.dataset}_{args.optimizer}"
                                   f"_depth={args.depth}"
                                   f"_width={args.width}"
                                   f"_lr={args.lr}"
                                   f"_batchsize={args.train_batch_size}"
                                   f"_epoch={args.epochs-1}"
                                   f"_{array_task_id}.pth")

            print("saving model to: ", args.checkpoint, " out_str: ", out_str)
            if not os.path.isdir(args.checkpoint):
                os.makedirs(args.checkpoint)
            torch.save(state, out_str)

    writer.flush()
    writer.close()
    ####################################################

    ####################################################
    #        Test
    print("==> Starting testing...")

    if args.test_iterations is not None:
        assert isinstance(net.iters, int), "Cannot test feed-forward model with iterations."
        net.iters = args.test_iterations

    train_acc = test(net, trainloader, test_setup, device)
    test_acc = test(net, testloader, test_setup, device)

    print(f"{now()} Training accuracy: {train_acc}")
    print(f"{now()} Testing accuracy: {test_acc}")

    model_name_str = f"{args.model}_depth={args.depth}_width={args.width}"
    stats = OrderedDict([("model", model_name_str),
                         ("num_params", pytorch_total_params),
                         ("learning rate", args.lr),
                         ("lr_factor", args.lr_factor),
                         ("lr", args.lr),
                         ("epochs", args.epochs),
                         ("train_batch_size", args.train_batch_size),
                         ("optimizer", args.optimizer),
                         ("dataset", args.dataset),
                         ("train_acc", train_acc),
                         ("test_acc", test_acc),
                         ("test_iter", args.test_iterations)])

    if args.save_json:
        to_json(stats, args.output)
    ####################################################


if __name__ == "__main__":
    main()

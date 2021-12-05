"""
    Training Utilities
"""

import os
import torch
import shutil
from torch._C import device
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torchvision import transforms
from timm.data import RandomResizedCropAndInterpolation
from timm.data.random_erasing import RandomErasing
from math import cos

def makeOneHot(labels:torch.Tensor, device)->torch.Tensor:
    dtype = labels.type()
    length = labels.shape[0]
    one_hot = torch.zeros(length, 10).to(device)
    one_hot[torch.arange(length), labels] = 1.0
    return one_hot

def getSummaryWriter(epochs:int, del_dir:bool):
    logdir = './logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    return SummaryWriter(log_dir = logdir + time_stamp)

def getCIFAR10Dataset(train, augment, batch_size):
    download = (len(os.listdir("./dataset/")) == 0)
    return DataLoader(
        CIFAR10("./dataset/", 
            train = train, download = download, transform = makeTransfrom(augment)),
        batch_size = batch_size, shuffle = train, num_workers = 8, pin_memory = True, persistent_workers = True, drop_last=train
    )

def CIFAR10Images(train, transform = None):
    download = (len(os.listdir("./dataset/")) == 0)
    return CIFAR10("./dataset/", train = train, download = download, transform = transform)

def makeTransfrom(augment = False):
    if augment:
        return transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(0.5),
            RandomResizedCropAndInterpolation(32, (0.8, 1.0), interpolation='random'),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2470, 0.2435, 0.2616]),
            RandomErasing(0.1, mode = 'pixel', device='cpu')
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2470, 0.2435, 0.2616]),
        ])

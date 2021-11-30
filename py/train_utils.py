"""
    Training Utilities
"""

import os
import torch
import shutil
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
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

def getCIFAR10Dataset(train, transform, batch_size):
    download = (len(os.listdir("./dataset/")) == 0)
    return DataLoader(
        CIFAR10("./dataset/", 
            train = train, download = download, transform = transform),
        batch_size = batch_size, shuffle = train,
    )

def CIFAR10Images(train, transform = None):
    download = (len(os.listdir("./dataset/")) == 0)
    return CIFAR10("./dataset/", train = train, download = download, transform = transform)


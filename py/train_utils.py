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

def getSummaryWriter(epochs:int, del_dir:bool):
    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    return SummaryWriter(log_dir = logdir + time_stamp)

def getCIFAR10Dataset(train, transform, batch_size):
    download = (len(os.listdir("./dataset/")) == 0)
    return DataLoader(
        CIFAR10("./dataset/", 
            train = train, download = download, transform = transform),
        batch_size = batch_size, shuffle = True,
    )



"""
    Training Utilities
"""

import os
import torch
import shutil
from torch.utils.data._utils import collate
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from datetime import datetime

from torchvision.datasets import CIFAR10
from torchvision import transforms

from timm.data import RandomResizedCropAndInterpolation
from timm.data.random_erasing import RandomErasing
from timm.data.mixup import FastCollateMixup, mixup_target
from timm.data.loader import fast_collate

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

def getCIFAR10Dataset(train, augment, batch_size, collate_fn = None):
    download = (len(os.listdir("../dataset/")) == 0)
    return DataLoader(
        CIFAR10("../dataset/", 
            train = train, download = download, transform = makeTransfrom(augment)), collate_fn = collate_fn,
        batch_size = batch_size, shuffle = train, num_workers = 8, pin_memory = True, persistent_workers = True, drop_last=train
    )

def CIFAR10Images(train, transform = None):
    download = (len(os.listdir("../dataset/")) == 0)
    return CIFAR10("../dataset/", train = train, download = download, transform = transform)

def oneHotAccCounter(pred:torch.FloatTensor, truth:torch.FloatTensor)->int:
    _, pred_max_pos = torch.max(pred, dim = 1)
    _, gt_max_pos = torch.max(truth, dim = 1)
    return torch.sum(pred_max_pos == gt_max_pos)

def accCounter(pred:torch.FloatTensor, truth:torch.FloatTensor)->int:
    _, max_pos = torch.max(pred, dim = 1)
    return torch.sum(max_pos == truth)

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

class FastCollate:
    def __init__(self, mix_up_configs:dict) -> None:
        self.collate_fn = FastCollateMixup(**mix_up_configs)
        self.label_smoothing = mix_up_configs['label_smoothing']
        self.num_classes = mix_up_configs['num_classes']

    def collate(self, X:torch.Tensor, y:torch.Tensor, collate_enabled:bool = True):
        if collate_enabled:
            np_xs = [x.numpy() for x in X]
            ys = [int(label) for label in y]
            batch = list(zip(np_xs, ys))
            output_x, output_y = self.collate_fn(batch)
            return output_x.float(), output_y
        else:
            return X, mixup_target(y, self.num_classes, 1.0, self.label_smoothing, device='cpu')
            
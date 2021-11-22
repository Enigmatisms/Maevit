#-*-coding:utf-8-*-
"""
    Sequential Pooling in CVT & CCT
    SeqPool is actually like some kind of "Attentional Pooling"
    @author Enigmatisms
"""

import torch
from torch import nn
from torch.nn import functional as F

class SeqPool(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.linear = nn.Linear(in_chan, 1)
        
    def foward(self, X:torch.Tensor):
        proba_x = F.softmax(torch.transpose(self.linear(X), -1, -2), dim = -1)
        return proba_x @ X
    
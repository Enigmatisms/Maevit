#-*-coding:utf-8-*-
"""
    Swin Transformer Layer implementation
"""

import torch
import einops
from torch import nn
from py.TEncoder import TransformerEncoderV2

class WinLayer(nn.Module):
    def __init__(self, M = 7, C = 256, img_size = 224, head_num = 4) -> None:
        super().__init__()
        self.M = M
        self.C = C
        self.img_size = img_size
        self.trans_layer = TransformerEncoderV2(C, C, 0.0, head_num)
        self.merge_lin = nn.Linear(C << 2, C << 1)

    def merge(self, X:torch.Tensor)->torch.Tensor:
        X = einops.rearrange(X, 'N C (m1 H) (m2 W) -> N (C m1 m2) H W', m1 = 2, m2 = 2)
        return self.merge_lin(X)

    # (N, C, H, W) -> (N, C, H, W)
    def forward(self, X:torch.Tensor)->torch.Tensor:
        bnum, _, size, _ = X.shape
        size //= self.M
        X:torch.Tensor = einops.rearrange(X, 'N C (m1 H) (m2 W) -> (N m1 m2) (H W) C', m1 = self.M, m2 = self.M)
        # now X is of shape (N * M * M, H/M * W/M, C), flattened in dim 0 and dim 1 for attention op
        X = self.trans_layer(X)
        # after attention op, tensor must be reshape back to the original shape
        return einops.rearrange(X, '(N m1 m2) (H W) C -> N C (m1 H) (m2 W)', N = bnum, m1 = self.M, m2 = self.M, H = size, W = size)

class SwinLayer(nn.Module):
    # roll op
    def __init__(self) -> None:
        super().__init__()

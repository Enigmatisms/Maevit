#-*-coding:utf-8-*-
"""
    Swin Transformer Layer implementation
"""

import torch
import einops
from torch import nn
from torch.nn import functional as F
from winMSA import WinMSA, SwinMSA
from timm.models import DropPath

#TODO: reverse shifting!
# 2 in here is the expansion ratio
def makeMLP(in_chan, mlp_dropout):
    return nn.Sequential(
        nn.Linear(in_chan, 2 * in_chan),
        nn.GELU(),
        nn.Dropout(mlp_dropout),
        nn.Linear(2 * in_chan, in_chan),
        nn.Dropout(mlp_dropout)
    )

class SwinTransformerLayer(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def __init__(self, M = 7, C = 256, img_size = 224, head_num = 4, mlp_dropout=0.1) -> None:
        super().__init__()
        self.M = M
        self.C = C
        self.img_size = img_size
        self.merge_lin = nn.Linear(C << 2, C << 1)
        self.drop_path = DropPath(0.1)
        self.pre_ln = nn.LayerNorm(C)
        self.post_ln = nn.LayerNorm(C)
        self.head_num = head_num
        self.mlp = makeMLP(256, mlp_dropout)
        self.win_size = img_size // M
        self.win_msa = WinMSA(img_size, img_size, self.win_size, C, head_num)
        self.swin_msa = SwinMSA(img_size, img_size, self.win_size, C, head_num)
        self.apply(self.init_weight)

    def merge(self, X:torch.Tensor)->torch.Tensor:
        X = einops.rearrange(X, 'N C (m1 H) (m2 W) -> N (C m1 m2) H W', m1 = 2, m2 = 2)
        return self.merge_lin(X)

    def layerForward(self, X:torch.Tensor, use_swin = False) -> torch.Tensor:
        tmp = self.pre_ln(X)
        tmp = self.swin_msa(tmp) if use_swin else self.win_msa(tmp)
        X = self.post_ln(X + self.drop_path(tmp))
        tmp2 = self.mlp(X)
        return X + self.drop_path(tmp2)

    # (N, C, H, W) -> (N, C, H, W)
    def forward(self, X:torch.Tensor)->torch.Tensor:
        bnum, _, size, _ = X.shape
        size //= self.M
        X:torch.Tensor = einops.rearrange(X, 'N C (m1 H) (m2 W) -> N (m1 m2) (H W) C', m1 = self.M, m2 = self.M)
        # now X is of shape (N,  M * M, H/M * W/M, C), flattened in dim 0 and dim 1 for attention op
        X = self.layerForward(X)
        # shifting, do not forget this
        X = torch.roll(X, shifts = (self.win_size // 2, self.win_size // 2), dims = (-1, -2))
        X = self.layerForward(X, use_swin = True)
        # after attention op, tensor must be reshape back to the original shape
        return einops.rearrange(X, '(N m1 m2) (H W) C -> N C (m1 H) (m2 W)', N = bnum, m1 = self.M, m2 = self.M, H = size, W = size)
        
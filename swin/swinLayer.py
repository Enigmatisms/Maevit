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
        nn.Linear(in_chan, 4 * in_chan),
        nn.GELU(),
        nn.Dropout(mlp_dropout),
        nn.Linear(4 * in_chan, in_chan),
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
    def __init__(self, M = 7, C = 96, img_size = 224, head_num = 4, mlp_dropout=0.1, patch_merge = False) -> None:
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
        self.merge_ln = None
        self.patch_merge = patch_merge
        if patch_merge == True:
            self.merge_ln = nn.LayerNorm(4 * C)

    def merge(self, X:torch.Tensor)->torch.Tensor:
        # TODO: all the rearrange must be experimented. Will this procedure cause trouble? This needs to be tested in experiment
        # TODO: record this in the notes
        X = einops.rearrange(X, 'N wn (H m1) (W m2) C -> N wn H W (m1 m2 C)', m1 = 2, m2 = 2)
        X = self.merge_ln(X)
        return self.merge_lin(X)

    def layerForward(self, X:torch.Tensor, use_swin = False) -> torch.Tensor:
        tmp = self.pre_ln(X)
        tmp = self.swin_msa(tmp) if use_swin else self.win_msa(tmp)
        X = self.post_ln(X + self.drop_path(tmp))
        tmp2 = self.mlp(X)
        return X + self.drop_path(tmp2)

    # (N, H, W, C) -> (N, H, W, C)
    def forward(self, X:torch.Tensor)->torch.Tensor:
        # patch partion is done in every layer
        X = self.layerForward(X)
        # shifting, do not forget this
        X = einops.rearrange(X, 'N wn (H W) C -> N wn H W C')
        X = torch.roll(X, shifts = (-(self.win_size >> 1), -(self.win_size >> 1)), dims = (-2, -3))
        X = einops.rearrange(X, 'N wn H W C -> N wn (H W) C')
        X = self.layerForward(X, use_swin = True)
        X = einops.rearrange(X, 'N wn (H W) C -> N wn H W C')
        # inverse shifting procedure
        X = torch.roll(X, shifts = (self.win_size >> 1, self.win_size >> 1), dims = (-1, -2))
        # after attention op, tensor must be reshape back to the original shape
        if self.patch_merge:
            X = self.merge(X)
        X = einops.rearrange(X, 'N wn H W C -> N wn (H W) C')
        return X

"""
    To my suprise, patching embedding generation in official implementation contains Conv2d... I thought this was conv-free
    Patch partition and embedding generation in one shot, in the paper, it is said that:
    'A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension (denoted as C)'
"""
class PatchEmbeddings(nn.Module):
    def __init__(self, patch_size = 4, M = 7, out_channels = 48, input_channel = 3, norm_layer = None) -> None:
        super().__init__()
        self.C = 256
        self.M = M                  # M is the number of window in each direction
        self.conv = nn.Conv2d(input_channel, out_channels, kernel_size = patch_size, stride = patch_size)
        if not norm_layer is None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.conv(X)
        if not self.norm is None:
            X = self.norm(X)
        X = X.permute(0, 2, 3, 1)
        return einops.rearrange(X, 'N (m1 H) (m2 W) C -> N (m1 m2) (H W) C', m1 = self.M, m2 = self.M)
        # output x is (N, (window_num ** 2), (img_size / patch_size / window_num)**2, C)

class SwinTransformer(nn.Module):
    def __init__(self, M = 7, C = 96, img_size = 224, head_num = 4, mlp_dropout = 0.1, emb_dropout = 0.1, init_patch_size = 4) -> None:
        super().__init__()
        self.M = M
        self.C = C
        self.img_size = img_size
        self.head_num = head_num
        current_img_size = img_size // init_patch_size
        self.patch_embed = PatchEmbeddings(init_patch_size, M, C, 3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.swin_layers = nn.ModuleList([
            SwinTransformerLayer(M, C, current_img_size, head_num, mlp_dropout), 
            SwinTransformerLayer(M, C, current_img_size, head_num, mlp_dropout, True)
        ])
        C <<= 1
        current_img_size >>= 1
        self.swin_layers.extend([
            SwinTransformerLayer(M, C, current_img_size, head_num, mlp_dropout),
            SwinTransformerLayer(M, C, current_img_size, head_num, mlp_dropout, True)
        ])
        C <<= 1
        current_img_size >>= 1
        self.swin_layers.extend([SwinTransformerLayer(M, C, current_img_size, head_num, mlp_dropout) for _ in range(5)])
        self.swin_layers.append(SwinTransformerLayer(M, C, current_img_size, head_num, mlp_dropout, True))
        C <<= 1 
        current_img_size >>= 1
        self.swin_layers.extend([SwinTransformerLayer(M, C, current_img_size, head_num, mlp_dropout) for _ in range(2)])
        self.classify = nn.Linear(C, 10)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = X.shape 
        X = self.patch_embed(X)
        X = self.emb_drop(X)
        X = self.swin_layers(X)
        channel_num = X.shape[-1]
        X = X.view(batch_size, -1, channel_num).transpose(-1, -2)
        return self.classify(self.avg_pool(X))
    # patch merging is implemented or not?
    
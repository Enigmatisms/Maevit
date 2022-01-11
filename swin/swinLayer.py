#-*-coding:utf-8-*-
"""
    Swin Transformer Layer implementation
"""

import torch
import einops
from torch import nn
from torch.nn import functional as F
from swin.winMSA import WinMSA, SwinMSA
from timm.models.layers.drop import DropPath

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
    def __init__(self, win_size = 7, emb_dim = 96, img_size = 224, head_num = 4, mlp_dropout=0.1, patch_merge = False) -> None:
        super().__init__()
        self.win_size = win_size
        self.emb_dim = emb_dim
        self.img_size = img_size
        self.win_num = img_size // win_size
        self.merge_lin = nn.Linear(emb_dim << 2, emb_dim << 1)
        self.drop_path = DropPath(0.1)
        self.pre_ln = nn.LayerNorm(emb_dim)
        self.post_ln = nn.LayerNorm(emb_dim)
        self.head_num = head_num
        self.mlp = makeMLP(emb_dim, mlp_dropout)
        self.win_msa = WinMSA(win_size, emb_dim, head_num)
        self.swin_msa = SwinMSA(img_size, win_size, emb_dim, head_num)
        self.merge_ln = None
        self.patch_merge = patch_merge
        if patch_merge == True:
            self.merge_ln = nn.LayerNorm(4 * emb_dim)

    def merge(self, X:torch.Tensor)->torch.Tensor:
        # TODO: all the rearrange must be experimented. Will this procedure cause trouble? This needs to be tested in experiment
        # TODO: record this in the notes
        X = einops.rearrange(X, 'N (winh winw) H W C -> N (winh H) (winw W) C', winh = self.win_num, winw = self.win_num)
        X = einops.rearrange(X, 'N (H m1) (W m2) C -> N H W (m1 m2 C)', m1 = 2, m2 = 2)
        # wn should be expanded back to (win_H, win_W)
        X = self.merge_lin(self.merge_ln(X))
        return einops.rearrange(X, 'N (m1 H) (m2 W) C -> N (m1 m2) (H W) C', H = self.win_size, W = self.win_size)

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
        X = einops.rearrange(X, 'N wn (H W) C -> N wn H W C', H = self.win_size)
        X = torch.roll(X, shifts = (-(self.win_size >> 1), -(self.win_size >> 1)), dims = (-2, -3))
        X = einops.rearrange(X, 'N wn H W C -> N wn (H W) C')
        X = self.layerForward(X, use_swin = True)
        X = einops.rearrange(X, 'N wn (H W) C -> N wn H W C', H = self.win_size)
        # inverse shifting procedure
        X = torch.roll(X, shifts = (self.win_size >> 1, self.win_size >> 1), dims = (-1, -2))
        # after attention op, tensor must be reshape back to the original shape
        if self.patch_merge:
            return self.merge(X)
        X = einops.rearrange(X, 'N wn H W C -> N wn (H W) C')
        return X

"""
    To my suprise, patching embedding generation in official implementation contains Conv2d... I thought this was conv-free
    Patch partition and embedding generation in one shot, in the paper, it is said that:
    'A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension'
"""
class PatchEmbeddings(nn.Module):
    def __init__(self, patch_size = 4, win_size = 7, out_channels = 48, input_channel = 3, norm_layer = None) -> None:
        super().__init__()
        self.win_size = win_size
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
        return einops.rearrange(X, 'N (m1 H) (m2 W) C -> N (m1 m2) (H W) C', H = self.win_size, W = self.win_size)
        # output x is (N, (window_num ** 2), (img_size / patch_size / window_num)**2, C)

class SwinTransformer(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            nn.init.kaiming_normal_(m)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def __init__(self, win_size = 7, emb_dim = 96, img_size = 224, head_num = (3, 6, 12, 24), mlp_dropout = 0.1, emb_dropout = 0.1, init_patch_size = 4) -> None:
        super().__init__()
        self.win_size = win_size
        self.emb_dim = emb_dim
        self.img_size = img_size
        self.head_num = head_num
        current_img_size = img_size // init_patch_size
        self.patch_embed = PatchEmbeddings(init_patch_size, win_size, emb_dim, 3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.emb_drop = nn.Dropout(emb_dropout)
        # input image_size / 4, output_imgae_size / 4
        self.swin_layers = nn.ModuleList([])
        num_layers = (2, 2, 4, 2)
        for i in range(4):
            num_layer = num_layers[i]
            num_head = head_num[i]
            for _ in range(num_layer - 1):
                self.swin_layers.append(SwinTransformerLayer(win_size, emb_dim, current_img_size, num_head, mlp_dropout))
            final_layer_merge_patch = True if i < 3 else False
            self.swin_layers.append(SwinTransformerLayer(win_size, emb_dim, current_img_size, num_head, mlp_dropout, final_layer_merge_patch))
            if i < 3:
                emb_dim <<= 1
                current_img_size >>= 1
        self.classify = nn.Linear(emb_dim, 10)
        self.apply(self.init_weight)

    def loadFromFile(self, load_path:str):
        save = torch.load(load_path)   
        save_model = save['model']                  
        model_dict = self.state_dict()
        state_dict = {k:v for k, v in save_model.items()}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict) 
        print("Swin Transformer Model loaded from '%s'"%(load_path))

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = X.shape 
        X = self.patch_embed(X)
        X = self.emb_drop(X)
        for _, layer in enumerate(self.swin_layers):
            X = layer(X)
        channel_num = X.shape[-1]
        X = X.view(batch_size, -1, channel_num).transpose(-1, -2)
        X = self.avg_pool(X).transpose(-1, -2)
        return self.classify(X).squeeze(dim = 1)
    
if __name__ == "__main__":
    stm = SwinTransformer(7, 96, 224).cuda()
    # for blc in stm.parameters():
    #     print(blc.device)
    test_image = torch.normal(0, 1, (4, 3, 224, 224)).cuda()
    result = stm.forward(test_image)
    
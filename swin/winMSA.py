#-*-coding:utf-8-*-
"""
    Swin Transformer Window-based multihead attention block
    WinMSA is implemented as a base class inheriting from nn.Module
    SwinMSA is the child class of WinMSA
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

# TODO: I have a problem: for swin layer (shift is applied), should relative positional embeddings be shifted?
# The answer seems to be a 'No', for I think PE tends to learn the information after shifting
class WinMSA(nn.Module):
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            nn.init.kaiming_normal_(m)
    def __init__(self, width, height, win_size, emb_dim = 96, head_num = 4) -> None:
        super().__init__()
        self.h = height
        self.w = width
        self.win_size = win_size
        self.emb_dim = emb_dim
        self.win_h = height // win_size
        self.win_w = width // win_size
        self.s = win_size // 2
        self.att_size = win_size ** 2       # this is actually sequence length, too
        self.half_att_size = self.att_size // 2
        self.emb_dim_h_k = emb_dim // head_num
        self.normalize_coeff = self.emb_dim_h_k ** (-0.5)
        self.head_num = head_num

        # positional embeddings
        self.pe = nn.Parameter(torch.zeros(2 * self.att_size - 1, emb_dim), requires_grad = True)
        self.relp_indices = self.getRelpeIndices()

        self.qkv_attn = nn.Linear(emb_dim, emb_dim * 3, bias = False)
        self.proj_o = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(0.1)
        self.attn_drop = nn.Dropout(0.1)
        self.apply(self.init_weight)

        # relative position embeddings
    def getRelpeIndices(self):
        all_indices = torch.arange(-self.att_size + 1, self.att_size).repeat(self.att_size, 1)
        totoal_relp = 2 * self.att_size - 1
        all_indices = torch.hstack((all_indices, torch.zeros(self.att_size, 1)))
        all_indices = torch.concat((all_indices.view(-1), torch.zeros(self.att_size - 1))).view(-1, totoal_relp)
        return all_indices[:self.att_size, -self.att_size:]

    def attention(self, X:torch.Tensor, mask:None) -> torch.Tensor:
        batch_size, win_num, seq_len, _ = X.shape
        # input (b, win_num, seq_len, embedding_dim) while output of qkv attn is (b, seq_len, 3 * embedding_dim)
        # output shape: (3, batch_size, win_num, head_num, seq_len, emb_dim / head_num)
        qkvs:torch.Tensor = self.qkv_attn(X).view(batch_size, win_num, seq_len, 3, self.head_num, self.emb_dim_h_k).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkvs[0], qkvs[1], qkvs[2]
        s = q @ self.pe[None, None, None, :, :].transpose(-2, -1)         # query directly mult pe (batch, win_num, head, seq, 2 * seq - 1)
        # Will this thing have big memory or speed overhead?
        rpe = torch.gather(s, -1, self.relp_indices.repeat(batch_size, win_num, self.head_num, 1, 1))        # self.relp_indices is already a 2d tensor
        # q @ k.T : shape (batch_size, win_num, head_num, seq, seq), att_mask added according to different window position
        attn = q @ k.transpose(-1, -2) * self.normalize_coeff + rpe
        if not mask is None:
            attn = attn + mask[None, :, None, :, :]
        proba:torch.Tensor = F.softmax(attn, dim = -1)
        proba = self.attn_drop(proba)
        # proba: batch_size, window, head_num, seq_len, seq_len -> output (batch_size, win_num, head_num, seq_len, emb_dim/ head_num)
        proj_o = self.proj_o((proba @ v).transpose(2, 3).reshape(batch_size, win_num, seq_len, -1))
        return self.proj_drop(proj_o)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.attention(X, None)

# default invariant shift: win_size / 2
class SwinMSA(WinMSA):
    def __init__(self, width, height, win_size, emb_dim = 96, head_num = 4) -> None:
        WinMSA.__init__(width, height, win_size, emb_dim, head_num)
        self.att_mask = self.getAttentionMask()

    """
        shape of input tensor (batch_num, window_num, seq_length(number of embeddings in a window), emb_dim)
        output remains the same shape. Notice that, shift is already done outside of this class
    """
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.attention(X, self.att_mask)

    # somehow I think this is not so elegant, yet attention mask is tested
    def getAttentionMask(self) -> torch.Tensor:
        mask = torch.zeros(self.win_h, self.win_w, self.att_size, self.att_size)
        # process the rightmost column
        rightmost = torch.ones(self.win_size, self.win_size)
        rightmost[:, self.s:] = -torch.ones(self.win_size, self.s)
        rightmost = rightmost.view(-1, 1)
        raw_mask = rightmost @ rightmost.transpose(0, 1)
        right_mask = torch.zeros_like(raw_mask)
        right_mask = right_mask.masked_fill(raw_mask < 0, -100)
        # process the bottom row
        bottom = torch.ones(self.win_size, self.win_size)
        bottom[self.s:, :] = -torch.ones(self.s, self.win_size)
        bottom = bottom.view(-1, 1)
        raw_mask = bottom @ bottom.transpose(0, 1)
        bottom_mask = torch.zeros_like(bottom)
        bottom_mask = bottom_mask.masked_fill(raw_mask < 0, -100)
        # bottom-right corner
        mask[:-1, -1, :, :] = right_mask
        mask[-1, :-1, :, :] = bottom_mask
        mask[-1, -1, :, :] = -100 * torch.ones(self.att_size, self.att_size)
        mask[-1, -1, :self.half_att_size, :self.half_att_size] = right_mask[:self.half_att_size, :self.half_att_size]
        mask[-1, -1, self.half_att_size:, self.half_att_size:] = right_mask[:self.half_att_size, :self.half_att_size]
        return mask.view(-1, self.att_size, self.att_size)

class SwinTransformerLayer(nn.Module):
    def __init__(self, width, height, win_size, emb_dim) -> None:
        super().__init__()

if __name__ == "__main__":
    sw = SwinMSA(12, 12, 6, 1)
    print(sw.getRelpeIndices())
    mask = sw.getAttentionMask()
    for i in range(mask.shape[0]):
        plt.figure(i)
        plt.imshow(mask[i])
    plt.show()

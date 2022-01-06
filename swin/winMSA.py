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

# TODO: 1 relative positional <bias>! Not embeddings, be careful here

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
    def __init__(self, img_size, win_num = 7, emb_dim = 96, head_num = 4, device = 'cpu') -> None:
        super().__init__()
        self.win_size = img_size // win_num
        self.emb_dim = emb_dim
        self.s = self.win_size // 2
        self.att_size = self.win_size ** 2       # this is actually sequence length, too
        self.half_att_size = self.att_size // 2
        self.emb_dim_h_k = emb_dim // head_num
        self.normalize_coeff = self.emb_dim_h_k ** (-0.5)
        self.head_num = head_num
        self.device = device

        # positional embeddings
        self.positional_bias = nn.Parameter(torch.zeros(head_num, 2 * self.att_size - 1, 2 * self.att_size - 1), requires_grad = True)
        # using register buffer, this tensor will be moved to cuda device if .cuda() is called, also it is stored in state_dict
        self.register_buffer('relp_indices', WinMSA.getIndex(self.win_size))            

        self.qkv_attn = nn.Linear(emb_dim, emb_dim * 3, bias = False)
        self.proj_o = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(0.1)
        self.attn_drop = nn.Dropout(0.1)
        self.apply(self.init_weight)

    @staticmethod
    def getIndex(win_size:int) -> torch.LongTensor:
        ys, xs = torch.meshgrid(torch.arange(win_size), torch.arange(win_size), indexing = 'ij')
        coords = torch.cat((ys.unsqueeze(dim = -1), xs.unsqueeze(dim = -1)), dim = -1).view(-1, 2)
        diff = coords[None, :, :] - coords[:, None, :]          # interesting broadcasting, needs notes
        diff += win_size - 1
        index = diff[:, :, 0] * (2 * win_size - 1) + diff[:, :, 1]
        return index

    def attention(self, X:torch.Tensor, mask:None) -> torch.Tensor:
        batch_size, win_num, seq_len, _ = X.shape
        # input (b, win_num, seq_len, embedding_dim) while output of qkv attn is (b, seq_len, 3 * embedding_dim)
        # output (3, batch, win, head, seq, emb_dim/head_num) 0  1      2       3       4               5
        qkvs:torch.Tensor = self.qkv_attn(X).view(batch_size, win_num, seq_len, 3, self.head_num, self.emb_dim_h_k).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkvs[0], qkvs[1], qkvs[2]
        # q @ k.T : shape (batch_size, win_num, head_num, seq, seq), att_mask added according to different window position  
        attn = q @ k.transpose(-1, -2) * self.normalize_coeff + self.positional_bias.view(self.head_num, -1)[self.relp_indices.view(self.head_num, -1)].view(self.head_num, seq_len, seq_len)
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
    def __init__(self, img_size, win_num = 7, emb_dim = 96, head_num = 4) -> None:
        WinMSA.__init__(img_size, win_num, emb_dim, head_num)
        self.att_mask = self.getAttentionMask()
        self.win_h = img_size // self.win_size
        self.win_w = img_size // self.win_size

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
        return mask.view(-1, self.att_size, self.att_size).to(self.device)

if __name__ == "__main__":
    sw = SwinMSA(12, 12, 6, 1)
    print(sw.getRelpeIndices())
    mask = sw.getAttentionMask()
    for i in range(mask.shape[0]):
        plt.figure(i)
        plt.imshow(mask[i])
    plt.show()

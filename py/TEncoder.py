#-*-coding:utf-8-*-
"""
    Vision Transformer (Lite), Vision Transformer Base has 12 layers and over
    80 Million parameters, which is almost 4 times the size of ResNet-50
    I want to shrink the network down and implement a light-weight [Lite] version
    There is actually ViT-Lite:
    Escaping the Big Data Paradigm with Compact Transformers
"""

import torch
from torch import nn
from math import sqrt
from torch.nn import functional as F

# 2 in here is the expansion ratio
def makeMLP(in_chan):
    return nn.Sequential(
        nn.Linear(in_chan, 2 * in_chan),
        nn.LayerNorm(in_chan),
        nn.GELU(),
        nn.Linear(2 * in_chan, in_chan),
        nn.LayerNorm(in_chan),
        nn.GELU()
    )

"""
    Transformer Encoder in ViT
"""
class TransformerEncoder(nn.Module):
    def __init__(self, dim_k, dim_v, mlp_chan, head_num = 2):
        super().__init__()
        # 我感觉自己又不是特别能记得Transformer的工作原理了
        # 原理上来说，有多少个Head就要进行多少次投影（Q,K,V都是）
        # 每个输入的X都是（batch，seq，embedding_dim），也就是说，一个patch只有一个一维的高维特征
        # 多头注意力需要将其变为(batch, seq, emd / head) 这样的形式
        # 注意Q, K 与 V不一定是同一维度，V是输出，Q,K均是与输入有关的
        assert(dim_k % head_num == 0)
        assert(dim_v % head_num == 0)
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_h_k = dim_k // head_num
        self.dim_h_v = dim_v // head_num
        self.normalize_coeff = sqrt(self.dim_h_k)
        self.head_num = head_num
        self.proj_q = nn.ModuleList([nn.Linear(dim_k, self.dim_h_k) for _ in range(head_num)])
        self.proj_k = nn.ModuleList([nn.Linear(dim_k, self.dim_h_k) for _ in range(head_num)])
        self.proj_v = nn.ModuleList([nn.Linear(dim_v, self.dim_h_v) for _ in range(head_num)])
        self.pre_ln = nn.LayerNorm(dim_k)
        self.post_ln = nn.LayerNorm(dim_v)
        self.mlp = makeMLP(mlp_chan, mlp_chan)
        
    def attention(self, X:torch.Tensor):
        result = []
        for i in range(self.head_num):
            Xq:torch.Tensor = self.proj_q[i](X)
            Xk:torch.Tensor = self.proj_k[i](X)
            proba_mat = Xq @ torch.transpose(Xk, -1, -2) / self.normalize_coeff
            proba = F.softmax(proba_mat, dim = -1)
            result.append(proba @ self.proj_v[i](X))
        return torch.cat(result, dim = -1)
    
    def forward(self, X, norm = False):
        tmp = self.pre_ln(X)
        tmp = self.attention(tmp)
        X = X + tmp
        tmp2 = self.post_ln(X)
        tmp2 = self.mlp(tmp2)
        return X + tmp2
        
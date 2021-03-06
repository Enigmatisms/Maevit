#-*-coding:utf-8-*-
"""
    Compact Convolution Transformer
    CCT 7/3X1
    @author Enigmatisms
"""

import torch
from torch import nn
from py.TEncoder import TransformerEncoder, TransformerEncoderV2
from py.SeqPool import SeqPool

"""
    - trans_num: number of transformer encoder
"""
class CCT(nn.Module):
    @staticmethod
    def makeConvBlock(in_chan, out_chan, ksize = 3, pool_k = 3, pool_stride = 2, pool_pad = 1):
        pad = (ksize // 2)
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, ksize, 1, pad, bias = False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = pool_k, stride = pool_stride, padding = pool_pad)
        )

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        
    def loadFromFile(self, load_path:str):
        save = torch.load(load_path)   
        save_model = save['model']                  
        model_dict = self.state_dict()
        state_dict = {k:v for k, v in save_model.items()}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict) 
        print("CCT Model loaded from '%s'"%(load_path))
    
    def __init__(self, trans_num = 7, ksize = 3, emb_dim = 256, conv_num = 1):
        super().__init__()
        self.conv = nn.ModuleList(
            [CCT.makeConvBlock(3, emb_dim, ksize)]
        )
        self.conv_num = conv_num
        if conv_num > 1:
            for _ in range(conv_num - 1):
                self.conv.append(CCT.makeConvBlock(emb_dim, emb_dim, ksize))
        self.transformers = nn.Sequential(
            *[TransformerEncoderV2(emb_dim, emb_dim, mlp_dropout=0.0, head_num = 4) for _ in range(trans_num)]
        )
        self.trans_num = trans_num
        self.sp = SeqPool(emb_dim)
        self.pre_dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(emb_dim)
        self.classify = nn.Linear(emb_dim, 10)
        self.flatten = nn.Flatten(2, 3)
        for i in range(self.conv_num):
            self.conv[i].apply(self.init_weight)
        self.seq_len = self.tokenize(torch.zeros(1, 3, 32, 32)).shape[1]
        self.position_emb = nn.Parameter(torch.zeros(1, self.seq_len, emb_dim), requires_grad = True)
        nn.init.trunc_normal_(self.position_emb, std = 0.2)

    def tokenize(self, X:torch.Tensor)->torch.Tensor:
        for i in range(self.conv_num):
            X = self.conv[i](X)
        # reshape convolution outputs: (n, emb_dim, w0, h0)
        return self.flatten(X).transpose(-1, -2)
        
    def forward(self, X:torch.Tensor):
        X = self.tokenize(X)
        # train with positional embedding
        X = X + self.position_emb
        X = self.pre_dropout(X)
        X = self.transformers(X)
        # The problem is: LayerNorm? When to norm? Whether to norm???
        z = self.norm(X)
        z = self.sp(z).squeeze(dim = 1)
        return self.classify(z)
        
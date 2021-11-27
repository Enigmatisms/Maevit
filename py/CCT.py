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
            *[TransformerEncoderV2(emb_dim, emb_dim, 0.0, head_num = 4) for _ in range(trans_num)]
        )
        self.trans_num = trans_num
        self.sp = SeqPool(emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.classify = nn.Linear(emb_dim, 10)
<<<<<<< HEAD
        self.flatten = nn.Flatten(2, 3)
        for i in range(self.conv_num):
            self.conv[i].apply(self.init_weight)
=======
>>>>>>> 8f8d0588e20e9b3d3e3e134c248790019a050532
        
    def forward(self, X:torch.Tensor):
        for i in range(self.conv_num):
            X = self.conv[i](X)
        # reshape convolution outputs: (n, emb_dim, w0, h0)
<<<<<<< HEAD
        X = self.flatten(X).transpose(-1, -2)
        X = self.transformers(X)
        # The problem is: LayerNorm? When to norm? Whether to norm？
        z = self.norm(X)
        z = self.sp(z).squeeze(dim = 1)
        return self.classify(z)
=======
        batch_size, emb_dim, _, _ = X.shape
        X = X.view(batch_size, -1, emb_dim)     # (n, seq_length=w0 * h0, emb_dim)
        X = self.transformers(X)
        z = self.sp(X)
        z = self.norm(z)
        return self.classify(z).squeeze(dim = 1)
>>>>>>> 8f8d0588e20e9b3d3e3e134c248790019a050532
        
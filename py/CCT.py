#-*-coding:utf-8-*-
"""
    Compact Convolution Transformer
    CCT 7/3X1
    @author Enigmatisms
"""

import torch
from torch import nn
from py.TEncoder import TransformerEncoder
from py.SeqPool import SeqPool

"""
    - trans_num: number of transformer encoder
"""
class CCT(nn.Module):
    @staticmethod
    def makeConvBlock(in_chan, out_chan, ksize = 3):
        pad = (ksize // 2)
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, ksize, 1, pad),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
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
            *[TransformerEncoder(emb_dim, emb_dim, emb_dim, head_num = 4) for _ in range(trans_num)]
        )
        self.trans_num = trans_num
        self.sp = SeqPool(emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.classify = nn.Linear(emb_dim, 10)
        
    def forward(self, X:torch.Tensor):
        for i in range(self.conv_num):
            X = self.conv[i](X)
        # reshape convolution outputs: (n, emb_dim, w0, h0)
        batch_size, emb_dim, _, _ = X.shape
        X = X.view(batch_size, -1, emb_dim)     # (n, seq_length=w0 * h0, emb_dim)
        X = self.transformers(X)
        z = self.sp(X)
        z = self.norm(z)
        return self.classify(z).squeeze(dim = 1)
        
#-*-coding:utf-8-*-
"""
    Vision transformer-Lite
    Using Embedded patches, CLS token and position encoding
    @author Enigmatisms
"""

import torch
from torch import nn
from torch.nn import parameter
from TEncoder import TransformerEncoder

class VitLite(nn.Module):
    """### Vision transformer-Lite

    Args:
        - token_dim [int]: [dimension of token (patch C * W * H)]
        - emb_dim [int]: [dimension of transformer embedding]
        - seq_len [int]: [sequence length (number of image patches)]
        - batch_size [int]: [Obviously]
        - patch_size: [width and (height which is equal to width) of a patch]
        - head_num [int]: [Number of heads in transformer (default 4)]
        - trans_num [int]: [Number of transformer layers (default 6)]
    """
    def __init__(self, token_dim, emb_dim, seq_len, batch_size, patch_size, head_num = 4, trans_num = 6):
        super().__init__()
        # 需要的是：positional embedding （需要知道seq length 也就是patch个数）
        # embedding 维度
        # 输入token的维度
        self.cls_token = parameter.Parameter(torch.normal(0, 1, (batch_size, 1, emb_dim)), requires_grad = True)
        
        # position embeddings are additive
        self.pos_embds = parameter.Parameter(
            torch.normal(0, 1, (batch_size, seq_len, emb_dim)), requires_grad = True
        )
        
        # In terms of making patches: (n, c, h, w) -> (n, patch_num, c, h, w) -> (n, seq_len, c * h * w)
        self.transformers = nn.Sequential(
            *[TransformerEncoder(emb_dim, emb_dim, emb_dim, head_num) for _ in range(trans_num)]
        ) 
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.token_dim = token_dim
        self.trans_num = trans_num
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.tokenize = nn.Linear(token_dim, emb_dim)
        
        # typical ViT has only one layer during fine-tuning time
        # ViT-Lite ain't no ordinary ViT XD, by default, I use CIFAR-10 which leads to 10 classes
        self.output = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(True),
            nn.Linear(emb_dim // 2, 10),
            nn.Sigmoid()
        )
        
    def loadFromFile(self, load_path:str):
        save = torch.load(load_path)   
        save_model = save['model']                  
        model_dict = self.state_dict()
        state_dict = {k:v for k, v in save_model.items()}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict) 
        print("ViT-Lite Model loaded from '%s'"%(load_path))
        
    def makePatch(self, images:torch.Tensor)->torch.Tensor:
        patches = torch.zeros(self.batch_size, self.seq_len, self.token_dim)
        _, _, h, w = images.shape
        row_pnum = h // self.patch_size
        col_pnum = w // self.patch_size
        for i in range (self.seq_len):
            id_r = i // row_pnum
            id_c = i % col_pnum
            patches[:, i, :] = images[:, :, id_r * self.patch_size : (id_r + 1) * self.patch_size,
                    id_c * self.patch_size, (id_c + 1) * self.patch_size].view(-1, self.token_dim)
        return patches
    
    def forward(self, X:torch.Tensor)->torch.Tensor:
        patches = self.makePatch(X)
        image_embs = self.tokenize(patches)
        embeddings = torch.cat((self.cls_token, image_embs), dim = 1)

        # output is (n, seq_len + 1, emb_dims)
        z = self.transformers(embeddings)
        cls_token = z[:, 0, :]
        return self.output(cls_token)
        
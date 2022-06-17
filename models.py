import os
from turtle import forward 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import math 


## SCALED DOT PRODUCT ATTENTION 

def scaled_dot_product_attention(Q,K,V, mask=None):
    d_k = Q.size()[-1]   #size Q  = T x Dk size K = T x Dk, size V  = Tx Dv 
    attention_logits  =  torch.matmul(Q, K.transpose(-2,-1))
    attention_logits = attention_logits/(math.sqrt(d_k))
    if mask is not None: 
        attention_logits = attention_logits.masked_fill(mask == 0, -9e15) 
    attention = F.softmax(attention_logits, dim = -1)
    values = torch.matmul(attention, V)
    return values, attention
 

 ## MULTIHEAD ATTENTION MODULE 

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim , num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim 
        self.input_dim = input_dim can
        self._reset_parameters()

    
    def _reset_parameters(self):
            # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask = None, return_attention = False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x) 

        # need to separate the qkv 
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0,2,1,3)  # [Batch, head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Calculate attention
        values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        else:
            return o


## TRANSFORMER ENCODER 

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers =  nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x , mask =None):
        for l in self.layers:
            x = l(x, mask = mask)
        return x 

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask = mask,return_attention = True )
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

## ENCODER BLOCK
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, droput = 0.0 ):
        super().__init__()

        # Attention Layers 
        self.self_attention = MultiHeadAttention(input_dim, input_dim, num_heads)

        # MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Droput(droput),
            nn.ReLU(),
            nn.Linear(dim_feedforward, input_dim)
            
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(droput) 

    def forward(self, x, mask = None):
        attention = self.self_attention(x, mask=mask)
        x = x + self.dropout(attention) 
        x = self.norm1(x) 
        linear_out = self.linear_net(x) 
        x = x+ self.dropout(linear_out)
        x = self.norm2(x)

        return x 

## POSITIONAL ENCODINGS 

class PositionalEncodings(nn.Module):
    def __init__(self, d_model, max_len = 5000 ):
        super().__init__()
        """
        d_model = Hidden dimebnsionality of the model 
        max_len = max len fo the sequence 
        """

        pe = torch.zeros(max_len, d_model)  # intilaize the pe with zeros 
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


## LEARNING RATE WARMUPs 


class LearningRateWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup 
        self.max_num_iters = max_iters
        super().__init__(optimizer)
    
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

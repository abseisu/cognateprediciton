import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Optional

# from model.decoder import TransformerBlock


class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.ln1 = nn.Linear(d_model, d_ff)
        self.ln2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ln2(x)
        return x


# Monolingual Encoder (tokens -> latent space)

class MonolingualEncoder(nn.Module):
    def __init__(self,
                 seq_len: int,
                 d_model: int,
                 n_attn_heads: int,
                 n_attn_layers: int,
                 dense_layers: List[int],
                 dropout: float
                 ):
        super(MonolingualEncoder, self).__init__()

        ff_dims = [seq_len * d_model] + dense_layers

        self.mhs_layers = nn.ModuleList()
        self.mhs_layers.extend([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                attn_pdrop=attn_pdrop,
                dropout=dropout,
                d_ff=d_ff,
                eps=eps)
            for i in range(num_layers)])
        
        self.dense_layers = nn.ModuleList()
        self.dense_layers.extend([FeedForward(ff_dims[i], ff_dims[i+1], dropout) for i in range(len(ff_dims - 1))])
    
    def forward(self, x):
        x = self.mhs_layers(x)
        x = self.dense_layers(x)
        return x

        



# Monolingual Decoder

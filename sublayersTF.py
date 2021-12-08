import pickle
from typing import List, Tuple, Dict

import numpy as np

import math
import copy

import torch
from torch import nn, optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.distributions.beta import Beta

PARENT_DIR = '/content/gdrive/MyDrive/CS287_Research_Project/Jennas_Code/' # for google colab. adjust accordingly
import sys
sys.path.append(PARENT_DIR)

### Layer Normalization###
class LayerNorm(nn.Module):
  def __init__(self, d_model, eps = 1e-6):
    super().__init__()

    self.size = d_model

    self.alpha = nn.Parameter(torch.ones(self.size))
    self.bias = nn.Parameter(torch.zeros(self.size))
    
    self.eps = eps
  
  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

### Residual Connections and Layer Normalization ###
class SublayerConnection(nn.Module):
  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    # residual connection
    return x + self.dropout(sublayer(self.norm(x)))

### Feed Forward Neural Net ###
class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff=2048, dropout = 0.1):
    super().__init__() 

    # Set d_ff as a default to 2048 bc Vaswani paper
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)
  
  def forward(self, x):
    x = self.dropout(F.relu(self.linear_1(x)))
    x = self.linear_2(x)
    return x

### Self-Attention Mechanism ####
def attention(query, key, value, mask=None, dropout=None):
  #Scaled dot product attention
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e9)
  p_attn = F.softmax(scores, dim = -1)
  if dropout is not None:
      p_attn = dropout(p_attn)
  return torch.matmul(p_attn, value), p_attn

### Multiple Self-Attention Heads ###
class MultiHeadedAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    super(MultiHeadedAttention, self).__init__()
    self.d_k = math.floor(d_model/h) # = d_v
    self.h = h
    self.linears = clones(nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = nn.Dropout(p=dropout)
      
  def forward(self, query, key, value, mask=None):
    if mask is not None:
      mask = mask.unsqueeze(1)
    nbatches = query.size(0)
      
    query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
      for l, x in zip(self.linears, (query, key, value))]

    x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

    x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
    return self.linears[-1](x)

### Used to Scale Model for Differing Encoder/Decoder Layers and Attention Heads ###
def clones(module, N):
    # For encoder/decoder stacks
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

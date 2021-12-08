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

from sublayersTF import *

### Encoder Layer ###
### Each Encoder Layer passinges the embeddings through the attention heads,
### residual connection and layer normalization, and the FFNN
class EncoderLayer(nn.Module):
  def __init__(self, size, self_attn, feed_forward, dropout=0.1):
      super(EncoderLayer, self).__init__()
      self.self_attn = self_attn
      self.feed_forward = feed_forward
      self.sublayer = clones(SublayerConnection(size, dropout), 2)
      self.size = size

  def forward(self, x, mask):
    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
    return self.sublayer[1](x, self.feed_forward)

### Decoder Layer ###
### Each Decoder Layer passinges the embeddings through the attention heads,
### residual connection and layer normalization, and the FFNN
class DecoderLayer(nn.Module):
  def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
    super(DecoderLayer, self).__init__()
    self.size = size
    self.self_attn = self_attn
    self.src_attn = src_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SublayerConnection(size, dropout), 3)

  def forward(self, x, memory, src_mask, tgt_mask):
    # Uses masking
    m = memory
    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
    x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
    return self.sublayer[2](x, self.feed_forward)
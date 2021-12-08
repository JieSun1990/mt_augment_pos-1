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

### Decoders-specific mask ###
### Generates mask for future words in the decoder
def future_mask(size):
  # Upper triangular masking future words
  attn_shape = (1, size, size)
  future_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  return torch.from_numpy(future_mask) == 0

### Batch object
### Creates object from source and target that holds right-shifted target values and source and target masks
class Batch:
    def __init__(self, src, trg=None, pad=0):
      self.src = src
      self.src_mask = (src != pad).unsqueeze(-2)
      if trg is not None:
        self.trg = trg[:, :-1]
        self.trg_y = trg[:, 1:]
        self.trg_mask = self.make_full_mask(self.trg, pad)
        self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_full_mask(tgt, pad):
      tgt_mask = (tgt != pad).unsqueeze(-2)
      tgt_mask = tgt_mask & future_mask(tgt.size(-1)).type_as(tgt_mask.data)
      return tgt_mask
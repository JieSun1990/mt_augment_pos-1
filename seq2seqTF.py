import pickle
from typing import List, Tuple, Dict

import numpy as np

import math
import copy
import random

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
from layersTF import *
from stacksTF import *
from encoderTF import *
from decoderTF import *

### Seq2Seq Transformer Model ###
### Contains both the custom Encoder and Decoder
### Takes in all relevant augmentation parameters
class Seq2SeqTF(nn.Module):
  def __init__(self,
               src_vocab_size, 
               tgt_vocab_size,
               device,
               padding_idx,
               N=6, 
               d_model=512, 
               d_ff=2048, 
               h=8, 
               dropout=0.1,
               augmentation_type = None,       # one of "swap", "drop", "blank", "smooth", "lmsample"
               gamma = None,                   # probability that single token is augmented
               k = None,                       # window size for "swap" method
               unk_idx = None,                 # placeholder for "blank" method
               unigram_freq = None,            # to sample from with "smooth" method
               language_model = None,
               source_pos_idx = None): ### Add input_pos IDX ###
    super(Seq2SeqTF, self).__init__()

    self.augmentation_type = augmentation_type
    c = copy.deepcopy
    self_attn = MultiHeadedAttention(h, d_model)
    ff = FeedForward(d_model, d_ff, dropout)

    self.encoder = FullEncoder(vocab_size = src_vocab_size,
                              d_model = d_model,
                              self_attn = c(self_attn),
                              feed_forward = c(ff),
                              dropout = dropout,
                              N = N,
                              padding_idx = padding_idx,
                              device = device,
                              augmentation_type = augmentation_type,
                              gamma = gamma,
                              k = k,                       # window size for "swap" method
                              unk_idx = unk_idx,                 # placeholder for "blank" method
                              unigram_freq = unigram_freq,            # to sample from with "smooth" method
                              language_model = language_model,
                              source_pos_idx = source_pos_idx)

    self.decoder = FullDecoder(vocab_size = tgt_vocab_size,
                              d_model = d_model,
                              self_attn = c(self_attn),
                              feed_forward = c(ff),
                              dropout = dropout,
                              N = N,
                              padding_idx = padding_idx)

    self.proj = nn.Linear(d_model, tgt_vocab_size)

    self.zero = torch.Tensor([0]).to(device)

  def forward(self, 
              input_seqs, #src
              output_seqs, #output_seqs
              src_mask, 
              tgt_mask,
              input_pos=None, 
              generate=False):
    
    memory = self.encode(input_seqs, src_mask, input_pos = input_pos,generate=generate)
    output = self.decode(output_seqs, memory, src_mask, tgt_mask, generate=generate)
    return(output)

  def encode(self, src, src_mask, input_pos = None,generate = False):
    return(self.encoder(src, src_mask,input_pos=input_pos,generate=generate))

  def decode(self, tgt, memory, src_mask, tgt_mask, generate = False):
    return(self.decoder(tgt,memory,src_mask,tgt_mask,generate=generate))
  
  def generator(self, x):
    return F.log_softmax(self.proj(x), dim=-1)



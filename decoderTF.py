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
from layersTF import *
from stacksTF import *
from embeddingTF import *

### Full Decoder ###
### Takes in all relevant augmentation parameters for output- and embeddings-level augmentations
### Still under construction for seqmix implementations
class FullDecoder(nn.Module):
  def __init__(
        self, 

        # regular encoder parameters
        vocab_size: int,				# Source vocabulary size
        d_model: int,					# Hidden size of model
        self_attn,						# Multiple attention head structure
        feed_forward,					# Feed-forward neural net
        dropout,						# Dropout parameter
        N,								# Number of layers in Encoder stack
        padding_idx, 					# Index of padding token
        device = "cuda",				# Device to do computation on

        # augmentation encoder parameters    
        augmentation_type: str = None,  # one of the decoder-level augmentations (seqmix, when implemented)
        lambda_ = None                 # seqmix lambda parameter
    ):
    super(FullDecoder, self).__init__()
    self.input_size = vocab_size
    self.hidden_size = d_model
    self.padding_idx = padding_idx
    self.device = device
    self.augmentation_type = augmentation_type
    self.lambda_ = lambda_

    # Embedder
    self.embedding = Embedder(vocab_size, d_model)

    # Positional Encoder
    self.positional_encoder = PositionalEncoding(d_model)

    # Decoder Layers
    decoder_layer = DecoderLayer(d_model, self_attn, self_attn, feed_forward, dropout)

    # Encoder
    self.decoder = Decoder(decoder_layer,N)

  def forward(self,
              # regular foward args
              input_seqs, #tgt
              memory,
              src_mask,
              tgt_mask,
              # augmentation forward args
              mix_seqs = None,
              lambda_ = None,
              generate = False, 
              input_pos = None, 
              mix_pos = None): 
      
    # Get embeddings
    embedded_input_seqs = self.embedding(input_seqs)

    # Augmentation
    # ************
    # ***INSERT***
    # ************

    # Positional encoding
    embedded_input_seqs = self.positional_encoder(embedded_input_seqs)

    # Encoding
    decoded_input_seqs = self.decoder(embedded_input_seqs, memory, src_mask, tgt_mask)

    return(decoded_input_seqs)
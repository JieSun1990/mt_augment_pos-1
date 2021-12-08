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
from torchtext.data.metrics import bleu_score

from tqdm.notebook import tqdm
from timeit import default_timer as timer

import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARENT_DIR = '/content/gdrive/MyDrive/CS287_Research_Project/Jennas_Code/' # for google colab. adjust accordingly
import sys
sys.path.append(PARENT_DIR)

from batchTF import *

### Training Epoch ###
### Augmentation methods applied here
def train_epoch(model, optimizer, loss_fn, train_dl,padding_idx):
    model.train()
    losses = 0

    for batch_idx, batch in enumerate(tqdm(train_dl,leave=False)):
        # Seqmix implementation goes here
        src = batch[0].to(device)
        tgt = batch[1].to(device)
        input_pos = batch[2].to(device)
        batch = Batch(src,tgt,pad=padding_idx)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask, input_pos=input_pos)
        out = model.generator(out)
        out_reshape = out.reshape(-1, out.shape[-1])
        trg_y_reshape = batch.trg_y.reshape(-1)
        loss = loss_fn(out_reshape, trg_y_reshape)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(train_dl)

### Validation Epoch ###
### Augmentation methods are NOT applied here
def val_epoch(model, loss_fn, val_dl,padding_idx):
    model.eval()
    losses = 0

    for batch_idx, batch in enumerate(tqdm(val_dl,leave=False)):

        src = batch[0].to(device)
        tgt = batch[1].to(device)
        batch = Batch(src,tgt,pad=padding_idx)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask,generate=True)
        out = model.generator(out)
        out_reshape = out.reshape(-1, out.shape[-1])
        trg_y_reshape = batch.trg_y.reshape(-1)
        loss = loss_fn(out_reshape, trg_y_reshape)
        losses += loss.item()

    return losses / len(val_dl)

### Greedy Decoding ###
### Used for language generation from output of seq2seq model
def greedy_decode(model, src, max_len, start_symbol, end_symbol):
  src = src.to(device)
  src_mask = torch.ones(1,1,len(src)).to(device)
  memory = model.encode(src, src_mask, generate = True)
  ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long()
  for i in range(max_len-1):
    tgt_mask = future_mask(ys.size(1)).type_as(src.data).long()
    #print(src.dtype, src_mask.dtype, memory.dtype, tgt_mask.dtype)
    #print(src.shape, src_mask.shape, memory.shape, tgt_mask.shape, ys.shape)
    out = model.decode(ys, memory, src_mask, tgt_mask, generate = True)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim = 1)
    next_word = next_word.data[0]
    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    if next_word == end_symbol:
      break
  return ys

### Translating Corpus ###
### Uses greedy decoding
def translate_corpus(
  vocab: Dict[str, int], 
  test_dl: DataLoader, 
  model: nn.Module,
  eos_idx,
  bos_idx,
  padding_idx,
  max_steps) -> Tuple[List[List[str]], List[List[List[str]]]]:
  
  index_to_word = {v: k for k, v in vocab.items()}
  candidate_text = []
  reference_text = []

  for batch_idx, batch in enumerate(tqdm(test_dl,leave=False)):
    source = batch[0]
    target = batch[1]
    for i in range(len(source)):  # for seq in batch
      source = source.to(device)
      src_seq = source[i,:].unsqueeze(0)
      cand = greedy_decode(model, src_seq, max_steps, bos_idx, eos_idx)[0]
      cand = [index_to_word[int(cand[j])] for j in range(len(cand))]
      if cand[-1] != "<EOS>":
        cand.append("<EOS>")
      ref = [index_to_word[int(target[i][j])] for j in range(len(target[i])) if int(target[i][j]) != padding_idx]
      candidate_text.append(cand)
      reference_text.append([ref])
      
  return candidate_text, reference_text

import random
import os

### Full Training Scheme ###
### Uses both train_epoch and val_epoch
### translate_corpus with greedy_decode occurs to generate validation BLEU in loop
### returns best model (selected by validation BLEU), train losses across epochs, val losses, bleus, and index (wrt epoch) of best performing model
def train(model, 
          train_ds, 
          train_dl, 
          val_ds, 
          val_dl, 
          device, 
          bos_idx, 
          eos_idx, 
          padding_idx, 
          epochs, 
          lr, 
          weight_decay,
          max_steps,
          path,
          overwrite = True):
  torch.manual_seed(100)

  if not os.path.isdir(path):
      print('creating directory', path)
      os.mkdir(path)
  else:
    print('directory already exists')
    print(path, 'contains', os.listdir(path))
    if overwrite:
      print('continuing anyways ... some files may be overwritten')
    else:
      print('set overwrite = True to continue')
      return None

  if not path.endswith('/'):
    path = path + '/'

  optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
  loss_fn = CrossEntropyLoss(ignore_index=padding_idx)                      
  epochs = epochs

  train_losses = []
  val_losses = []
  bleus = []
  model_paths = []

  for epoch in range(epochs):
      start_time = timer()
      train_loss = train_epoch(model, optimizer, loss_fn, train_dl,padding_idx)
      train_losses.append(train_loss)
      end_time = timer()
      val_loss = val_epoch(model, loss_fn, val_dl,padding_idx)
      val_losses.append(val_loss)
      print((f"Epoch: {epoch+1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f},"f"Epoch time = {(end_time - start_time):.3f}s"))

      candidate_text, reference_text = translate_corpus(train_ds.target_vocab, val_dl, model, eos_idx, bos_idx, padding_idx, max_steps)
      bleu = bleu_score(candidate_text, reference_text)
      bleus.append(bleu)
      print((f"Validation Bleu: {bleu}"))

      model_path = path + 'epoch' + str(epoch)
      torch.save(model, model_path)
      model_paths.append(model_path)

  best_bleu = max(bleus)
  which = bleus.index(best_bleu)
  best_model_path = model_paths[which]
  best_model = torch.load(best_model_path)

  return(best_model, train_losses, val_losses, bleus, which)


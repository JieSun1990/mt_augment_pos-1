from tqdm import tqdm
import pickle
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.distributions.beta import Beta
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
import os

from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARENT_DIR = '/content/gdrive/MyDrive/CS287_Research_Project/Jennas_Code/' # for google colab. adjust accordingly
import sys
sys.path.append(PARENT_DIR)
from Seq2Seq import Seq2Seq
from EncoderLSTM import EncoderLSTM
from DecoderLSTM import DecoderLSTM

MAX_STEPS = 40

@torch.no_grad()
def compute_test_loss(model, val_dl, loss_fn):
    all_preds, all_targs = [], []
    for i, (input_seqs, output_seqs, input_pos, output_pos) in enumerate(val_dl):
        preds, targs = model(input_seqs.to(device), output_seqs.to(device), generate = True)
        all_preds.append(preds)
        all_targs.append(targs)
    preds = torch.cat(all_preds, dim=0)
    targs = torch.cat(all_targs, dim=0)
    return loss_fn(preds, targs).item()

def seqmix_loss(loss_fn, pred, targ, padding_idx, lambda_):
    l1 = loss_fn(pred, targ[0])
    l2 = loss_fn(pred, targ[1])
    loss = l1 * lambda_.view(-1, 1) + l2 * (1-lambda_).view(-1, 1)
    
    return loss
    
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

def seqmix_ind_loss(loss_fn, pred, targ, batch_target, mix_batch_target, padding_idx, lambda_targ):
    # label_smoothed_nll_loss(pred, targ[0], epsilon = 0.1, ignore_index = padding_idx, reduce = False)
    # l1 = label_smoothed_nll_loss(pred, targ[0], epsilon = 0.1, ignore_index = padding_idx, reduce = False)
    l1 = loss_fn(pred, targ[0])
    # l2 = label_smoothed_nll_loss(pred, targ[1], epsilon = 0.1, ignore_index = padding_idx, reduce = False)
    l2 = loss_fn(pred, targ[1])
    loss = l1 * lambda_targ + l2 * (1-lambda_targ)
    return torch.mean(loss)

def seqmix_ind_pos_loss(loss_fn, pred, targ, targ_pos, padding_idx, lambda_):
    l1 = loss_fn(pred, targ[0])
    l2 = loss_fn(pred, targ[1])
    
    # loss = torch.zeros(targ[0].size())
    # for i in range(pred.size()[1]):
    #     index = targ_pos[i]
    #     loss[i] = l1[i] * lambda_[index].reshape(-1, 1, 1) + l2[i] * (1 - lambda_[index]).reshape(-1, 1, 1)

    loss = torch.zeros(l1.size())
    # for i in range(targ_pos[0].size()[0]):
    #     index = targ_pos[0][i]
    #     loss[i] = l1[i] * lambda_[index] + l2[i] * (1 - lambda_[index])
    pos_len = targ_pos[0].size()[0]
    input_len = l1.size()[0]
    for i in range(input_len):
        if i < pos_len:
            index = targ_pos[0][i]
            loss[i] = l1[i] * lambda_[index] + l2[i] * (1 - lambda_[index])
        else:
            loss[i] = l2[i]
    return torch.mean(loss)

def train(encoder, train_ds: Dataset, train_dl: DataLoader, val_ds: Dataset, val_dl: DataLoader, 
          device, bos_idx, eos_idx, path, overwrite = True, epochs: int = 8, hidden_size = 32, padding_idx = 1, lr = 0.001, weight_decay = 0,
          full_loss = False, augmentation_type = None, accum_iter = 32):
    
    if not os.path.isdir(path):
      print('creating directory', path)
      os.mkdir(path)
    else:
      print('directory already exists')
      print(path, 'contains', os.listdir(path))
      if overwrite:
        print('continuing anyways... some files may be overwritten')
      else:
        print('set overwrite = True to continue')
        return None

    if not path.endswith('/'):
      path = path + '/'

    # initialize model, optimizer, and loss functon
    model = Seq2Seq(encoder, device, len(train_ds.source_vocab), len(train_ds.target_vocab), 
                    hidden_size = hidden_size, padding_idx = padding_idx, augmentation_type = augmentation_type).to(device)
    optimizer = Adam(params = model.parameters(), lr = lr, weight_decay = weight_decay)
    if augmentation_type == 'seqmix_pos_ind' or augmentation_type == 'seqmix_pos_ind_r' or augmentation_type == 'seqmix_ind'  or augmentation_type == 'seqmix_ind_2' or augmentation_type == 'seqmix_pos_ind_r' or augmentation_type == 'seqmix_pos2' or augmentation_type == 'seqmix_ind_pos':
        loss_fn_seqmix_ind = CrossEntropyLoss(ignore_index = padding_idx, reduce = False)
        # loss_fn_seqmix_ind = label_smoothed_nll_loss(epsilon = 0.1, ignore_index = padding_idx, reduce = False)

        loss_fn = CrossEntropyLoss(ignore_index = padding_idx)
    else:
        loss_fn = CrossEntropyLoss(ignore_index = padding_idx)
        
    # keep track of losses
    losses = {
        'train': [], 
        'val': []
    }

    val_bleus = []
    model_paths = []

    if augmentation_type == 'seqmix' or augmentation_type == 'seqmix_lenmatch' or augmentation_type == 'seqmix_ind'  or augmentation_type == 'seqmix_ind_2' or augmentation_type == 'seqmix_pos' or augmentation_type == 'seqmix_pos_ind' or augmentation_type == 'seqmix_pos_ind_r' or augmentation_type == 'seqmix_pos2' or augmentation_type == 'seqmix_ind_pos':
      mix_dl = iter(train_dl)

    for e in range(epochs):
        running_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dl, leave = False)):            
            if augmentation_type == 'seqmix' or augmentation_type == 'seqmix_lenmatch' or augmentation_type == 'seqmix_pos' or augmentation_type == 'seqmix_pos_ind' or augmentation_type == 'seqmix_ind'  or augmentation_type == 'seqmix_ind_2' or augmentation_type == 'seqmix_ind_pos' or augmentation_type == 'seqmix_pos_ind_r' or augmentation_type == 'seqmix_pos2':
              # mixer_batch = next(iter(train_dl))
              try:
                  mix_batch = next(mix_dl)
              except StopIteration:
                  mix_dl = iter(train_dl)
                  mix_batch = next(mix_dl)
              
              ### experiment: similar length for seqmix
              if augmentation_type == 'seqmix_lenmatch':
                  param = 0
                  while abs((len(mix_batch[0][0])- len(batch[0][0])) > param) or torch.equal(batch[0][0], mix_batch[0][0]):
                      try:
                          mix_batch = next(mix_dl)
                      except StopIteration:
                          mix_dl = iter(train_dl)
                          mix_batch = next(mix_dl)
                          param = param + 1
                          if param > 10:
                              param += 40
            
              input_batch = pad_sequence([batch[0][0],mix_batch[0][0]], batch_first=True, padding_value=padding_idx)
              target_batch = pad_sequence([batch[1][0],mix_batch[1][0]], batch_first=True, padding_value=padding_idx)
              if augmentation_type == 'seqmix' or augmentation_type == 'seqmix_lenmatch':
                  pred, targ, lambda_ = model(input_batch[0].unsqueeze(0).to(device), target_batch[0].unsqueeze(0).to(device), seqmix_batch = (input_batch[1].unsqueeze(0).to(device), target_batch[1].unsqueeze(0).to(device)))
                  
                  l = seqmix_loss(loss_fn, pred, targ, padding_idx, lambda_)
              elif augmentation_type == 'seqmix_pos':
                  
                  pred, targ, lambda_ = model(input_batch[0].unsqueeze(0).to(device), target_batch[0].unsqueeze(0).to(device), seqmix_batch = (input_batch[1].unsqueeze(0).to(device), target_batch[1].unsqueeze(0).to(device)), input_pos = (batch[2].to(device), batch[3].to(device)), mix_pos = (mix_batch[2].to(device), mix_batch[3].to(device)), seqmix_pos = True)

                  l = seqmix_loss(loss_fn, pred, targ, padding_idx, lambda_)

              elif augmentation_type == 'seqmix_pos_ind_r':
                  pred, targ = model(input_batch[0].unsqueeze(0).to(device), target_batch[0].unsqueeze(0).to(device), seqmix_batch = (input_batch[1].unsqueeze(0).to(device), target_batch[1].unsqueeze(0).to(device)), input_pos = (batch[2].to(device), batch[3].to(device)), mix_pos = (mix_batch[2].to(device), mix_batch[3].to(device)), seqmix_pos_ind = True)
                  
                  l = loss_fn(pred, targ)

              elif augmentation_type == 'seqmix_pos_ind' or augmentation_type == 'seqmix_pos2':
                  pred, targ, lambda_targ = model(input_batch[0].unsqueeze(0).to(device), target_batch[0].unsqueeze(0).to(device), seqmix_batch = (input_batch[1].unsqueeze(0).to(device), target_batch[1].unsqueeze(0).to(device)), input_pos = (batch[2].to(device), batch[3].to(device)), mix_pos = (mix_batch[2].to(device), mix_batch[3].to(device)), seqmix_pos_ind = True)
                  
                  l = seqmix_ind_loss(pred, targ, target_batch[0], target_batch[1], padding_idx, lambda_targ)

              elif augmentation_type == 'seqmix_ind' or augmentation_type == 'seqmix_ind_2':
                  pred, targ, lambda_targ = model(input_batch[0].unsqueeze(0).to(device), target_batch[0].unsqueeze(0).to(device), seqmix_batch = (input_batch[1].unsqueeze(0).to(device), target_batch[1].unsqueeze(0).to(device)), seqmix_ind = True)
                  
                  l = seqmix_ind_loss(loss_fn, pred, targ, target_batch[0], target_batch[1], padding_idx, lambda_targ)
                  
              elif augmentation_type == 'seqmix_ind_pos':
                  pred, targ, lambda_pos = model(input_batch[0].unsqueeze(0).to(device), target_batch[0].unsqueeze(0).to(device), seqmix_batch = (input_batch[1].unsqueeze(0).to(device), target_batch[1].unsqueeze(0).to(device)), input_pos = (batch[2].to(device), batch[3].to(device)), mix_pos = (mix_batch[2].to(device), mix_batch[3].to(device)))
                #   loss_fn, pred, targ, targ_pos, padding_idx, lambda_targ
                  l = seqmix_ind_pos_loss(loss_fn_seqmix_ind, pred, targ, batch[3].to(device), padding_idx, lambda_pos)

            else:
              # predict + compute loss
              pred, targ = model(input_seqs = batch[0].to(device), output_seqs = batch[1].to(device), input_pos = batch[2].to(device))
              l = loss_fn(pred, targ)
            
            running_loss += l.item()
            l = l/accum_iter

            # update
            l.backward()

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dl)):
              optimizer.step()
              optimizer.zero_grad()

        if full_loss:
          # record losses computed on full training and validation datasets
          full_training_loss = compute_test_loss(model, train_dl, loss_fn)
        else:
          full_training_loss = running_loss/len(train_dl)

        candidate_text, reference_text = translate_corpus(
          train_ds.target_vocab, val_dl, model, 
          eos_idx = eos_idx, bos_idx = bos_idx,
          padding_idx = padding_idx
        )
        bleu = bleu_score(
          candidate_text, reference_text
        )
        val_bleus.append(bleu)

        full_val_loss = compute_test_loss(model, val_dl, loss_fn)
        losses['train'].append(full_training_loss)
        losses['val'].append(full_val_loss)

        model_path = path + 'epoch' + str(e)
        torch.save(model, model_path)
        model_paths.append(model_path)

        s = 'epoch ' + str(e) +': training loss = ' + \
            str(round(full_training_loss, 5)) + ' - validation loss = ' + str(round(full_val_loss, 5)) + \
            ' - validation bleu = ' + str(round(bleu, 5))
            
        print(s)

    best_bleu = max(val_bleus)
    which = val_bleus.index(best_bleu)
    best_model_path = model_paths[which]
    print("best model path:", best_model_path)
    best_model = torch.load(best_model_path)

    return (best_model, losses, val_bleus, which)


def translate_corpus(
    vocab: Dict[str, int], 
    test_dl: DataLoader, 
    model: nn.Module,
    eos_idx, bos_idx,
    padding_idx,
    max_steps = MAX_STEPS
) -> Tuple[List[List[str]], List[List[List[str]]]]:
    
    index_to_word = {v: k for k, v in vocab.items()}
    candidate_text = []
    reference_text = []

    for source, target, source_pos, target_pos in test_dl:
        for i in range(len(source)):  # for seq in batch
            source = source.to(device)

            cand = model.generate(source[i].view(1, -1), max_steps = max_steps, eos_idx = eos_idx, bos_idx = bos_idx)    
            cand = [index_to_word[int(cand[j])] for j in range(len(cand))]
            if cand[-1] != '<EOS>':
              cand.append('<EOS>')
            ref = [index_to_word[int(target[i][j])] for j in range(len(target[i])) if int(target[i][j]) != padding_idx]
            candidate_text.append(cand)
            reference_text.append([ref])
        
    return candidate_text, reference_text


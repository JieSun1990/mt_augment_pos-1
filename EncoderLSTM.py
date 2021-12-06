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
from torchtext.data.metrics import bleu_score
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderLSTM(nn.Module):
    def __init__(
        self, 

        # regular encoder parameters
        input_size: int,
        hidden_size: int, 
        padding_idx: int, 
        device,

        # augmentation encoder parameters    
        augmentation_type: str = None,  # one of "swap", "drop", "blank", "smooth", "lmsample"
        lambda_ = None,                 # seqmix lambda parameter
        gamma = None, 
        k = None,                       # window size for "swap" method
        unk_idx = None,                 # placeholder for "blank" method
        unigram_freq = None,            # to sample from with "smooth" method
        language_model = None,          # to sample from in "lmsample" method
        source_vocab_to_tags = None,
        source_pos_idx = None
    ):
        super().__init__()
        
        self.input_size = input_size        # vocab size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.embedding_size = 64
        self.device = device

        self.embedding = torch.nn.Embedding(
            num_embeddings = self.input_size,
            embedding_dim = self.embedding_size,
            padding_idx = padding_idx
        )
        
        self.lstm = torch.nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = self.hidden_size,
            batch_first = True
        )

        self.augmentation_type = augmentation_type
        self.gamma = gamma
        self.k = k
        self.unk_idx = unk_idx
        self.unigram_freq = unigram_freq
        self.language_model = language_model
        self.source_vocab_to_tags = source_vocab_to_tags
        self.source_pos_idx = source_pos_idx

    def forward(self, input_seqs: torch.tensor, input_pos = None, generate = False, lambda_ = None, mix_seqs = None, mix_pos = None) -> Tuple[torch.tensor, torch.tensor]: 
        N = len(input_seqs)
        # packing requires lengths of non-padded sequences
        lengths = torch.Tensor([int(sum(x != self.padding_idx)) for x in input_seqs])
        
        if not generate and self.augmentation_type == 'swap':
          input_seqs = self.augment_swap(input_seqs, lengths)

        elif not generate and self.augmentation_type == 'drop':
          input_seqs, lengths = self.augment_drop(input_seqs, lengths)

        elif not generate and self.augmentation_type == 'blank':
          input_seqs = self.augment_blank(input_seqs, lengths)

        elif not generate and self.augmentation_type == 'smooth':
          input_seqs = self.augment_smooth(input_seqs, lengths)

        elif not generate and self.augmentation_type == 'smooth_pos':
          input_seqs = self.augment_smooth_pos(input_seqs, input_pos, lengths)

        elif not generate and self.augmentation_type == 'lmsample':
          input_seqs = self.augment_lmsample(input_seqs, lengths)

        elif not generate and self.augmentation_type == 'lmsample_pos':
          input_seqs = self.augment_lmsample_pos(input_seqs, input_pos, lengths)      
        
        if not generate and (self.augmentation_type == 'seqmix'):
          embedded_input_seqs = self.augment_seqmix(input_seqs, mix_seqs, lengths, lambda_)
          
        elif not generate and (self.augmentation_type == 'seqmix_ind'  or self.augmentation_type == 'seqmix_ind_2'):
          embedded_input_seqs = self.augment_seqmix_ind(input_seqs, mix_seqs, lengths, lambda_)
          
        elif not generate and self.augmentation_type == 'seqmix_ind_pos':
          embedded_input_seqs = self.augment_seqmix_ind_pos(input_seqs, mix_seqs, input_pos, mix_pos, lengths, lambda_)
        elif not generate and self.augmentation_type == 'seqmix_pos':
          embedded_input_seqs = self.augment_seqmix_pos(input_seqs, mix_seqs, input_pos, mix_pos, lengths, lambda_)
        elif not generate and (self.augmentation_type == 'seqmix_pos_ind' or self.augmentation_type == 'seqmix_pos_ind_r'):
          embedded_input_seqs = self.augment_seqmix_pos_ind(input_seqs, mix_seqs, input_pos, mix_pos, lengths, lambda_)
        else:
          embedded_input_seqs = self.embedding(input_seqs.to(self.device))
          if not generate and self.augmentation_type == 'soft':
            embedded_input_seqs = self.augment_soft(input_seqs, embedded_input_seqs, lengths)        # embedding layer
          elif not generate and self.augmentation_type == 'soft_pos':
            embedded_input_seqs = self.augment_soft_pos(input_seqs, input_pos, embedded_input_seqs, lengths)
          
        # pack the padded embeddings
        packed_padded_seqs = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_input_seqs, batch_first = True, lengths = lengths,
            enforce_sorted = False
        )

        # run lstm
        lstm_out, (h_t, c_t) = self.lstm(packed_padded_seqs)
        
        # return (a) the final hidden state (i.e., "short-term memory") 
        # and (b) the final cell state (i.e., "long-term memory") of the LSTM.
        return (h_t, c_t)

    ### Implementations of augmentation methods ###
    def augment_swap(self, input_seqs, lengths):
      # selecting indexes to augment (swap) with probability gamma
      swap_idx1 = [
        # don't consider first or last <bos> <eos>
        np.array(list(range(1, int(l)-1)))[torch.rand(int(l-2)).numpy() < self.gamma] 
        for l in lengths
      ]

      # for each index we're swapping, select idx to swap with (within a window size k)
      swap_idx2 = [[
          self.swap_with(swap_idx = int(swap_idx1[i][j]), length = int(lengths[i]), k = self.k) 
          for j in range(len(swap_idx1[i]))] 
        for i in range(len(swap_idx1))]

      # identity matrix for each seq, 0s where padded
      reorder_diags = [
        torch.Tensor([1]*int(length) + [0]*(int(max(lengths)- length))) 
        for length in lengths
      ]
      reorder = torch.stack([torch.diag(reorder_diags[i]) for i in range(len(lengths))])

      # swapping swap_idx1 and swap_idx2 rows in identity matrix
      reorder = torch.stack([
        self.swap_rows(reorder[i], swap_idx1[i], swap_idx2[i]) 
        for i in range(len(reorder))
      ])

      # matrix multiplication returns sequences with swapped tokens
      input_seqs = torch.stack([
        torch.matmul(input_seqs[i].to('cpu'), reorder[i].long()) 
        for i in range(len(input_seqs))
      ]).to(self.device)

      return input_seqs

    def augment_drop(self, input_seqs, lengths):
      drop_idx = [
        # don't consider first or last <bos> <eos>
        np.array(list(range(1, int(l)-1)))[torch.rand(int(l-2)).numpy() < self.gamma] 
        for l in lengths
      ]
      drop_idx = [
        drop_idx[i] if len(drop_idx[i]) < lengths[i] else drop_idx[i][:-1]
        for i in range(len(drop_idx))
      ]
      drop_dags = [torch.Tensor([
          1 if (j not in drop_idx[i] and j < int(lengths[i]))
          else 0 
          for j in range(int(max(lengths)))
        ])
        for i in range(len(lengths))
      ]
      drop = torch.stack([torch.diag(drop_dags[i]) for i in range(len(lengths))])

      input_seqs = torch.stack([
        torch.matmul(input_seqs[i].to('cpu'), drop[i].long()) 
        for i in range(len(input_seqs))
      ])

      # push zeros to the end again
      input_seqs = torch.stack([
        torch.cat([
          seq[seq != self.padding_idx],
          torch.Tensor([0]*(int(max(lengths) - sum(seq != self.padding_idx))))
        ])
        for seq in input_seqs
      ]).long().to(self.device)

      lengths = torch.Tensor([int(sum(x != self.padding_idx)) for x in input_seqs])

      return input_seqs, lengths

    def augment_blank(self, input_seqs, lengths):
      blank_idx = [
        # don't consider first or last <bos> <eos>
        np.array(list(range(1, int(l)-1)))[torch.rand(int(l-2)).numpy() < self.gamma] 
        for l in lengths
      ]
      input_seqs = torch.stack([
        self.fill_blanks(input_seqs[i].to('cpu'), blank_idx[i])
        for i in range(len(input_seqs))
      ]).to(self.device)
      return input_seqs

    def augment_smooth(self, input_seqs, lengths):
      to_replace_idx = [
        # don't consider first or last <bos> <eos>
        np.array(list(range(1, int(l)-1)))[torch.rand(int(l-2)).numpy() < self.gamma] 
        for l in lengths
      ]
      replace_with_val = [
        self.select_from_unigram(
            freq_dist = self.unigram_freq,
            n = len(to_replace_idx[i])
        )
        for i in range(len(to_replace_idx))
      ]
      input_seqs = torch.stack([
        self.replace_tokens(input_seqs[i].to('cpu'), to_replace_idx[i], replace_with_val[i])
        for i in range(len(input_seqs))
      ]).to(self.device)
      return input_seqs

    def augment_smooth_pos(self, input_seqs, input_pos, lengths):
      to_replace_idx = [
        # don't consider first or last <bos> <eos>
        # first dimension is sequence
        # second dimension is indices to replace in each sentence
        np.array(list(range(1, int(l)-1)))[torch.rand(int(l-2)).numpy() < self.gamma] 
        for l in lengths
      ]
      replace_with_val = [
        # ith sequence
        # idk word to replaace
        torch.Tensor([
          self.select_from_unigram_pos(
            freq_dist = self.unigram_freq,
            n = 1,
            pos = input_pos[i][idx]
          )
          for idx in to_replace_idx[i]
        ])
        for i in range(len(to_replace_idx))
      ]
      input_seqs = torch.stack([
        self.replace_tokens(input_seqs[i].to('cpu'), to_replace_idx[i], replace_with_val[i])
        for i in range(len(input_seqs))
      ]).to(self.device)
      return input_seqs

    def augment_lmsample(self, input_seqs, lengths):
      to_replace_idx = [
        # don't consider first or last <bos> <eos>
        np.array(list(range(1, int(l)-1)))[torch.rand(int(l-2)).numpy() < self.gamma] 
        for l in lengths
      ]
      replace_with_val = [
        [int(self.select_from_probs(
          np.exp(self.language_model.generate(input_seqs[i][:idx].unsqueeze(0)).to('cpu'))[0]
        )) for idx in to_replace_idx[i]]
        for i in range(len(to_replace_idx))
      ]
      input_seqs = torch.stack([
        self.replace_tokens(
            input_seqs[i].to('cpu'), 
            to_replace_idx[i], 
            replace_with_val[i]
        )
        for i in range(len(input_seqs))
      ]).to(self.device)
      return input_seqs

    def augment_lmsample_pos(self, input_seqs, input_pos, lengths):
      to_replace_idx = [
        # don't consider first or last <bos> <eos>
        np.array(list(range(1, int(l)-1)))[torch.rand(int(l-2)).numpy() < self.gamma] 
        for l in lengths
      ]
      replace_with_val = [
        [int(self.select_from_probs(
          self.get_probs_pos(input_seqs[i][:idx], input_pos[i][idx])
        )) for idx in to_replace_idx[i]]
        for i in range(len(to_replace_idx))
      ]
      input_seqs = torch.stack([
        self.replace_tokens(
            input_seqs[i].to('cpu'), 
            to_replace_idx[i], 
            replace_with_val[i]
        )
        for i in range(len(input_seqs))
      ]).to(self.device)
      return input_seqs

    def augment_soft(self, input_seqs, embedded_input_seqs, lengths):
      # select indexes of embeddingsto be replaced with an average
      to_replace_idx = [
        # don't consider first or last <bos> <eos>
        np.array(list(range(1, int(l)-1)))[torch.rand(int(l-2)).numpy() < self.gamma] 
        for l in lengths
      ]
      # tensor with all embeddings in idx order
      idxs = torch.Tensor(list(range(self.input_size)))
      all_embeddings = self.embedding(torch.Tensor(idxs).to(self.device).long()).to('cpu')
      embedded_input_seqs = embedded_input_seqs.to('cpu')
      for i in range(len(to_replace_idx)):
          if len(to_replace_idx[i]) > 0:
              embedded_input_seqs[i][to_replace_idx[i]] = torch.stack([
                # multiply probs vector by embeddings matrix -> weighted average
                np.matmul(
                    # use lm to predict next word given context
                    # context = seq up until (not including) idx
                    np.exp(self.language_model.generate(input_seqs[i][:idx].unsqueeze(0)).to('cpu'))[0], 
                    all_embeddings.detach()
                ) 
                for idx in to_replace_idx[i]
              ])
      return embedded_input_seqs.to(self.device)
      
    def augment_soft_pos(self, input_seqs, input_pos, embedded_input_seqs, lengths):
      # select indexes of embeddingsto be replaced with an average
      to_replace_idx = [
        # don't consider first or last <bos> <eos>
        np.array(list(range(1, int(l)-1)))[torch.rand(int(l-2)).numpy() < self.gamma] 
        for l in lengths
      ]
      # tensor with all embeddings in idx order
      idxs = torch.Tensor(list(range(self.input_size)))
      all_embeddings = self.embedding(torch.Tensor(idxs).to(self.device).long()).to('cpu')
      embedded_input_seqs = embedded_input_seqs.to('cpu')
      for i in range(len(to_replace_idx)):
          # ith sequence
          if len(to_replace_idx[i]) > 0:
              # to_replace_idx[i] are words to replace
              embedded_input_seqs[i][to_replace_idx[i]] = torch.stack([
                # multiply probs vector by embeddings matrix -> weighted average
                np.matmul(
                    # use lm to predict next word given context
                    # context = seq up until (not including) idx
                    self.get_probs_pos(input_seqs[i][:idx], input_pos[i][idx]), 
                    all_embeddings.detach()
                ) 
                for idx in to_replace_idx[i]
              ])
      return embedded_input_seqs.to(self.device)
    
    def augment_seqmix(self, input_seqs, mix_seqs, lengths, lambda_):
        src_tokens_a = input_seqs
        src_tokens_b = mix_seqs

        encoder_embedding_a = self.embedding(src_tokens_a)
        encoder_embedding_b = self.embedding(src_tokens_b)

        encoder_embedding = encoder_embedding_a * lambda_.reshape(-1, 1, 1) + encoder_embedding_b * (1 - lambda_).reshape(-1, 1, 1)
        
        return encoder_embedding
        
    def augment_seqmix_ind(self, input_seqs, mix_seqs, lengths, lambda_):
        src_tokens_a = input_seqs
        src_tokens_b = mix_seqs

        encoder_embedding_a = self.embedding(src_tokens_a)
        encoder_embedding_b = self.embedding(src_tokens_b)

        embedding_size = encoder_embedding_a.size()
        encoder_embedding = lambda_.unsqueeze(1).expand(embedding_size) * encoder_embedding_a[0] + (1 - lambda_).unsqueeze(1).expand(embedding_size) * encoder_embedding_a[0]

        return encoder_embedding
        
    def augment_seqmix_ind_pos(self, input_seqs, mix_seqs, input_pos, mix_pos, lengths, lambda_):
        src_tokens_a = input_seqs
        src_tokens_b = mix_seqs
    
        encoder_embedding_a = self.embedding(src_tokens_a)
        encoder_embedding_b = self.embedding(src_tokens_b)
    
        merged_embedding = torch.zeros_like(encoder_embedding_a)
        embedding_size = encoder_embedding_a.size()
        # merged_embedding = torch.zeros((input_pos.size()[0],input_pos.size()[1],encoder_embedding_a.size()[2]))
        # merged_embedding = torch.zeros(src_tokens_a.size())
        
        pos_len = input_pos[0].size()[0]
        input_len = encoder_embedding_a.size()[1]
        
        for i in range(input_len):
            if i < pos_len:
                index = input_pos[0][i]
                merged_embedding[0][i] = encoder_embedding_a[0][i] * lambda_[index] + encoder_embedding_b[0][i] * (1 - lambda_[index])
            else:
                merged_embedding[0][i] = encoder_embedding_b[0][i]
                
        return merged_embedding.to(self.device)
        
    def augment_seqmix_pos(self, input_seqs, mix_seqs, input_pos, mix_pos, lengths, lambda_):
        encoder_embedding_a = self.embedding(input_seqs)
        encoder_embedding_b = self.embedding(mix_seqs)

        ### get the positions of each type of speech
        list1 = input_pos.cpu().tolist()[0]
        list2 = mix_pos.cpu().tolist()[0]

        pos_dict = {}
        for i in range(50):
            pos_dict[i] = []
        pos_dict[-1] = []
        for i in range(len(list2)):
            if list2[i] == 0:
                pos_dict[list2[i]].append(i)
            
        merge_indices = []
        for i in range(max(len(list1), len(list2))):
            try:
                inner_list = pos_dict[list1[i]]
                merge_indices.append(inner_list.pop(0))
                # pos_dict = None
            except:
                merge_indices.append(-1)
                
        # unmerged = []
        # for l in pos_dict.values():
        #     if len(l) != 0:
        #         for val in l:
        #             unmerged.append(val)
        
        # unmerged.sort()
        
        # j_counter = -1
        # for val in unmerged:
        #     while j_counter < len(merge_indices):
        #         j_counter += 1
        #         if merge_indices[j_counter] == -1:
        #             merge_indices[j_counter] = val
        #             break
            
        merged_embedding = torch.zeros_like(encoder_embedding_a)
        
        for i in range(len(merged_embedding)):
            index = merge_indices[i]
            if index == -1:
                merged_embedding[0][i] = encoder_embedding_a[0][i]
                # * lambda_.reshape(-1, 1, 1)
            else:
                merged_embedding[0][i] = encoder_embedding_a[0][i] * lambda_.reshape(-1, 1, 1) + encoder_embedding_b[0][index] * (1 - lambda_).reshape(-1, 1, 1)
        return merged_embedding
        
    def augment_seqmix_pos_ind(self, input_seqs, mix_seqs, input_pos, mix_pos, lengths, lambda_):
        encoder_embedding_a = self.embedding(input_seqs)
        encoder_embedding_b = self.embedding(mix_seqs)

        ### get the positions of each type of speech
        list1 = input_pos.cpu().numpy()
        list2 = mix_pos.cpu().numpy()

        pos_dict = {}
        for i in range(18):
            pos_dict[i] = []
        if self.augmentation_type == 'seqmix_pos_ind_r':
            for i in range(len(list2[0])):
                # if list2[0][i] == 7:
                pos_dict[list2[0][i]].append(i)
                # print(i)
        else:
            for i in range(len(list2[0])):
                pos_dict[list2[0][i]].append(i)
                    
        merge_indices = []
        for i in range(len(list1)):
            try:
                merge_indices.append(pos_dict[list1[i]].pop(0))
            except:
                merge_indices.append(-1)
        
        # unmerged = []
        # for l in pos_dict.values():
        #     if len(l) != 0:
        #         for val in l:
        #             unmerged.append(l)
        
        # unmerged.sort()
        
        # j_counter = 0
        # for val in unmerged:
        #     while j_counter < len(merge_indices):
        #         j_counter += 1
        #     # for j in range(len(merge_indices)):
        #     #     j_counter+
        #         if merge_indices[j_counter] == -1:
        #             merge_indices[j_counter] = val
        #             break

        merged_embedding = torch.zeros_like(encoder_embedding_a)
        
        
        for i in range(len(merged_embedding)):
            index = merge_indices[i]
            if index == -1:
                merged_embedding[0][i] = encoder_embedding_a[0][i]
                # * lambda_.reshape(-1, 1, 1)
            else:
                lambda_index = input_pos[i]
                merged_embedding[0][i] = encoder_embedding_a[0][i] * lambda_[lambda_index].reshape(-1, 1, 1) + encoder_embedding_b[0][index] * (1 - lambda_[lambda_index]).reshape(-1, 1, 1)
                
        return merged_embedding
        
    ### Utility functions for augmentation methods ###
    
    def augment_lm_mix(self, input_seqs, lengths, lambda_):
        input_seqs = input_seqs

    def swap_with(self, swap_idx, length, k):
        lower = max(1, swap_idx - k)
        upper = min(length - 1, swap_idx + k + 1)
        choices = list(range(lower, upper))
        choices.remove(swap_idx)
        if len(choices) == 0:
          return swap_idx
        else:
          return int(random.choice(choices))

    def swap_row(self, tensor, row1, row2):
        index = torch.LongTensor(list(range(len(tensor))))
        index[row1] = row2
        index[row2] = row1
        return tensor[index]

    def swap_rows(self, tensor, row1s, row2s):
        for i in range(len(row1s)):
            tensor = self.swap_row(tensor, row1s[i], row2s[i])
        return tensor

    def fill_blanks(self, this_seq, indexes):
        for i in indexes:
          this_seq[i] = self.unk_idx
        return this_seq

    def replace_tokens(self, this_seq, to_replace_idx, replace_with_val):
      for idx, val in zip(to_replace_idx, replace_with_val):
        this_seq[idx] = val
      return this_seq

    def select_from_unigram(self, freq_dist, n):
      return torch.Tensor(
          np.random.choice(
            list(range(len(freq_dist))), 
            n, replace = True,
            p = freq_dist
          )
      )

    def select_from_unigram_pos(self, freq_dist, n, pos):
      select = self.source_pos_idx[int(pos)]
      if sum(select) > 0:
        freq_dist = np.array(freq_dist)*np.array(select)
        freq_dist = freq_dist/sum(freq_dist)

      return torch.Tensor(
          np.random.choice(
            list(range(len(freq_dist))), 
            size = n, replace = True,
            p = freq_dist
          )
      )

    def select_from_probs(self, probs, n = 1): 
      return torch.Tensor(
          np.random.choice(
            np.array(list(range(len(probs)))), 
            size = n, replace = True,
            p = probs.numpy()
          )
      )

    def get_probs_pos(self, context, pos):
      softmaxout = self.language_model.generate(context.unsqueeze(0))
      probs = np.exp(softmaxout.to('cpu')).squeeze(0)
      select = self.source_pos_idx[int(pos)]
      if sum(select) > 0:
        temp = np.multiply(
          np.array(probs),
          np.array(select)
        )
        if sum(temp) > 0:
          probs = temp
      probs = torch.Tensor(probs/sum(probs))
      return probs



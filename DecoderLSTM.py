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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARENT_DIR = '/content/gdrive/MyDrive/CS287_Research_Project/Jennas_Code/' # for google colab. adjust accordingly
import sys
sys.path.append(PARENT_DIR)
from EncoderLSTM import EncoderLSTM

class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, padding_idx: int, device, augmentation_type: str = None):
        super().__init__()
        
        # variables
        self.input_size = input_size        # vocab size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.embedding_size = 64
        self.device = device
        self.augmentation_type = augmentation_type
        # layers
        self.embedding = torch.nn.Embedding(
            num_embeddings = self.input_size,
            embedding_dim = self.embedding_size,
            padding_idx = padding_idx
        )
        
        self.lstm = torch.nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = self.hidden_size
        )
        
    def forward(
        self, 
        input_seqs: torch.tensor, 
        hidden_init: torch.tensor, 
        cell_init: torch.tensor,
        mix_seqs = None,
        lambda_ = None,
        generate = False,
        input_pos = None,
        mix_pos = None
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        
        N = len(input_seqs)
        
        # packing requires lengths of non-padded sequences
        lengths = torch.Tensor([int(sum(x != self.padding_idx)) for x in input_seqs])
        
        # embedding layer
        if (self.augmentation_type == 'seqmix_pos' or self.augmentation_type == 'seqmix_pos_ind') and generate == False:
            
            # lengths = torch.Tensor([int(sum(x != self.padding_idx)) for x in input_seqs])
            lengths2 = torch.Tensor([int(sum(x != self.padding_idx)) for x in mix_seqs])
            lengths = torch.max(lengths, lengths2)
            embedded_input_seqs, reorder_mix_seqs = self.augment_seqmix_pos(input_seqs, mix_seqs, input_pos, mix_pos, lengths, lambda_)
        elif (self.augmentation_type == 'seqmix') and generate == False:
            lengths2 = torch.Tensor([int(sum(x != self.padding_idx)) for x in mix_seqs])
            lengths = torch.max(lengths, lengths2)
            embedded_input_seqs = self.augment_seqmix(input_seqs, mix_seqs, lengths = lengths, lambda_ = lambda_)
        elif (self.augmentation_type == 'seqmix_ind' or self.augmentation_type == 'seqmix_ind_2') and generate == False:
            embedded_input_seqs = self.augment_seqmix_ind(input_seqs, mix_seqs, lengths = lengths, lambda_ = lambda_)
        elif (self.augmentation_type == 'seqmix_ind_pos') and generate == False:
            embedded_input_seqs = self.augment_seqmix_ind_pos(input_seqs, mix_seqs, input_pos, mix_pos, lengths = lengths, lambda_ = lambda_)
        else:
            embedded_input_seqs = self.embedding(input_seqs.to(device))
        
        # pack the padded embeddings
        packed_padded_seqs = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_input_seqs, batch_first = True, lengths = lengths,
            enforce_sorted = False 
        )
        
        # run lstm
        lstm_out, (h_t, c_t) = self.lstm(packed_padded_seqs, (hidden_init, cell_init))
        
        # return (a) the hidden states of the LSTM for all time steps (lstm_out), (b) the final hidden state of the LSTM, and (c) the final cell state of the LSTM
        if not generate and (self.augmentation_type == 'seqmix_pos' or self.augmentation_type == 'seqmix_pos_ind'):
            return lstm_out, (h_t, c_t), torch.tensor(reorder_mix_seqs).unsqueeze(0).to(device)
        else:
            return lstm_out, (h_t, c_t)
        
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
        # encoder_embedding = torch.zeros_like(encoder_embedding_a)
        # encoder_embedding = encoder_embedding_a[0] * lambda_ + encoder_embedding_b[0] * (1 - lambda_)
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

        list2_words = mix_seqs.cpu().numpy()[0]

        # length = max(len(list1), len(list2))
        pos_dict = {}
        for i in range(50):
            pos_dict[i] = []
        pos_dict[-1] = []
        
        for i in range(len(list2)):
            if list2[i] == 0:
                pos_dict[list2[i]].append(i)
            # pos_dict[list2[i]].append(i)
            
        reorder_mix_seqs = []
        merge_indices = []

        for i in range(int(lengths.item())):
            try: 
                inner_list = pos_dict[list1[i]]
                word_index = inner_list.pop(0)
                merge_indices.append(word_index)
                reorder_mix_seqs.append(list2_words[word_index])
                # pos_dict = None
            except:
                merge_indices.append(-1)
                reorder_mix_seqs.append(1)
                
        # unmerged = []
        # for l in pos_dict.values():
        #     if len(l) != 0:
        #         for val in l:
        #             unmerged.append(val)

        # unmerged.sort()
                
                
        ### commenting out the forced merge
        # j_counter = 0
        # for val in unmerged:
        #     while j_counter < len(reorder_mix_seqs):
        #         if merge_indices[j_counter] == -1:
        #             merge_indices[j_counter] = val
        #             reorder_mix_seqs[j_counter] = list2_words[val]
        #             j_counter += 1
        #             break
        #         else:
        #             j_counter += 1
        merged_embedding = torch.zeros_like(encoder_embedding_a)
        if self.augmentation_type == 'seqmix_pos':
            for i in range(len(merged_embedding)):
                index = merge_indices[i]
                if index == -1:
                    merged_embedding[0][i] = encoder_embedding_a[0][i] 
                    # * lambda_.reshape(-1, 1, 1)
                else:
                    merged_embedding[0][i] = encoder_embedding_a[0][i] * lambda_.reshape(-1, 1, 1) + encoder_embedding_b[0][index] * (1 - lambda_).reshape(-1, 1, 1)
        elif self.augmentation_type == 'seqmix_pos_ind':
            for i in range(len(merged_embedding)):
                index = merge_indices[i]
                if index == -1:
                    merged_embedding[0][i] = encoder_embedding_a[0][i]
                    # * lambda_[i]
                else:
                    merged_embedding[0][i] = encoder_embedding_a[0][i] * lambda_[i] + encoder_embedding_b[0][index] * (1 - lambda_)[i]
                    
        return merged_embedding, reorder_mix_seqs
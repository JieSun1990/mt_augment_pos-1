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
from torch.distributions.uniform import Uniform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARENT_DIR = '/content/gdrive/MyDrive/CS287_Research_Project/Jennas_Code/' # for google colab. adjust accordingly
import sys
sys.path.append(PARENT_DIR)
from EncoderLSTM import EncoderLSTM
from DecoderLSTM import DecoderLSTM

class Seq2Seq(nn.Module):
    def __init__(
        self, 
        encoder,
        device,
        input_vocab_size: int, 
        output_vocab_size: int, 
        hidden_size: int, 
        padding_idx: int,
        augmentation_type: str
    ):
        super().__init__()
        if augmentation_type == 'seqmix_lenmatch':
            augmentation_type = 'seqmix'
            
        # initialize encoder and decoder
        self.augmentation_type = augmentation_type
        if self.augmentation_type == 'seqmix_pos_ind' or self.augmentation_type == 'seqmix_ind':
            self.dist = Beta(0.1, 0.1)
            # Beta(0.1, 0.1)
        elif self.augmentation_type == 'seqmix_ind_2':
            self.dist = Beta(0.1, 0.1)
            
        elif self.augmentation_type == 'seqmix_pos_ind_r' or self.augmentation_type == 'seqmix_pos':
            self.dist = Uniform(0.95, 1)
            # Beta(0.1, 10)
        # elif self.augmentation_type == 'seqmix':
        #     self.dist = Uniform(0.5, 1)
            # self.augmentation_type == 'seqmix'
        else:
            self.dist = Beta(0.1, 0.1)
            
        self.encoder = encoder
        self.decoder = DecoderLSTM(
            input_size = output_vocab_size, 
            hidden_size = hidden_size, 
            padding_idx = padding_idx,
            device = device,
            augmentation_type = augmentation_type
        )
        
        # projection layer to map hidden states of the decoder to the target vocabulary for word prediction
        # from size of hidden layer to size of output vocab
        self.linear = nn.Linear(hidden_size, output_vocab_size)

        # utility
        self.zero = torch.Tensor([0]).to(device)

        # initialize logsoftmax
        self.logsoft = nn.LogSoftmax(dim = 1)
        
    def forward(
        self, input_seqs: torch.tensor, output_seqs: torch.tensor, input_pos = None, generate = False, seqmix_pos_ind = False, seqmix_ind = False, seqmix_pos = False, seqmix_batch = None, mix_pos = None
    ) -> Tuple[torch.tensor, torch.tensor]:
        
        
        
        # take a BxTi (batch by seq length of input) input_seqs tensor and its corresponding target translation, 
        #    the BxTo (batch by seq length of output) output_seqs tensor
        # return a tuple of (1) a flattened predictions tensor preds of size MoxVo (number non-padded tokens in output_seqs by size of output vocab) and 
        # (2) a flattened target words tensor targs of length Mo (number non-padded tokens in output_seqs) (Note: targs should not contain any padding indices)
        
        # strip off the final <EOS> tag of output_seqs for training
        # strip off the initial <EOS> tag of output_seqs for targs
        with torch.cuda.device_of(output_seqs):
          output_seqs_wo_end = output_seqs.clone()
          output_seqs_wo_start = output_seqs.clone()
        for i in range(len(output_seqs)):
            output_seqs_wo_end[i][1:][output_seqs_wo_end[i][1:] == 1] = 0
            output_seqs_wo_start[i] = torch.cat([output_seqs_wo_start[i][1:], self.zero])
        if seqmix_batch and not generate:

            mix_output_seqs = seqmix_batch[1]
            mix_output_seqs_wo_end = mix_output_seqs.clone()
            mix_output_seqs_wo_start = mix_output_seqs.clone()
            
            for i in range(len(output_seqs)):
                mix_output_seqs_wo_end[i][1:][mix_output_seqs_wo_end[i][1:] == 1] = 0
                mix_output_seqs_wo_start[i] = torch.cat([mix_output_seqs_wo_start[i][1:], self.zero.to(device)])

            if self.augmentation_type == 'seqmix_pos_ind' or self.augmentation_type == 'seqmix_ind' or self.augmentation_type == 'seqmix_pos_ind_r':
                
                input_size = max(input_seqs.size()[1], seqmix_batch[0].size()[1])
                target_size = max(output_seqs.size()[1], seqmix_batch[1].size()[1])
                
                lambda_ = self.dist.sample(sample_shape=[input_size])
                lambda_ = torch.max(lambda_, 1 - lambda_)
                lambda_ = lambda_.to(device)
                
                lambda_targ = self.dist.sample(sample_shape=[target_size])
                lambda_targ = torch.max(lambda_targ, 1 - lambda_targ)
                lambda_targ = lambda_targ.to(device)
                
            elif self.augmentation_type == 'seqmix_ind_2':
                input_size = max(input_seqs.size()[1], seqmix_batch[0].size()[1])
                target_size = max(output_seqs.size()[1], seqmix_batch[1].size()[1])
                
                lambda_val = self.dist.sample(sample_shape=[1])
                lambda_val = torch.max(lambda_val, 1 - lambda_val)
                
                # norm = torch.normal(lambda_val, 0.05)
                # (0.12**0.5)*torch.randn(1,)
                lambda_ = (0.005**0.5)*torch.randn(input_size) + lambda_val
                lambda_ = torch.min(lambda_, torch.ones(input_size))
                # norm.sample([1,sample_size])
                lambda_ = lambda_.to(device)
                
                # norm = torch.normal(lambda_val, 0.05)
                # (0.12**0.5)*torch.randn(5, 10, 20)
                lambda_targ = (0.005**0.5)*torch.randn(target_size) + lambda_val
                lambda_targ = torch.min(lambda_targ, torch.ones(target_size))
                # norm.sample([1,sample_size])
                lambda_targ = lambda_targ.to(device)

            elif self.augmentation_type == 'seqmix_ind_pos':
                # input_size = max(input_seqs.size()[1], seqmix_batch[0].size()[1])
                # target_size = max(output_seqs.size()[1], seqmix_batch[1].size()[1])
                
                lambda_pos = self.dist.sample(sample_shape=[20])
                lambda_pos = torch.max(lambda_pos, 1 - lambda_pos)
                lambda_pos = lambda_pos.to(device)
                
                # lambda_pos = self.dist.sample(sample_shape=[18])
                # lambda_pos = lambda_pos.to(device)
            else:
                lambda_ = self.dist.sample(sample_shape=[input_seqs.size()[0]])
                lambda_ = torch.max(lambda_, 1 - lambda_)
                lambda_ = lambda_.to(device)
            
            # pass encoder hidden states and outputs through decoder
            if self.augmentation_type == 'seqmix_pos_ind':
              
                (h_source, c_source) = self.encoder(input_seqs, mix_seqs = seqmix_batch[0], generate = generate, lambda_ = lambda_, input_pos = input_pos[0], mix_pos = mix_pos[0])
                
                states_pred, (h_pred, c_pred), reorder_mix_seqs = self.decoder(output_seqs_wo_end, mix_seqs = mix_output_seqs_wo_end, hidden_init = h_source, cell_init = c_source, lambda_ = lambda_targ, input_pos = input_pos[1], mix_pos = mix_pos[1])
                
            elif self.augmentation_type == 'seqmix_pos_ind_r':
                (h_source, c_source) = self.encoder(input_seqs, mix_seqs = seqmix_batch[0], generate = generate, lambda_ = lambda_, input_pos = input_pos[0], mix_pos = mix_pos[0])
                
                states_pred, (h_pred, c_pred) = self.decoder(
                  output_seqs_wo_end, mix_seqs = mix_output_seqs_wo_end, hidden_init = h_source, cell_init = c_source, lambda_ = lambda_
                )
                
            elif self.augmentation_type == 'seqmix_pos':
                (h_source, c_source) = self.encoder(input_seqs, mix_seqs = seqmix_batch[0], generate = generate, lambda_ = lambda_, input_pos = input_pos[0], mix_pos = mix_pos[0])
                
                states_pred, (h_pred, c_pred), reorder_mix_seqs = self.decoder(output_seqs_wo_end, mix_seqs = mix_output_seqs_wo_end, hidden_init = h_source, cell_init = c_source, lambda_ = lambda_, input_pos = input_pos[1], mix_pos = mix_pos[1])
            elif self.augmentation_type == 'seqmix_ind' or self.augmentation_type == 'seqmix_ind_2':
                (h_source, c_source) = self.encoder(input_seqs, mix_seqs = seqmix_batch[0], generate = generate, lambda_ = lambda_)
                
                states_pred, (h_pred, c_pred) = self.decoder(
                  output_seqs_wo_end, mix_seqs = mix_output_seqs_wo_end, hidden_init = h_source, cell_init = c_source, lambda_ = lambda_targ
                )
            elif self.augmentation_type == 'seqmix_ind_pos':
                (h_source, c_source) = self.encoder(input_seqs, mix_seqs = seqmix_batch[0], generate = generate, lambda_ = lambda_pos, input_pos = input_pos[0], mix_pos = mix_pos[0])
                
                states_pred, (h_pred, c_pred) = self.decoder(
                  output_seqs_wo_end, mix_seqs = mix_output_seqs_wo_end, hidden_init = h_source, cell_init = c_source, lambda_ = lambda_pos, input_pos = input_pos[1], mix_pos = mix_pos[1]
                )
                
            else:
              
                # pass input through encoder
                (h_source, c_source) = self.encoder(input_seqs, mix_seqs = seqmix_batch[0], generate = generate, lambda_ = lambda_)
                
                states_pred, (h_pred, c_pred) = self.decoder(
                  output_seqs_wo_end, mix_seqs = mix_output_seqs_wo_end, hidden_init = h_source, cell_init = c_source, lambda_ = lambda_
                )
              

        else:          # pass input through encoder
          (h_source, c_source) = self.encoder(input_seqs, input_pos = input_pos, generate = generate)
          
          # pass encoder hidden states and outputs through decoder
          states_pred, (h_pred, c_pred) = self.decoder(
              output_seqs_wo_end, hidden_init = h_source, cell_init = c_source, generate = generate
          )
        
        # unpack states, currently a packed padded seq
        # item [0] is hidden states over time (batch size x # tokens in target seq x hidden size)
        # item [1] is unpadded sequence lengths
        padded_states_pred = pad_packed_sequence(states_pred, batch_first=True)
         
        # put through linear layer to move from hidden size to output vocab size
        # dim of linear_out_pred: batch size x # tokens in target seq x target vocab size
        linear_out_pred = self.linear(padded_states_pred[0])
        
        # flatten / remove padding
        # padded_states[1] gives length of each sequence
        # dim of flattened_pred: # number of non-padded words in target x target vocab size
        flattened_pred = torch.cat([
            linear_out_pred[i][:padded_states_pred[1][i]] 
            for i in range(len(linear_out_pred))
        ])

        flattened_targets = torch.cat([
            output_seqs_wo_start[i][:padded_states_pred[1][i]] 
            for i in range(len(output_seqs_wo_start))
        ])
        
        # return (1) a flattened predictions tensor preds of size Mo*Vo and 
        #        (2) a flattened target words tensor targs of length Mo 
        #        (Note: targs should not contain any padding indices)
        
        if not generate and (self.augmentation_type == 'seqmix_pos'):
            flattened_mix_targets = torch.cat([
                reorder_mix_seqs[i][:padded_states_pred[1][i]] 
                for i in range(len(reorder_mix_seqs))
            ])
            
            return(flattened_pred, (flattened_targets, flattened_mix_targets), lambda_)
        
        elif not generate and (self.augmentation_type == 'seqmix_pos_ind'):
            flattened_mix_targets = torch.cat([
                reorder_mix_seqs[i][:padded_states_pred[1][i]] 
                for i in range(len(reorder_mix_seqs))
            ])
            return(flattened_pred, (flattened_targets, flattened_mix_targets), lambda_targ)
        elif not generate and (self.augmentation_type == 'seqmix_pos_ind_r'):
            
            return(flattened_pred, flattened_targets)
            
        elif not generate and (self.augmentation_type == 'seqmix_ind' or self.augmentation_type == 'seqmix_ind_2'):
            
            flattened_mix_targets = torch.cat([
                mix_output_seqs_wo_start[i][:padded_states_pred[1][i]] 
                for i in range(len(mix_output_seqs_wo_start))
            ])
            return(flattened_pred, (flattened_targets, flattened_mix_targets), lambda_targ)
        elif not generate and (self.augmentation_type == 'seqmix_ind_pos'):
            flattened_mix_targets = torch.cat([
                mix_output_seqs_wo_start[i][:padded_states_pred[1][i]] 
                for i in range(len(mix_output_seqs_wo_start))
            ])
            
            return(flattened_pred, (flattened_targets, flattened_mix_targets), lambda_pos)
            
        elif not generate and (self.augmentation_type == 'seqmix'):
            
            flattened_mix_targets = torch.cat([
                mix_output_seqs_wo_start[i][:padded_states_pred[1][i]] 
                for i in range(len(mix_output_seqs_wo_start))
            ])
            return(flattened_pred, (flattened_targets, flattened_mix_targets), lambda_)
            
        else:
            
            return(flattened_pred, flattened_targets)
    
        
    @torch.no_grad()
    def generate(self, source: torch.tensor, max_steps: int, eos_idx: int, bos_idx: int) -> torch.tensor:
        # the argument source will be a 1D tensor of size ti
        # you will return a 1D translation tensor with maximium length max_steps
        
        # pass source through encoder
        (h_last, c_last) = self.encoder(source, generate = True)
        
        # first token through decoder: beginning of sentence tag
        current_in = torch.Tensor([[bos_idx]]).long().to(device)
        # first token in sequence: end of sentence tag
        seq = [bos_idx]

        while (len(seq) < max_steps):
            # pass current input and encoder states to decoder
            state_last, (h_last, c_last) = self.decoder(
                current_in, hidden_init = h_last.to(device), cell_init = c_last.to(device), generate = True
            )
            
            # predict next token
            pred = self.linear(h_last)
            logsoftpred = self.logsoft(pred[0])
            idxpred = int((logsoftpred[0][1:].max(0))[1]) + 1
            
            # next token through decoder: prediction
            current_in = torch.Tensor([[idxpred]]).long()
            
            # add token to sequence
            seq.append(idxpred)

            if idxpred == eos_idx: # end if we predict end of sentence or seq too long
                break
            if idxpred == 1:
                break

        return seq
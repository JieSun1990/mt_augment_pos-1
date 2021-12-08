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
from embeddingTF import *

### Full Encoder ###
### Takes in all relevant augmentation parameters for input- and embeddings-level augmentations
class FullEncoder(nn.Module):
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
	        augmentation_type: str = None,  # one of "swap", "drop", "blank", "smooth", "smooth_pos", "lmsample", "lmsample_pos", "soft", and "soft_pos"
	        gamma = None, 					# probability that single token is augmented
	        k = None,                       # window size for "swap" method
	        unk_idx = None,                 # placeholder for "blank" method
	        unigram_freq = None,            # to sample from with "smooth" method
	        language_model = None,          # to sample from in "lmsample" method
	        source_pos_idx = None			# Part of speech indices corresponding to 
	    ):
	    super(FullEncoder, self).__init__()

	    self.input_size = vocab_size
	    self.hidden_size = d_model
	    self.padding_idx = padding_idx
	    self.device = device
	    self.augmentation_type = augmentation_type
	    self.gamma = gamma
	    self.k = k
	    self.unk_idx = unk_idx
	    self.unigram_freq = unigram_freq
	    self.language_model = language_model
	    self.source_pos_idx = source_pos_idx
	    # Embedder
	    self.embedding = Embedder(vocab_size, d_model)

	    # Positional Encoder
	    self.positional_encoder = PositionalEncoding(d_model)

	    # Encoder Layers
	    encoder_layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)

	    # Encoder
	    self.encoder = Encoder(encoder_layer,N)

	def forward(self,
              # regular foward args
              input_seqs,
              src_mask,
              # augmentation forward args
              generate = False,
              lambda_ = None,
              mix_seqs = None, 
              input_pos = None, 
              mix_pos = None): 


		lengths = torch.Tensor([int(sum(x != self.padding_idx)) for x in input_seqs])

		
		# Input level augmentation
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

		# Get embeddings
		embedded_input_seqs = self.embedding(input_seqs)

		# Embedding level augmentation
		if not generate and self.augmentation_type == 'soft':
			embedded_input_seqs = self.augment_soft(input_seqs, embedded_input_seqs, lengths)
		elif not generate and self.augmentation_type == 'soft_pos':
			embedded_input_seqs = self.augment_soft_pos(input_seqs, input_pos, embedded_input_seqs, lengths)


		# Positional encoding
		embedded_input_seqs = self.positional_encoder(embedded_input_seqs)

		# Encoding
		encoded_input_seqs = self.encoder(embedded_input_seqs, src_mask)
		return(encoded_input_seqs)

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
						np.exp(self.language_model.generate(input_seqs[i][:idx].to('cpu').unsqueeze(0)).to('cpu'))[0], 
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
	### Utility functions for augmentation methods ###

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

	def select_from_unigram(self, freq_dist, n):
		return torch.Tensor(
			np.random.choice(
				list(range(len(freq_dist))), 
				n, replace = True,
				p = freq_dist
			)
		)
	def replace_tokens(self, this_seq, to_replace_idx, replace_with_val):
		for idx, val in zip(to_replace_idx, replace_with_val):
			this_seq[idx] = val
		return this_seq

	def select_from_unigram_pos(self, freq_dist, n, pos):
		select = self.source_pos_idx[int(pos)]
		if sum(select) > 0:
			freq_dist = np.array(freq_dist)*np.array(select)
			freq_dist = freq_dist/sum(freq_dist)

		return torch.Tensor(
			np.random.choice(
				list(range(len(freq_dist))), 
				n, replace = True,
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

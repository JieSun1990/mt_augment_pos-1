import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from typing import List, Tuple, Dict
PARENT_DIR = '/content/gdrive/MyDrive/CS287_Research_Project/Jennas_Code/'


class Autoreg_LM(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, padding_idx: int, device):
        super().__init__()
        
        self.input_size = input_size        # vocab size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.embedding_size = hidden_size   # embedding size was not specified so I made it same as hidden size
        
        self.embedding = torch.nn.Embedding(
            num_embeddings = self.input_size,
            embedding_dim = self.embedding_size,
            padding_idx = PADDING_IDX
        )

        self.lstm = torch.nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = self.embedding_size,
            batch_first = True,
            num_layers = 2,
            bidirectional = True
        )

        self.linear = torch.nn.Linear(
            self.embedding_size*8,
            self.input_size # value for each word
        )

        self.logsoft = nn.LogSoftmax(dim = 1)

    def split_one_seq(self, seq):
      contexts = []
      targets = []
      for i in range(1,len(seq)):
        zeros = torch.Tensor([0]*(int(len(seq) - i))).to(device)
        contexts.append(torch.cat([seq[:i], zeros]))
        targets.append(int(seq[i]))

      contexts = torch.stack(contexts)
      return contexts, targets

    def split_seqs(self, input_seqs):
      contexts = []
      targets = []
      for seq in input_seqs:
        this_context, this_target = self.split_one_seq(seq)
        contexts.append(this_context)
        targets.append(this_target)
      contexts = torch.stack(contexts)
      targets = torch.Tensor(targets)
      return contexts, targets

    def flatten(self, contexts, targets, padding_idx):
      contexts_flat = torch.flatten(contexts, end_dim = 1)
      targets_flat = torch.flatten(targets)

      filter = (targets_flat != padding_idx)

      return contexts_flat[filter,:].long(), targets_flat[filter]

    def forward(self, input_seqs) -> Tuple[torch.tensor, torch.tensor]: 
        context, target = self.split_seqs(input_seqs)
        context_flat, target_flat = self.flatten(context, target, PADDING_IDX)
        
        lengths = torch.Tensor([int(sum(x != self.padding_idx)) for x in context_flat])

        embedded = self.embedding(context_flat.long())

        packed_padded_seqs = torch.nn.utils.rnn.pack_padded_sequence(
          embedded, batch_first = True, lengths = lengths,
          enforce_sorted=False
        )

        lstm_out, (h, c) = self.lstm(packed_padded_seqs)

        # concatenate hidden and cell states of both directions and both layers
        meaning = torch.cat([h[0], h[1], h[2], h[3], c[0], c[1], c[2], c[3]], dim = 1)

        linear_out = self.linear(meaning)

        return linear_out, target_flat.to(device)

    def generate(self, input_seqs):
      with torch.no_grad():
        lengths = torch.Tensor([int(sum(x != self.padding_idx)) for x in input_seqs])

        embedded = self.embedding(input_seqs.long())

        packed_padded_seqs = torch.nn.utils.rnn.pack_padded_sequence(
          embedded, batch_first = True, lengths = lengths,
          enforce_sorted=False
        )

        lstm_out, (h, c) = self.lstm(packed_padded_seqs)

        # concatenate hidden and cell states of both directions and both layers
        meaning = torch.cat([h[0], h[1], h[2], h[3], c[0], c[1], c[2], c[3]], dim = 1)

        linear_out = self.linear(meaning)
        pred = self.logsoft(linear_out)
        return pred

def generate(self, input_seqs):
  with torch.no_grad():
    lengths = torch.Tensor([int(sum(x != self.padding_idx)) for x in input_seqs])

    embedded = self.embedding(input_seqs.long())

    packed_padded_seqs = torch.nn.utils.rnn.pack_padded_sequence(
      embedded, batch_first = True, lengths = lengths,
      enforce_sorted=False
    )

    lstm_out, (h, c) = self.lstm(packed_padded_seqs)

    # concatenate hidden and cell states of both directions and both layers
    meaning = torch.cat([h[0], h[1], h[2], h[3], c[0], c[1], c[2], c[3]], dim = 1)

    linear_out = self.linear(meaning)
    pred = self.logsoft(linear_out)
    return pred

def load_lm(PARENT_DIR = PARENT_DIR):
  lm = torch.load(PARENT_DIR+'jennas_lm')

  # I originally messed up the generate function and didn't 
  # want to wait another 2 hours for this to train again. 
  # My hacky workaround was to create a bound method
  lm.generate = generate.__get__(lm)

  return lm






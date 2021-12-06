## run this before importing file
# !pip install -U spacy

import spacy
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import collections

### constants ###

PARENT_DIR = '/content/gdrive/MyDrive/CS287_Research_Project/Jennas_Code/' # for google colab. adjust accordingly
n_extra_tokens = 4
UNK_IDX = 0 
PADDING_IDX = 1 
BOS_IDX = 2
EOS_IDX = 3

### classes ###
class TranslationDataset(Dataset):   
    def __init__(
      self, source: List[List[str]], target: List[List[str]], 
      source_nlp, target_nlp,
      pos_map=None, min_freq=3, source_vocab=None, target_vocab=None
    ):
        super().__init__()
        
        if source_vocab is None and target_vocab is None:
            self.source_vocab, self.source_vocab_size = generate_vocab(get_word_counts(source), min_freq)
            self.target_vocab, self.target_vocab_size = generate_vocab(get_word_counts(target), min_freq)
        else:
            self.source_vocab = source_vocab
            self.source_vocab_size = len(source_vocab)

            self.target_vocab = target_vocab
            self.target_vocab_size = len(target_vocab)

        self.source_nlp = source_nlp
        self.target_nlp = target_nlp

        self.source, self.target, self.source_pos, self.target_pos, self.pos_map = self._get_idx_dataset(
            source, target, self.source_vocab, self.target_vocab, pos_map
        )
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return torch.tensor(self.source[idx]), torch.tensor(self.target[idx]), torch.tensor(self.source_pos[idx]), torch.tensor(self.target_pos[idx]) # pos lists are strings so we can't make them tensors

    def _get_idx_dataset(self, source: List[List[str]], target: List[List[str]], source_vocab: Dict[str, int], target_vocab: Dict[str, int], pos_map):
        source_toks = []
        target_toks = []
        source_pos = []
        target_pos = []

        if pos_map is None:
            pos_map = {'X':0}
            build_pos_map = True
        else:
            build_pos_map = False

        self.source_vocab_to_tags = {v:[] for k, v in source_vocab.items()}
        self.target_vocab_to_tags = {v:[] for k, v in target_vocab.items()}

        for src, tar in zip(source, target):
            source_toks.append([source_vocab["<BOS>"]] + [source_vocab[token] if token in source_vocab else source_vocab["<UNK>"] for token in src] + [source_vocab["<EOS>"]])
            target_toks.append([target_vocab["<BOS>"]] + [target_vocab[token] if token in target_vocab else target_vocab["<UNK>"] for token in tar] + [target_vocab["<EOS>"]])
            source_nlps = self.source_nlp(' '.join(src))
            target_nlps = self.target_nlp(' '.join(tar))

            for i in range(len(src)):
                if build_pos_map and source_nlps[i].pos_ not in pos_map.keys():
                    pos_map[source_nlps[i].pos_] = max(pos_map.values()) + 1

                if src[i] in source_vocab.keys():
                  self.source_vocab_to_tags[
                      source_vocab[src[i]]      # index of ith word in source
                  ].append(pos_map[source_nlps[i].pos_]) # ith part of speech tag in source

                else:
                  self.source_vocab_to_tags[UNK_IDX].append(pos_map[source_nlps[i].pos_])

            for i in range(len(tar)):
                if build_pos_map and target_nlps[i].pos_ not in pos_map.keys():
                    pos_map[target_nlps[i].pos_] = max(pos_map.values()) + 1
                
                if tar[i] in target_vocab.keys():
                  self.target_vocab_to_tags[
                      target_vocab[tar[i]]      # index of ith word in target
                  ].append(pos_map[target_nlps[i].pos_]) # ith part of speech tag in target

                else:
                   self.target_vocab_to_tags[UNK_IDX].append(pos_map[target_nlps[i].pos_])


            source_pos.append([pos_map['X']] + [pos_map[s.pos_] for s in source_nlps] + [pos_map['X']])
            target_pos.append([pos_map['X']] + [pos_map[t.pos_] for t in target_nlps] + [pos_map['X']])
        
        for key in self.source_vocab_to_tags.keys():
          counts = collections.Counter(self.source_vocab_to_tags[key])
          counts['X'] = 0
          tag_final = max(counts, key=counts.get)
          self.source_vocab_to_tags[key] = tag_final

        for key in self.target_vocab_to_tags.keys():
          counts = collections.Counter(self.target_vocab_to_tags[key])
          counts['X'] = 0
          tag_final = max(counts, key=counts.get)
          self.target_vocab_to_tags[key] = tag_final


        self.source_pos_idx = {}
        self.target_pos_idx = {}

        for pos in pos_map.values():
          self.source_pos_idx[pos] = [int(v == pos) for v in self.source_vocab_to_tags.values()]
          self.target_pos_idx[pos] = [int(v == pos) for v in self.target_vocab_to_tags.values()]

        return source_toks, target_toks, source_pos, target_pos, pos_map


### functions ###  

def pad_collate(batch: List[Tuple[torch.tensor, torch.tensor]]) -> Tuple[torch.tensor, List, torch.tensor]:
    sources = pad_sequence([tup[0] for tup in batch], 
                           batch_first = True, padding_value = PADDING_IDX)
    targets = pad_sequence([tup[1] for tup in batch], 
                           batch_first = True, padding_value = PADDING_IDX)
    sources_pos = pad_sequence([tup[2] for tup in batch], 
                           batch_first = True, padding_value = PADDING_IDX)
    targets_pos = pad_sequence([tup[3] for tup in batch], 
                           batch_first = True, padding_value = PADDING_IDX)
    return (sources, targets, sources_pos, targets_pos)

def get_word_counts(reviews: List[List[str]]) -> Dict[str, int]:
    """
    Given a tokenized corpus (in this case reviews), we count the frequency of
    each word in the corpus
    """
    word_counts = {}
    for review in reviews:
        for word in review:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    return word_counts

def generate_vocab(word_counts: Dict[str, int], min_freq: int) -> Tuple[Dict[str, int], int]:
    """
    Given a set of word counts, we generate a vocabulary. We return two things
    from this method:

        1. A dict mapping tokens to indices
        2. THe length of the vocab
    
    Words that occur fewer than `min_freq` are replaced with <UNK>
    """

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: i+4 for i, (word, count) in enumerate(sorted_words) if count > min_freq}
    vocab["<PAD>"] = PADDING_IDX
    vocab["<EOS>"] = EOS_IDX 
    vocab["<UNK>"] = UNK_IDX
    vocab["<BOS>"] = BOS_IDX 
    return vocab, len(vocab)

def load_dictionaries(small = True):
  # Load small dataset: 10%
  if (small):
    end = '_small'
  else:
    end = ''
  with open(PARENT_DIR+"data/iwlst_train_sentences" + end + ".pkl", "rb") as f: # Can edit based on need
      train_dict = pickle.load(f)
  with open(PARENT_DIR+"data/iwlst_valid_sentences" + end + ".pkl", "rb") as f:
      val_dict = pickle.load(f)
  with open(PARENT_DIR+"data/iwlst_test_sentences" + end + ".pkl", "rb") as f:
      test_dict = pickle.load(f)

  return train_dict, val_dict, test_dict
    
def build_dataloaders(small = True, batch_size = 1):

  import spacy.cli
  spacy.cli.download("de_core_news_sm")
  spacy.cli.download("en_core_web_sm")

  source_nlp = spacy.load("de_core_news_sm")
  target_nlp = spacy.load("en_core_web_sm") 

  train_dict, val_dict, test_dict = load_dictionaries(small = small)

  mt_train_ds = TranslationDataset(
      train_dict["de"], train_dict["en"],
      source_nlp = source_nlp,
      target_nlp = target_nlp
  )
  mt_test_ds = TranslationDataset(
      test_dict["de"], test_dict["en"], 
      source_vocab=mt_train_ds.source_vocab, 
      target_vocab=mt_train_ds.target_vocab,
      pos_map = mt_train_ds.pos_map,
      source_nlp = source_nlp,
      target_nlp = target_nlp
  )
  mt_train_dl = DataLoader(
      mt_train_ds, collate_fn = pad_collate, num_workers=0, shuffle=True, batch_size=batch_size
  )
  mt_test_dl = DataLoader(
      mt_test_ds, collate_fn = pad_collate, num_workers=0, shuffle=True, batch_size=batch_size
  )
  
  if (small):
    mt_val_ds = TranslationDataset(
        val_dict["de"], val_dict["en"], 
        source_vocab=mt_train_ds.source_vocab, 
        target_vocab=mt_train_ds.target_vocab,
        pos_map = mt_train_ds.pos_map,
        source_nlp = source_nlp,
        target_nlp = target_nlp
    )
    
    mt_val_dl = DataLoader(
      mt_val_ds, collate_fn = pad_collate, num_workers=0, shuffle=True, batch_size=batch_size
    )

    return mt_train_ds, mt_val_ds, mt_test_ds, mt_train_dl, mt_val_dl, mt_test_dl

  else:
    return mt_train_ds, mt_test_ds, mt_train_dl, mt_test_dl

def load_pickled_dataloaders(parent_dir = PARENT_DIR, small = True, batch1 = True):
  if small:
    if batch1:
      folder = "dataloaders10perc"
    else: 
      folder = "dataloaders10perc_batchsize32"
  else:
    folder = "dataloaders_full"

  with open(parent_dir+'data/'+folder+'/mt_train_ds.pickle', 'rb') as handle:
      mt_train_ds = pickle.load(handle)

  with open(parent_dir+'data/'+folder+'/mt_test_ds.pickle', 'rb') as handle:
      mt_test_ds = pickle.load(handle)

  with open(parent_dir+'data/'+folder+'/mt_train_dl.pickle', 'rb') as handle:
      mt_train_dl = pickle.load(handle)

  with open(parent_dir+'data/'+folder+'/mt_test_dl.pickle', 'rb') as handle:
      mt_test_dl = pickle.load(handle)

  if small:
    with open(parent_dir+'data/'+folder+'/mt_val_dl.pickle', 'rb') as handle:
      mt_val_dl = pickle.load(handle)

    with open(parent_dir+'data/'+folder+'/mt_val_ds.pickle', 'rb') as handle:
      mt_val_ds = pickle.load(handle)

    return mt_train_ds, mt_val_ds, mt_test_ds, mt_train_dl, mt_val_dl, mt_test_dl
  
  return mt_train_ds, mt_test_ds, mt_train_dl, mt_test_dl


## Code to load datasets and save small to pickles
## takes about 5 minutes

# mt_train_ds, mt_val_ds, mt_test_ds, mt_train_dl, mt_val_dl, mt_test_dl = build_dataloaders()

# with open(PARENT_DIR+'/data/dataloaders10perc/mt_train_ds.pickle', 'wb') as handle:
#     pickle.dump(mt_train_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc/mt_test_ds.pickle', 'wb') as handle:
#     pickle.dump(mt_test_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc/mt_val_ds.pickle', 'wb') as handle:
#     pickle.dump(mt_val_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc/mt_train_dl.pickle', 'wb') as handle:
#     pickle.dump(mt_train_dl, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc/mt_test_dl.pickle', 'wb') as handle:
#     pickle.dump(mt_test_dl, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc/mt_val_dl.pickle', 'wb') as handle:
#     pickle.dump(mt_val_dl, handle, protocol=pickle.HIGHEST_PROTOCOL)

## Code to load datasets and save small with batch size 50 to pickles
## takes about 5 minutes

# mt_train_ds, mt_val_ds, mt_test_ds, mt_train_dl, mt_val_dl, mt_test_dl = build_dataloaders(batch_size = 32)

# with open(PARENT_DIR+'/data/dataloaders10perc_batchsize32/mt_train_ds.pickle', 'wb') as handle:
#     pickle.dump(mt_train_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc_batchsize32/mt_test_ds.pickle', 'wb') as handle:
#     pickle.dump(mt_test_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc_batchsize32/mt_val_ds.pickle', 'wb') as handle:
#     pickle.dump(mt_val_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc_batchsize32/mt_train_dl.pickle', 'wb') as handle:
#     pickle.dump(mt_train_dl, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc_batchsize32/mt_test_dl.pickle', 'wb') as handle:
#     pickle.dump(mt_test_dl, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders10perc_batchsize32/mt_val_dl.pickle', 'wb') as handle:
#     pickle.dump(mt_val_dl, handle, protocol=pickle.HIGHEST_PROTOCOL)

## Code to load datasets and save full to pickles
## takes about 35 minutes

# mt_train_ds, mt_test_ds, mt_train_dl, mt_test_dl = build_dataloaders(small = False)

# with open(PARENT_DIR+'/data/dataloaders_full/mt_train_ds.pickle', 'wb') as handle:
#     pickle.dump(mt_train_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders_full/mt_test_ds.pickle', 'wb') as handle:
#     pickle.dump(mt_test_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders_full/mt_train_dl.pickle', 'wb') as handle:
#     pickle.dump(mt_train_dl, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(PARENT_DIR+'/data/dataloaders_full/mt_test_dl.pickle', 'wb') as handle:
#     pickle.dump(mt_test_dl, handle, protocol=pickle.HIGHEST_PROTOCOL)
  


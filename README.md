# mt_augment_pos
Machine Translation Data Augmentation Methods Maintaining Part of Speech


# files and descriptions
- notebooks obtain the data:
  - `Download_dataset_iwslt2017.ipynb`: download and produce 10% of the data for the paper
- notebooks to train models:
  - `TrainLSTM.ipynb`: all LSTM methods excluding sequence matching methods
  - `similarity_ds_k=2.ipynb` and `similarity_ds_k=10.ipynb`: LSTM sequence matching methods using similarity
  - `TrainTransformer.ipynb`: all Transformer methods
  - `LanguageModel.ipynb`: training the language model used in `LMsample` and `soft` methods
- other notebooks:
  - `BEAM_BLEU.ipynb`: *evaluation*, re-compute BLEU score with beam search, compute POS BLEU score
  - `LM_POS_Experiments.ipynb`: *experiment*, looking at how well the language model matches part of speech
  - `CustomTransformer.ipynb`: *development*, developing and testing the transformer architecture, contains links to transformer resources
- functions for transformer models:
  - `embeddingTF.py`: `Embedder` and `PositionalEncoding`
  - `sublayersTF.py`: `SublayerConnection` (layer norm & residual connection), `FeedForward`, `attention`, `MultiHeadedAttention`, and `clones` (replicates layers)
  - `layersTF.py`: `EncoderLayer` and `DecoderLayer`
  - `stacksTF.py`: `Encoder` and `Decoder`, which construct the encoder and decoder stacks from the encoder and decoder layers, respectively
  - `encoderTF.py`: `FullEncoder`, which allows for augmentation to occur in the embedding - positional encoding - encoder structure
  - `decoderTF.py`: `FullDecoder`, which allows for augmentation to occur in the embedding - positional encoding - decoder structure
  - `seq2seqTF.py`: `Seq2SeqTF`, which contains the custom encoder and decoders and fully defines the transformer seq2seq model
  - `batchTF.py`: `BatchTF`, which formats source and target inputs to yield shifted targets, source mask, and target mask (`future_mask` provides decoder-specific masking)
  - `trainTF.py`: `train`, which uses `train_epoch` and `val_epoch` to create the training scheme, `greedy_decode`, and `translate_corpus`
- functions for lstm models:
  - `train.py`
  - `Seq2Seq.py`
  - `EncoderLSTM.py`
  - `DecoderLSTM.py`
- other functions:
  - `train.py`: training loop and translating corpus
  - `load_data.py`: creating and loading pickled datasets and dataloaders
  - `similarity_load_data.py`: .
  - `load_lm.py`: load the language model developed in `LanguageModel.ipynb`

# data access

## option 1. download and re-build dataloaders
- Download the full data from torchtext:

`from torchtext.datasets import IWSLT2017`

`train_iter, valid_iter, test_iter = IWSLT2017(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en'))`

- Run `Download_dataset_iwslt2017.ipynb` to get 10% sample of dataset and save as pickles
- Run `load_and_save(batch1 = True)` from `load_data.py` to build the dataloaders used in our LSTM models and save to pickle files
- Run `load_and_save(batch1 = False)` from `load_data.py` to build the dataloaders used in our Transformer models and save to pickle files
- In our code, we use `load_pickled_dataloaders(batch1 = True)` and `load_pickled_dataloaders(batch1 = False)` from `load_data.py` to load dataloaders from the pickle files. You'll need to pass in `PARENT_DIR` as the location of your `data` folder.

## option 2. use our pre-built dataloaders
- Save download the following directories and save to your own `data` folder
  - `dataloaders10perc`: used for LSTM models and the LM https://drive.google.com/drive/folders/18K6XpYgTmLZkPtLQw4-8gqUyeAGWPF-u?usp=sharing
  - `dataloaders10perc_batchsize32`: used for transformer models, larger batch size https://drive.google.com/drive/folders/16_hx53i473FjJfn4sfdLQ4ZTdxDehUBT?usp=sharing
- In our code, we use `load_pickled_dataloaders(batch1 = True)` and `load_pickled_dataloaders(batch1 = False)` from `load_data.py` to load dataloaders from the pickle files for the LSTM and transformer, respectively. You'll need to pass in `PARENT_DIR` as the location of your `data` folder.



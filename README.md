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
  - `Transformer.ipynb`: *development*, developing and testing the transformer architecture
- functions for transformer models:
  - `trainTF.py`
  - `seq2seqTF.py`
  - `embeddingTF.py`
  - `layersTF.py`
  - `sublayersTF.py`
  - `stacksTF.py`
  - `encoderTF.py`
  - `decoderTF.py`
  - `batchTF.py`
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
- Download the full data from torchtext:

`from torchtext.datasets import IWSLT2017`

`train_iter, valid_iter, test_iter = IWSLT2017(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en'))`

- Run `Download_dataset_iwslt2017.ipynb` to get 10% sample of dataset and save as pickles
- Run `build_dataloaders` from `load_data.py` to build the dataloaders used in our training loop and save to pickle files
- Run `load_pickled_dataloaders` from `load_data.py` to load dataloaders from the pickle files

# GPT-1
This repository contains a pytorch implementation of the GPT-1 model introduced
by OpenAI in the paper [Improving Language Understanding with Unsupervised
Learning][1]. This repository contains source code for the model as well as code
for preprocessing training data and the pre-training/fine-tuning process.

## Setup
All required modules for running the code is in the `env.yml` file in the
*confs* directory. You can create the conda environment with:
    
    conda env create -f confs/env.yml

## Data Collection
The original BookCorpus data set used to pretrain GPT is no longer distributed.
However, [this][2] repository provides several resources for recreating or
downloading a similar data set.

## Preprocessing
GPT uses [byte pair encodings][3] to tokenize. The *preprocessing* directory
contains a script for training a byte pair encoding tokenizer (`train_bpe.py`)
and another script for tokenizing a dataset using the trained tokenizer
(`tokenize_dataset.py`).

### Training the tokenizer
The `train_bpe.py` script takes an input file containing a list of filepaths to
text files to be trained on. I used a randomly selected 10% sample of my
downloaded BooksCorpus dataset (about 1700 books). You can create the required
input file using the following command: 
    
    find [BookCorpus filepath]/epubtxt -iname "*.txt" | shuf | head -n 1700 >  files.txt

Then the tokenizer can be trained as follows:

    python -m preprocessing.train_bpe -i files.txt \
                                      -o checkpoints/tokenizer \
                                      -m 40000 \
                                      -n 10

This trains a byte pair encoding tokenizer with 40000 merges, disregarding any
vocabulary words that appear with a frequency less than 10 in the dataset.

### Tokenizing the dataset
The `tokenize_dataset.py` script takes in an input file containing a list of
filepaths to text files to tokenize. You can create the required input file
using the following command:

    find [BookCorpus filepath]/epubtxt -iname "*.txt" >  files.txt

Next, tokenize the dataset:

    python -m preprocessing.tokenize_dataset -c checkpoints/tokenizer \
                                             -i files.txt \
                                             -o data/pretrain/tokenized \
                                             -l 1024 \
                                             -j 8

This command tokenizes the files in `files.txt` and places the tokenized
versions in the *tokenized* directory. Lines are split into particular length
specified by the `-l` flag. During training, the `TokenIDDataset` class returns
random sequence-size segments of each line, so be sure to set your line length
to be greater than the sequence size you intend to use in your model instance.

## Training
Both pre-training and fine-tuning can be performed with the `train.py` script.
If a checkpoint directory is specified with the `-ch` flag, training will
continue from that checkpoint.

The parameters I used for pretraining are in the `pretrain.yml` file in the
*confs* directory. All model parameters are the same as mentioned in [Improving
Language Understanding with Unsupervised Learning][1], with the exception of
sequence size and batch size. I sequence size to 128 rather than 512, and batch
size to 32 rather than 64 in order to train on a single GPU. 

## Text Generation
Text generation is implemented using top-k sampling and can be performed with
the `generate.py` script. All generation parameters are located in the
`generate.yml` file in the *confs* folder.

[1]: https://openai.com/blog/language-unsupervised/
[2]: https://github.com/soskek/bookcorpus
[3]: https://arxiv.org/abs/1508.07909

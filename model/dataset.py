"""
Dataset class for BooksCorpus movie review dataset
"""
from typing import Dict, List

from spacy.tokenizer import Tokenizer
from torch.utils.data import Dataset
from spacy.lang.en import English
from tqdm import trange, tqdm
import torch


def freqs_to_maps(path: str, vocab: int, unk: str, 
                  pad: str) -> (Dict[str, int], Dict[int, str]):
    """ Create two maps from file of word frequencies: 1 mapping words to
        indices, and one mapping indices to words 

    Args:
        path: path to file containing frequencies
        vocab: size of vocabulary
        unk: token to use for unknown words
        pad: token to use for padding

    Return:
        Dict[str, int]: dict mapping tokens to indices
        Dict[str, int]: dict mapping indices to tokens
    """

    vmap, imap, count = {}, {}, 0
    with open(path, 'r') as infile:

        line = infile.readline()
        while line:
            word, freq = line.split('\t')

            if count == vocab-2: 
                break

            line = infile.readline()
            vmap[word] = count
            imap[count] = word
            count += 1

    vmap[unk] = count
    imap[count] = unk
    vmap[pad] = count+1
    imap[count+1] = pad
    return vmap, imap


def sopen(filepath: str, linecount: int) -> List[str]:
    """ Reads in all lines from filepath with progress bar

    Args:
        filepath: file path of file to read from
        linecount: length of file in lines

    Return:
        (List[str]): lines from file
    """

    lines = []
    progress = tqdm(total=linecount, desc='Loading data')
    with open(filepath, 'r') as infile:

        line = infile.readline()
        while line:
            lines.append(line)
            line = infile.readline()
            progress.update(1)

    return lines


class BooksCorpusTokenizer:


    def __init__(self, countpath: str, window: int, vocab: int, start: str, 
                 unk: str, pad: str, end: str) -> 'BooksCorpusTokenizer':
        """ Tokenizer for BooksCorpus dataset

        Args:
            countpath: path of file containing term frequencies
            window: number of tokens in each window
            vocab: size of vocabulary
            start: token for start of document
            unk: token for unknown terms
            pad: token for padding
            end: token for end of document

        Return:
            (BooksCorpusTokenizer): tokenizer instance
        """
        self.vmap, self.imap = freqs_to_maps(countpath, vocab, unk, pad)
        self.tokenizer = Tokenizer(English().vocab)
        self.window = window
        self.start = start
        self.vocab = vocab
        self.unk = unk
        self.pad = pad 
        self.end = end


    def tokenize_line(self, line: str) -> (torch.Tensor, torch.Tensor):
        """ Tokenize a single line of text

        Args:
            line: line of text to be tokenized

        Return:
            (torch.Tensor): input sequence
            (torch.Tensor): final term in sequence
        """

        tokens = [
            t.text if t.text in self.vmap else self.unk 
            for t in self.tokenizer(line)
        ]

        if len(tokens) < self.window + 1:
            padding = [self.pad for i in range(self.window+1-len(tokens))]
            tokens = padding + tokens

        ids = torch.Tensor([self.vmap[t] for t in tokens])
        ids = ids.type(torch.LongTensor)
        xids, yid = ids[:-1], ids[-1]
        xids = torch.nn.functional.one_hot(xids, self.vocab)

        return xids, yid


    def tokenize(self, lines: List[str]) -> (torch.Tensor, torch.Tensor):
        """ Tokenize batch of lines

        Args:
            lines: list of lines to be tokenized

        Return:
            (torch.Tensor): batch of input sequences
            (torch.Tensor): batch of correcponding final terms for sequences
        """

        X = torch.empty((len(lines), self.window, self.vocab))
        Y = torch.empty((len(lines)))

        for i in range(len(lines)):
            x, y = self.tokenize_line(lines[i])
            X[i], Y[i] = x, y

        return X, Y

    
    def decode_line(self, x: torch.Tensor) -> List[str]:
        """ Turn one-hot vectors into tokens

        Args:
            x: one-hot vectors to retrieve tokens for

        Return:
            (List[str]): list of corresponding tokens for vector
        """
        ids = torch.argmax(x, dim=1)
        tokens = [self.imap[i.item()] for i in ids]
        return tokens


    def decode(self, x: torch.Tensor) -> List[List[str]]:
        """ Turn batch of one-hot vectors into tokens

        Args:
            x: batch of one-hot vectors to retrieve tokens for

        Return:
            (List[List[str]]): list of lists of tokens for batch of one-hot
                               vectors
        """

        lines = []

        for i in range(x.shape[0]):
            lines.append(self.decode_line(x[i]))

        return lines


class BooksCorpus(Dataset):


    def __init__(self, datapath: str, linecount: int=None) -> 'BooksCorpus':
        """ Dataset class for BooksCorpus dataset

        Args:
            datapath: filepath to dataset
            linecount: number of lines in dataset

        Return:
            (BooksCorpus): instance of dataset 
        """
        super(Dataset, self).__init__()

        if linecount is None:
            self.data = open(datapath, 'r').readlines()

        else:
            self.data = sopen(datapath, linecount)


    def __len__(self) -> int:
        """ Return length of dataset

        Return:
            (int): length of dataset
        """

        return len(self.data)


    def __getitem__(self, idx: int) -> str:
        """ Return item at specified index of dataset

        Args:
            idx: index of item in dataset

        Return:
            (str): line of dataset at provided index
        """

        return self.data[idx].strip()



from abc import ABC, abstractmethod
from typing import Dict, List

from spacy.tokenizer import Tokenizer as Tok
from spacy.lang.en import English
import torch


class Tokenizer(ABC):

    @abstractmethod
    def tokenize_line(self):
        pass

    @abstractmethod
    def tokenize(self):
        pass


class BooksCorpusTokenizer(Tokenizer):


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
        self.tokenizer = Tok(English().vocab)
        self.window = window
        self.start = start
        self.vocab = vocab
        self.unk = unk
        self.pad = pad 
        self.end = end


    def tokenize_line(self, line: str) -> torch.Tensor:
        """ Tokenize a single line of text

        Args:
            line: line of text to be tokenized

        Return:
            (torch.Tensor): input sequence ids
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
        ids = torch.nn.functional.one_hot(ids, self.vocab)

        return ids


    def tokenize(self, lines: List[str]) -> (torch.Tensor, torch.Tensor):
        """ Tokenize batch of lines

        Args:
            lines: list of lines to be tokenized

        Return:
            (torch.Tensor): batch of input sequences
        """

        X = torch.empty((len(lines), self.window+1, self.vocab))

        for i in range(len(lines)):
            x = self.tokenize_line(lines[i])
            X[i] = x

        return X


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

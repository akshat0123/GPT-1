"""
Dataset class for aclImdb movie review dataset
"""

from torch.utils.data import Dataset
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import torch


class ReviewDataset(Dataset):


    def __init__(self, dpath, vpath, vsize, start_token='<START>',
                 end_token='<END>', pad_token='<PAD>', limit=128):

        self.tokenizer = Tokenizer(English().vocab)
        self.data = open(dpath, 'r').readlines()

        vocab = [word.split(' ')[0] for word in open(vpath, 'r').readlines()]
        vocab = vocab[:vsize]

        self.start = start_token
        self.end = end_token
        self.pad = pad_token
        self.limit = limit

        self.vmap = { vocab[i]: i for i in range(len(vocab)) }
        self.vmap[self.end] = len(vocab) + 1
        self.vmap[self.pad] = len(vocab) + 2
        self.vmap[self.start] = len(vocab)


    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        path = self.data[idx].strip()
        text = open(path, 'r').read()
        tokens = [token.text for token in self.tokenizer(text)]
        ids = [self.vmap[token] for token in tokens if token in self.vmap]
        ids = [self.vmap[self.start]] + ids + [self.vmap[self.end]]

        if len(ids) > self.limit+1:
            ids = ids[:self.limit+1]

        elif len(ids) < self.limit+1:
            ids = [self.vmap[self.pad] for i in range(self.limit+1-len(ids))] + ids

        ids = torch.Tensor(ids)
        ids = ids.type(torch.LongTensor)
        xids, yid = ids[:-1], ids[-1]

        return xids, yid

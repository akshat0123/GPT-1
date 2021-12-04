"""
Dataset class for aclImdb movie review dataset
"""

from torch.utils.data import Dataset
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


class ReviewDataset(Dataset):

    def __init__(self, dpath, vpath, size=20000):
        self.data = open(dpath, 'r').readlines()
        self.tokenizer = Tokenizer(English().vocab)
        self.vocab = open(vpath, 'r').readlines()[:size]
        self.vocab = set([line.split(' ')[0] for line in self.vocab])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        path = self.data[idx].strip()
        text = open(path, 'r').read()
        tokens = [token.text for token in self.tokenizer(text)]
        tokens = [token for token in tokens if token in self.vocab]
        return tokens


    def get_vocab(self):
        return list(self.vocab)

"""
Dataset class for aclImdb movie review dataset
"""
from spacy.tokenizer import Tokenizer
from torch.utils.data import Dataset
from spacy.lang.en import English
import torch


def get_vmap_from_countfile(path, limit, unknown, pad):

    vmap, count = {}, 0
    with open(path, 'r') as infile:

        line = infile.readline()
        while line:
            word, freq = line.split('\t')

            if count == limit-2: 
                break

            line = infile.readline()
            vmap[word] = count
            count += 1

    vmap[unknown] = count
    vmap[pad] = count+1
    return vmap
        

class BooksCorpus(Dataset):


    def __init__(self, datapath, countpath, window, vocab=float('inf'),
                 unknown='<UNK>', pad='<PAD>'):
        super(Dataset, self).__init__()

        self.vmap = get_vmap_from_countfile(countpath, vocab, unknown, pad)
        self.tokenizer = Tokenizer(English().vocab)
        self.data = open(datapath, 'r').readlines()
        self.unknown = unknown
        self.window = window
        self.pad = pad


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        tokens = [
            t.text if t.text in self.vmap else self.unknown 
            for t in self.tokenizer(self.data[idx].strip())
        ]

        if len(tokens) < self.window + 1:
            padding = [self.pad for i in range(self.window+1-len(tokens))]
            tokens = padding + tokens

        ids = torch.Tensor([self.vmap[t] for t in tokens])
        ids = ids.type(torch.LongTensor)
        xids, yid = ids[:-1], ids[-1]

        return xids, yid

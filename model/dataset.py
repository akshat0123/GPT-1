"""
Dataset class for BooksCorpus movie review dataset
"""
from typing import Dict

from spacy.tokenizer import Tokenizer
from torch.utils.data import Dataset
from spacy.lang.en import English
import torch


def get_vmap_from_countfile(path: str, limit: int, unknown: str, 
                            pad: str) -> Dict[str, int]:
    """ Takes in path to tsv file mapping words to their frequencies and returns
        a dictionary mapping vocabulary terms to indices

    Args:
        path: path to counts tsv file
        limit: maximum vocabulary size
        unknown: token to represent unknown words
        pad: token to represent padding

    Returns:
        (Dict[str, int]): dictionary mapping terms to indices
    """

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


    def __init__(self, datapath: str, countpath: str, window: int, 
                 vocab: int=float('inf'), unknown: str='<UNK>', 
                 pad: str='<PAD>') -> 'BooksCorpus':
        """ Dataset class for BooksCorpus dataset

        Args:
            datapath: path containing split BooksCorpus texts
            countpath: path listing term frequencies
            window: size of window for each line of text
            vocab: vocabulary size
            unknown: token to use for unknown tokens
            pad: token to use for padding

        Returns:
            (BooksCorpus): BooksCorpus dataset instance
        """

        super(Dataset, self).__init__()

        self.vmap = get_vmap_from_countfile(countpath, vocab, unknown, pad)
        self.tokenizer = Tokenizer(English().vocab)
        self.data = open(datapath, 'r').readlines()
        self.unknown = unknown
        self.window = window
        self.pad = pad


    def __len__(self) -> int:
        """ Returns dataset size

        Returns:
            (int): dataset size
        """
        return len(self.data)


    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """ Returns item at specified index

        Args:
            idx: index of item to return

        Returns:
            (torch.Tensor): list of token ids of size {self.window}
            (torch.Tensor): token id of next word (to be predicted)
        """

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

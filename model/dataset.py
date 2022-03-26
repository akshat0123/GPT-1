from random import sample

from torch import FloatTensor, LongTensor, Tensor, cat
from torch.utils.data import IterableDataset
from torch.nn.functional import one_hot


class TokenIDDataset(IterableDataset):


    def __init__(self, datapath: str, window_size: int, vocab_size: int, 
                 pad: int):
        """ Dataset class for dataset of variable length lines of text token
            byte pair ids

        Args:
            datapath: file where data is located
            window_size: size of window of data to return
            vocab_size: total vocab size for one-hot encodings
            pad: token id for pad token
        """
        super().__init__()
        self.pad = [pad for i in range(window_size-1)]
        self.data = open(datapath).readlines()
        self.window_size = window_size
        self.vocab_size = vocab_size


    def __iter__(self):
        for line_idx in range(len(self.data)):

            # Add padding to line to make it window length
            line = self.pad + [int(x) for x in self.data[line_idx].split(' ')]

            # Return each window length sequence of the line
            start, end = 0, self.window_size + 1
            while end < len(line):
                ids = LongTensor(line[start:end])
                ids = one_hot(ids, num_classes=self.vocab_size)
                yield ids, line_idx
                start += 1
                end += 1


    def __len__(self):
        return len(self.data)


    @staticmethod
    def collate(batch: Tensor) -> (Tensor, Tensor):
        """ Join batch of TokenIDDataset members

        Args:
            batch: batch of ids 

        Returns:
            (Tensor): Tensor of joined batch ids 
            (Tensor): Tensor of line numbers
        """

        ids = [batch[i][0][None, :] for i in range(len(batch))]
        line_idx = batch[-1][1]
        ids = cat(ids, dim=0)
        ids = ids.type(FloatTensor)
        return ids, line_idx


class TokenIDSubset(TokenIDDataset):


    def __init__(self, dataset: TokenIDDataset, size: int):
        """ Dataset class for subset of byte pair token id dataset 

        Args:
            dataset: token id dataset to subset
            size: number of lines to sample from token id dataset
        """
        self.data = sample(dataset.data, size)
        self.window_size = dataset.window_size
        self.vocab_size = dataset.vocab_size
        self.pad = dataset.pad


    def __iter__(self):
        yield from super().__iter__()


    def __len__(self):
        return super().__len__()

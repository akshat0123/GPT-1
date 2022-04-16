from random import sample

from torch import FloatTensor, LongTensor, Tensor, stack, cat
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
            line = self.data[line_idx].strip().split(' ')
            line = self.pad + [int(x) for x in line]

            # Return each window length sequence of the line
            start, end = 0, self.window_size + 1
            while end < len(line):
                ids = LongTensor(line[start:end])
                pads = (ids!=self.pad[0]).float()
                yield ids[:-1], ids[-1], pads[:-1], line_idx
                start += 1
                end += 1


    def __len__(self):
        return len(self.data)


    @staticmethod
    def collate(batch: Tensor) -> (Tensor, Tensor, Tensor, int):
        """ Join batch of TokenIDDataset members

        Args:
            batch: batch of ids 

        Returns:
            (Tensor): Tensor of joined batch ids 
            (Tensor): Tensor of joined batch ids 
            (Tensor): Tensor of joined pad indicators
            (int): Last line number in batch
        """

        xids = [batch[i][0][None, :] for i in range(len(batch))]
        pads = [batch[i][2][None, :] for i in range(len(batch))]
        yids = [batch[i][1] for i in range(len(batch))]
        line_idx = batch[-1][3]
        xids = cat(xids, dim=0)
        pads = cat(pads, dim=0)
        yids = stack(yids, dim=0)
        return xids, yids, pads, line_idx


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

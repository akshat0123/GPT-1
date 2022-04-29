from random import randint, sample

from torch import FloatTensor, LongTensor, Tensor, stack, cat
from torch.utils.data import IterableDataset
from torch.nn.functional import one_hot


class TokenIDDataset(IterableDataset):


    def __init__(self, datapath: str, window_size: int, vocab_size: int, 
                 unk: int):
        """ Dataset class for dataset of variable length lines of text token
            byte pair ids

        Args:
            datapath: file where data is located
            window_size: size of window of data to return
            vocab_size: total vocab size for one-hot encodings
            unk: token id for unknown token
        """
        super().__init__()
        self.data = open(datapath).readlines()
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.unk_token = unk


    def __iter__(self):
        for line_idx in range(len(self.data)):

            line = self.data[line_idx].strip().split(' ')
            start = randint(0, len(line)-self.window_size-1)
            end = start + self.window_size + 1

            ids = LongTensor([int(x) for x in line[start:end]])
            ignore = (ids==self.unk_token).float()

            yield ids[:-1], ids[1:], ignore[:-1]


    def __len__(self):
        return len(self.data)


    @staticmethod
    def collate(batch: Tensor) -> (Tensor, Tensor, Tensor):
        """ Join batch of TokenIDDataset members

        Args:
            batch: batch of ids 

        Returns:
            (Tensor): Tensor of joined batch ids 
            (Tensor): Tensor of joined batch ids 
            (Tensor): Tensor of joined indicators for indices to ignore
        """

        xids = cat([batch[i][0][None, :] for i in range(len(batch))], dim=0)
        yids = cat([batch[i][1][None, :] for i in range(len(batch))], dim=0)
        ignore = cat([batch[i][2][None, :] for i in range(len(batch))], dim=0)
        return xids, yids, ignore 


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
        self.unk_token = dataset.unk_token


    def __iter__(self):
        yield from super().__iter__()


    def __len__(self):
        return super().__len__()

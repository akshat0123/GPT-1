from typing import List

from torch import LongTensor, Tensor, empty
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from tqdm import tqdm


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


class BytePairDataset(Dataset):


    def __init__(self, datapath: str, linecount: int=None):
        """ Dataset class for dataset segmented into bytes using a byte-pair
            tokenizer

        Args:
            datapath: filepath to dataset
            linecount: number of lines in dataset
        """
        super(Dataset, self).__init__()

        if linecount is None:
            self.data = open(datapath, 'r').readlines()

        else:
            self.data = sopen(datapath, linecount)


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> str:
        return self.data[idx].strip()


class BytePairCollator:


    def __init__(self, window_size, vocab_size, device):
        """ Collator function class for BytePairDataset

        Args:
            window_size: length of sequence window
            vocab_size: vocabulary size
            device: device to put batch on
        """
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.device = device


    def __call__(self, batch: List[str]) -> Tensor:
        """ Collate batch of byte pair token id strings into a tensor batch

        Args:
            batch: list of byte pair token id strings

        Returns:
            (Tensor): one-hot batch of token ids 
        """
        X = empty(
            len(batch), 
            self.window_size + 1, 
            self.vocab_size
        )

        for i in range(len(batch)):
            line = Tensor([int(x) for x in batch[i].split(' ')])
            line = line.type(LongTensor)
            X[i] = one_hot(line, self.vocab_size)[None, :, :]

        X = X.to(device=self.device)
        return X

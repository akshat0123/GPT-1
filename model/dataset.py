from typing import List

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

import yaml, os

from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import torch

from model.dataset import BooksCorpusTokenizer, BooksCorpus


configpath = 'confs/testparams.yml'


def main():

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)

    tokenizer = BooksCorpusTokenizer(**confs['tokenizer']) 
    dataset = BooksCorpus(**confs['dataset'])
    loader = DataLoader(dataset=dataset, **confs['loader'])

    for item in loader:
        x, y = tokenizer.tokenize(item)
        lines = tokenizer.decode(x)
        break


if __name__ == '__main__':
    main()

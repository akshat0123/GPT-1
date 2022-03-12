import yaml, os

from torch.utils.data import random_split, DataLoader
from torch import Tensor

from model.tokenizer import BytePairTokenizer
from model.dataset import BooksCorpus


configpath = 'confs/params.yml'


def main():

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)
    dataset = BooksCorpus(**confs['dataset'])

    # Split into train and dev datasets
    train_size = int(0.9 * len(dataset))
    dev_size = len(dataset) - train_size
    train, dev = random_split(dataset, [train_size, dev_size])

    # Initialize train and dev data loaders
    tloader = DataLoader(dataset=train, **confs['loader'])
    dloader = DataLoader(dataset=dev, **confs['loader'])

    # Initialize tokenizer
    bpt = BytePairTokenizer()
    bpt.load(**confs['tokenizer'])

    for batch in tloader:
        for item in batch:
            ids = Tensor(bpt.to_ids(batch, 512, tokenized=True))
            print(ids.shape)
            break
        break
    


if __name__ == '__main__':
    main()


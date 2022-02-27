from functools import partial
from textwrap import dedent
import yaml, os

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss 
from tqdm import trange, tqdm
from torch.optim import Adam
import torch

from model.tokenizer import BooksCorpusTokenizer
from model.model import TransformerDecoder
from model.dataset import BooksCorpus
from model.trainer import Trainer


configpath = 'confs/params.yml'


def main():

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)

    # Load tokenizer and dataset 
    tokenizer = BooksCorpusTokenizer(**confs['tokenizer']) 
    dataset = BooksCorpus(**confs['dataset'])

    # Split into train and dev datasets
    train_size = int(0.9 * len(dataset))
    dev_size = len(dataset) - train_size
    train, dev = random_split(dataset, [train_size, dev_size])

    # Initialize train and dev data loaders
    tloader = DataLoader(dataset=train, **confs['loader'])
    dloader = DataLoader(dataset=dev, **confs['loader'])

    # Initialize model        
    model = TransformerDecoder(**confs['model'])

    # Initialize optimizer, scheduler, and loss
    optimizer = Adam(model.parameters(), **confs['optimizer'])
    scheduler = CosineAnnealingLR(optimizer=optimizer, **confs['scheduler'])
    loss_fn = CrossEntropyLoss()

    trainer = Trainer(model, tokenizer, optimizer, loss_fn, scheduler)
    for epoch in range(confs['epochs']):
        trainer.train(tloader)
        trainer.validate(dloader)


if __name__ == '__main__':
    main()

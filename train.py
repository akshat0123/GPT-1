from functools import partial
import yaml

from torch.nn import ModuleList, CrossEntropyLoss 
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from torch.optim import SGD
import torch

from model.model import Embedding, Decoder
from model.dataset import BooksCorpus


configpath = 'confs/params.yml'


def train_epoch(embedding: Embedding, decoder: Decoder, loader: DataLoader,
                criterion: CrossEntropyLoss, optimizer: SGD, 
                scheduler: CyclicLR) -> float:
    """ Run one training epoch

    Args:
        embedding: word embedding module
        decoder: transformer-based decoder
        loader: dataset batch loader
        criterion: loss function
        optimizer: optimization function
        scheduler: learning rate scheduler

    Returns:
        (float): average loss across epoch
    """

    total_loss = 0.0
    count = 0

    for x, y in tqdm(loader):

        optimizer.zero_grad()

        x = x.to(device=decoder.device)
        y = y.to(device=decoder.device)

        pred = decoder(embedding(x))
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += x.shape[0]

    scheduler.step()
    return total_loss / count


def val_epoch(embedding: Embedding, decoder: Decoder, loader: DataLoader,
              criterion: CrossEntropyLoss) -> float:
    """ Run one validation epoch

    Args:
        embedding: word embedding module
        decoder: transformer-based decoder
        loader: dataset batch loader
        criterion: loss function

    Returns:
        (float): average loss across epoch
    """

    total_loss = 0.0
    count = 0

    for x, y in tqdm(loader):

        x = x.to(device=decoder.device)
        y = y.to(device=decoder.device)

        pred = decoder(embedding(x))
        loss = criterion(pred, y)

        total_loss += loss.item()
        count += x.shape[0]

    return total_loss / count


def main():

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)

    # Load dataset
    dataset = BooksCorpus(**confs['dataset'])

    # Split into train and dev datasets
    train_size = int(0.9 * len(dataset))
    dev_size = len(dataset) - train_size
    train, dev = random_split(dataset, [train_size, dev_size])

    # Initialize train and dev data loaders
    tloader = DataLoader(batch_size=confs['loader']['batch_size'],
                         dataset=train, drop_last=True, shuffle=True)
    dloader = DataLoader(batch_size=confs['loader']['batch_size'], dataset=dev,
                         drop_last=True, shuffle=True)

    # Initialize model        
    embedding = Embedding(**confs['embedding'])
    decoder = Decoder(**confs['decoder'])

    # Initialize optimizer, scheduler, and loss
    optimizer = SGD(ModuleList([embedding, decoder]).parameters(),
                    **confs['optimizer'])
    scheduler = CyclicLR(optimizer=optimizer, **confs['scheduler'])
    criterion = CrossEntropyLoss()

    # Train model
    for epoch in range(confs['epochs']):

        train_loss = train_epoch(embedding, decoder, tloader, criterion,
                                 optimizer, scheduler)
        val_loss = val_epoch(embedding, decoder, dloader, criterion)



if __name__ == '__main__':
    main()

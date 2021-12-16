from functools import partial
import yaml, os

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss 
from tqdm import trange, tqdm
from torch.optim import SGD
import torch

from model.dataset import BooksCorpus
from model.model import Decoder


configpath = 'confs/params.yml'


def train_epoch(decoder: Decoder, loader: DataLoader, 
                criterion: CrossEntropyLoss, optimizer: SGD, 
                scheduler: CyclicLR) -> (float, float):
    """ Run one training epoch

    Args:
        decoder: transformer-based decoder
        loader: dataset batch loader
        criterion: loss function
        optimizer: optimization function
        scheduler: learning rate scheduler

    Returns:
        (float): average loss across epoch
        (float): average error across epoch
    """

    total_loss = 0.0
    total_err = 0.0
    count = 0

    progress = tqdm(total=len(loader), desc='Train Loss: | Train Err: ')
    for x, y in loader:

        optimizer.zero_grad()

        x = x.to(device=decoder.device)
        x = torch.nn.functional.one_hot(x, 1000)
        x = x.type(torch.FloatTensor)
        y = y.to(device=decoder.device)

        pred = decoder(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        yhat = torch.argmax(pred, dim=1)
        total_err += torch.sum(yhat!=y).item()
        total_loss += loss.item()
        count += x.shape[0]

        desc = f'Train Loss: {total_loss/count:.4f} | Train Err: {total_err/count:.4f}'
        progress.set_description(desc)
        progress.update(1)

    scheduler.step()

    return total_loss/count, total_err/count


def val_epoch(decoder: Decoder, loader: DataLoader, 
              criterion: CrossEntropyLoss) -> (float, float):
    """ Run one validation epoch

    Args:
        decoder: transformer-based decoder
        loader: dataset batch loader
        criterion: loss function

    Returns:
        (float): average loss across epoch
        (float): average error across epoch
    """

    total_loss = 0.0
    total_err = 0.0
    count = 0

    progress = tqdm(total=len(loader), desc='Val Loss: | Val Err: ')
    for x, y in loader:

        x = x.to(device=decoder.device)
        x = torch.nn.functional.one_hot(x, 1000)
        x = x.type(torch.FloatTensor)
        y = y.to(device=decoder.device)

        pred = decoder(x)
        loss = criterion(pred, y)

        yhat = torch.argmax(pred, dim=1)
        total_err += torch.sum(yhat!=y).item()
        total_loss += loss.item()
        count += x.shape[0]

        desc = f'Val Loss: {total_loss/count:.4f} | Val Err: {total_err/count:.4f}'
        progress.set_description(desc)
        progress.update(1)

    return total_loss/count, total_err/count


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
    model = Decoder(**confs['decoder'])

    # Initialize optimizer, scheduler, and loss
    optimizer = SGD(model.parameters(), **confs['optimizer'])
    scheduler = CyclicLR(optimizer=optimizer, **confs['scheduler'])
    criterion = CrossEntropyLoss()

    writer = SummaryWriter()

    # Train model
    min_vloss = float('inf')
    for epoch in range(confs['epochs']):

        tloss, terr = train_epoch(model, tloader, criterion, optimizer,
                                  scheduler)
        vloss, verr = val_epoch(model, dloader, criterion)

        writer.add_scalar('Train Loss', tloss, epoch)
        writer.add_scalar('Val Loss', vloss, epoch)
        writer.add_scalar('Train Err', terr, epoch)
        writer.add_scalar('Val Err', verr, epoch)

        checkpoint = { 'optimizer': optimizer.state_dict(), 
                       'scheduler': scheduler.state_dict(), 
                       'model': model.state_dict(), 'train_loss': tloss,
                       'val_loss': vloss, 'train_err': terr, 'val_err': verr,
                       'epoch': epoch }

        if vloss < min_vloss:
            torch.save(checkpoint, os.path.join(confs['checkpoint'], 'best.pt'))
        torch.save(checkpoint, os.path.join(confs['checkpoint'], 'latest.pt'))


if __name__ == '__main__':
    main()

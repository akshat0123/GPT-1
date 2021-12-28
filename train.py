from functools import partial
import yaml, os

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss 
from tqdm import trange, tqdm
from torch.optim import Adam
import torch

from model.model import TransformerDecoder
from model.dataset import BooksCorpus


configpath = 'confs/params.yml'


def train_step(decoder: TransformerDecoder, criterion: CrossEntropyLoss,
               optimizer: Adam, x: torch.Tensor, y: torch.tensor, 
               vsize: int, tensortype: torch.dtype, 
               val: bool) -> (torch.Tensor, torch.Tensor):
    """ Run one training step

    Args:
        decoder: transformer-based decoder
        criterion: loss function
        optimizer: optimization function
        x: batch for model input
        y: batch of input labels
        vsize: vocabulary size
        val: boolean flag called for validation dataset

    Returns:
        (torch.Tensor): model predicted labels
        (torch.Tensor): loss calculated for step
    """
    

    if not val:
        optimizer.zero_grad()

    x = x.to(device=decoder.device)
    x = torch.nn.functional.one_hot(x, vsize)
    x = x.type(tensortype)
    y = y.to(device=decoder.device)

    pred = decoder(x)
    loss = criterion(pred, y)

    if not val:
        loss.backward()
        optimizer.step()

    return pred, loss


def run_epoch(decoder: TransformerDecoder, loader: DataLoader, 
              criterion: CrossEntropyLoss, optimizer: Adam, 
              scheduler: CosineAnnealingLR, vsize: int, 
              val: bool=False) -> (float, float):
    """ Run one training epoch

    Args:
        decoder: transformer-based decoder
        loader: dataset batch loader
        criterion: loss function
        optimizer: optimization function
        scheduler: learning rate scheduler
        vsize: vocabulary size
        val: boolean flag called for validation dataset

    Returns:
        (float): average loss across epoch
        (float): average error across epoch
    """
    if decoder.device != 'cpu':         
        tensortype = torch.cuda.FloatTensor 

    else:
        tensortype = torch.FloatTensor

    total_loss = 0.0
    total_err = 0.0
    count = 0

    dname = 'Train' if not val else 'Val'
    progress = tqdm(total=len(loader), desc=f'{dname} Loss: | {dname} Err: ')
    for x, y in loader:

        pred, loss = train_step(decoder, criterion, optimizer, x, y, vsize,
                                tensortype, val)

        yhat = torch.argmax(pred, dim=1)
        total_err += torch.sum(yhat!=y).item()
        total_loss += loss.item()
        count += x.shape[0]

        desc = f'{dname} Loss: {total_loss/count:.6f} | {dname} Err: {total_err/count:.6f}'
        progress.set_description(desc)
        progress.update(1)

    if not val:
        scheduler.step()

    return total_loss/count, total_err/count


def main():

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)
    vsize = confs['dataset']['vocab']

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
    model = TransformerDecoder(**confs['model'])

    # Initialize optimizer, scheduler, and loss
    opt = Adam(model.parameters(), **confs['optimizer'])
    scheduler = CosineAnnealingLR(optimizer=opt, **confs['scheduler'])
    loss = CrossEntropyLoss()

    writer = SummaryWriter()

    # Train model
    min_vloss = float('inf')
    for epoch in range(confs['epochs']):

        tloss, terr = run_epoch(model, tloader, loss, opt, scheduler, vsize)
        vloss, verr = run_epoch(model, tloader, loss, opt, scheduler, vsize,
                                val=True)

        writer.add_scalar('Train Loss', tloss, epoch)
        writer.add_scalar('Val Loss', vloss, epoch)
        writer.add_scalar('Train Err', terr, epoch)
        writer.add_scalar('Val Err', verr, epoch)

        checkpoint = { 'optimizer': opt.state_dict(), 
                       'scheduler': scheduler.state_dict(), 
                       'model': model.state_dict(), 'train_loss': tloss,
                       'val_loss': vloss, 'train_err': terr, 'val_err': verr,
                       'epoch': epoch }

        if vloss < min_vloss:
            torch.save(checkpoint, os.path.join(confs['checkpoint'], 'best.pt'))
        torch.save(checkpoint, os.path.join(confs['checkpoint'], 'latest.pt'))


if __name__ == '__main__':
    main()

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

from model.dataset import BooksCorpusTokenizer, BooksCorpus
from model.model import TransformerDecoder


configpath = 'confs/params.yml'


def train_step(decoder: TransformerDecoder, criterion: CrossEntropyLoss,
               optimizer: Adam, x: torch.Tensor, y: torch.tensor, 
               val: bool) -> (torch.Tensor, torch.Tensor):
    """ Run one training step

    Args:
        decoder: transformer-based decoder
        criterion: loss function
        optimizer: optimization function
        x: batch for model input
        y: batch of input labels
        val: boolean flag called for validation dataset

    Returns:
        (torch.Tensor): model predicted labels
        (torch.Tensor): loss calculated for step
    """
    

    if not val:
        optimizer.zero_grad()

    pred = decoder(x)
    loss = criterion(pred, y)

    if not val:
        loss.backward()
        optimizer.step()

    return pred, loss


def run_epoch(decoder: TransformerDecoder, loader: DataLoader, 
              tokenizer: BooksCorpusTokenizer, criterion: CrossEntropyLoss,
              optimizer: Adam, scheduler: CosineAnnealingLR, 
              val: bool=False) -> (float, float):
    """ Run one training epoch

    Args:
        decoder: transformer-based decoder
        loader: dataset batch loader
        tokenizer: dataset tokenizer
        criterion: loss function
        optimizer: optimization function
        scheduler: learning rate scheduler
        val: boolean flag called for validation dataset

    Returns:
        (float): average loss across epoch
        (float): average error across epoch
    """
    tensortype = torch.cuda.LongTensor if decoder.device != 'cpu' \
                 else torch.LongTensor

    total_loss = 0.0
    total_err = 0.0
    count = 0

    with torch.set_grad_enabled(not val):

        dname = 'Train' if not val else 'Val'
        progress = tqdm(total=len(loader), desc=f'{dname} Loss: | Err: ')
        for batch in loader:

            x, y = tokenizer.tokenize(batch)
            x = x.to(device=decoder.device)
            y = y.to(device=decoder.device)
            y = y.type(tensortype)

            pred, loss = train_step(decoder, criterion, optimizer, x, y, val)

            yhat = torch.argmax(pred, dim=1)
            total_err += torch.sum(yhat!=y).item()
            total_loss += loss.item()
            count += x.shape[0]

            desc = f'{dname} Loss: {total_loss/count:.10f} | Err: {total_err/count:.10f}'
            progress.set_description(desc)
            progress.update(1)

        if not val:
            scheduler.step()

    return total_loss/count, total_err/count


def main():

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)

    # Load tokenizer and dataset 
    tok = BooksCorpusTokenizer(**confs['tokenizer']) 
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
    opt = Adam(model.parameters(), **confs['optimizer'])
    sch = CosineAnnealingLR(optimizer=opt, **confs['scheduler'])
    loss = CrossEntropyLoss()

    writer = SummaryWriter()

    # Train model
    min_vloss = float('inf')
    for epoch in range(confs['epochs']):

        print(f"\nEpoch {epoch+1}/{confs['epochs']}")
        tloss, terr = run_epoch(model, tloader, tok, loss, opt, sch)
        vloss, verr = run_epoch(model, dloader, tok, loss, opt, sch, val=True)

        writer.add_scalar('Train Loss', tloss, epoch)
        writer.add_scalar('Val Loss', vloss, epoch)
        writer.add_scalar('Train Err', terr, epoch)
        writer.add_scalar('Val Err', verr, epoch)

        checkpoint = { 'optimizer': opt.state_dict(), 
                       'scheduler': sch.state_dict(), 
                       'model': model.state_dict(), 'train_loss': tloss,
                       'val_loss': vloss, 'train_err': terr, 'val_err': verr,
                       'epoch': epoch }

        if vloss < min_vloss:
            torch.save(checkpoint, os.path.join(confs['checkpoint'], 'best.pt'))
        torch.save(checkpoint, os.path.join(confs['checkpoint'], 'latest.pt'))


if __name__ == '__main__':
    main()

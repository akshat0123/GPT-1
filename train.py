import yaml, os

from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import torch

from model.model import Embedding, Decoder
from model.dataset import ReviewDataset


def main():

    with open('confs/params.yml', 'r') as infile:
        params = yaml.load(infile, Loader=yaml.Loader)

    dataset = ReviewDataset(
        dpath=params['dataset']['dpath'],
        vpath=params['dataset']['vpath'],
        limit=params['dataset']['limit'],
        vsize=params['dataset']['size']
    )

    loader = DataLoader(
        dataset=dataset, 
        batch_size=params['train']['batch_size'],
        shuffle=True
    )

    embedding = Embedding(
        size=params['dataset']['size'] + 3,
        dim=params['embedding']['dim']
    )

    decoder = Decoder(
        n_layers=params['model']['n_layers'],
        n_heads=params['model']['n_heads'],
        d_out=params['dataset']['size']+3,
        d_in=params['model']['dim']
    )

    optimizer = torch.optim.SGD(
        torch.nn.ModuleList([embedding, decoder]).parameters(), 
        lr=params['train']['max_lr']
    )

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        base_lr=params['train']['base_lr'],
        max_lr=params['train']['max_lr'],
        optimizer=optimizer
    )

    criterion = torch.nn.CrossEntropyLoss()

    checkpoint = params['checkpoint']
    if os.path.isfile(checkpoint):
        state = torch.load(checkpoint)
        decoder.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch']
        loss = state['loss']

    else:
        start_epoch = 0

    if start_epoch + 1 < params['train']['epochs']:
        for epoch in trange(start_epoch+1, params['train']['epochs']):
            for x, y in tqdm(loader):
                optimizer.zero_grad()
                pred = decoder(embedding(x))
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

            torch.save({
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model': model.state_dict(),
                'epoch': epoch,
                'loss': loss
            }, checkpoint)



if __name__ == '__main__':
    main()

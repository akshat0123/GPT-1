import yaml

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

    loader = DataLoader(
        dataset=dataset, 
        batch_size=params['train']['batch_size'],
        shuffle=True
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

    for epoch in trange(params['train']['epochs']):
        for x, y in tqdm(loader):
            optimizer.zero_grad()
            pred = decoder(embedding(x))[:, -1, :]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()


if __name__ == '__main__':
    main()

from functools import partial
import yaml

from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import torch

from model.model import Embedding, Decoder
from model.dataset import BooksCorpus


configpath = 'confs/params.yml'


def main():

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)

    dataset = BooksCorpus(**confs['dataset'])
    loader = DataLoader(
        batch_size=confs['loader']['batch_size'],
        dataset=dataset,
        drop_last=True,
        shuffle=True
    )

    embedding = Embedding(**confs['embedding'])
    decoder = Decoder(**confs['decoder'])

    optimizer = torch.optim.SGD(
        torch.nn.ModuleList([embedding, decoder]).parameters(),
        **confs['optimizer']
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        **confs['scheduler']
    )

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(confs['epochs']):
        for x, y in tqdm(loader):

            optimizer.zero_grad()
            embedding.train()
            decoder.train()

            pred = decoder(embedding(x))
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        scheduler.step()


if __name__ == '__main__':
    main()

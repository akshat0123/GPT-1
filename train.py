import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from model.dataset import TokenIDDataset, TokenIDSubset
from model.model import TransformerDecoder
from model.utils import RollingCounter


confpath = './confs/params.yml'


def main():

    confs = yaml.safe_load(open(confpath))

    train_data = TokenIDDataset(**confs['train_data'])
    dev_data = TokenIDDataset(**confs['dev_data'])
    model = TransformerDecoder(**confs['model'])

    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100)
    crit = torch.nn.CrossEntropyLoss()

    train = TokenIDSubset(train_data, **confs['train_subset'])
    dev = TokenIDSubset(dev_data, **confs['dev_subset'])

    for epoch in range(confs['epochs']):

        print(f'\n\nEpoch {epoch + 1}')

        tloader = DataLoader(collate_fn=TokenIDDataset.collate, **confs['loader'], dataset=train)
        dloader = DataLoader(collate_fn=TokenIDDataset.collate, **confs['loader'], dataset=dev)

        barsize = len(tloader.dataset)
        progress = tqdm(total=barsize, desc='LR: | Loss: | Err: ')
        model.train()

        loss_metric, err_metric = RollingCounter(1000), RollingCounter(1000)
        for batch, line_idx in tloader:
            x, y = batch[:, :-1], batch[:, -1].long()

            opt.zero_grad()

            y_pred = model(x.long())
            loss = crit(y_pred, y)

            loss.backward()
            opt.step()
            sch.step()

            y_pred = torch.argmax(y_pred, dim=1)
            err = (y_pred!=y).sum() / y.shape[0]

            loss_metric.add(loss.item())
            err_metric.add(err)

            if line_idx > progress.n:
                progress.set_description(
                    f'LR: {sch.get_last_lr()[-1]:.8f} | '
                    f'Loss: {loss_metric.rolling_average():.8f} | '
                    f'Err: {err_metric.rolling_average():.8f}'
                )
                progress.update(line_idx - progress.n)

        progress.update(barsize - progress.n)


if __name__ == '__main__':
    main()

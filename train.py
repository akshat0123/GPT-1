import shutil, yaml, os

from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch import ones, save

from model.dataset import TokenIDDataset, TokenIDSubset
from model.model import TransformerDecoder
from model.trainer import Trainer


confpath = 'confs/params.yml'


def save_checkpoint(path, model, opt, sch, epoch):

    filepath = f'{path}/epoch_{epoch}'
    if os.path.exists(filepath):
        shutil.rmtree(filepath)

    os.makedirs(filepath)
    save(model.state_dict(), f'{filepath}/model.pth')
    save(opt.state_dict(), f'{filepath}/opt.pth')
    save(sch.state_dict(), f'{filepath}/sch.pth')


def publish_metrics(logger, train_metrics, dev_metrics, epoch):

    for key in train_metrics:
        logger.add_scalar(f'train_{key}', train_metrics[key], epoch)

    for key in dev_metrics:
        logger.add_scalar(f'dev_{key}', train_metrics[key], epoch)


def main():

    confs = yaml.safe_load(open(confpath))

    train_data = TokenIDDataset(**confs['train_data'])
    dev_data = TokenIDDataset(**confs['dev_data'])

    model = TransformerDecoder(**confs['model'])
    opt = Adam(model.get_parameters(), **confs['opt'])
    sch = OneCycleLR(opt, **confs['sch'])
    crit = CrossEntropyLoss(ignore_index=confs['unk'])
    trainer = Trainer(model, crit, opt, sch, **confs['trainer'])
    logger = SummaryWriter(**confs['logger'])

    for epoch in range(confs['epochs']):

        print(f'\n\nEpoch {epoch+1}')
        train = TokenIDSubset(train_data, **confs['train_subset'])
        dev = TokenIDSubset(dev_data, **confs['dev_subset'])

        collate = TokenIDDataset.collate
        tloader = DataLoader(collate_fn=collate, **confs['loader'], dataset=train)
        dloader = DataLoader(collate_fn=collate, **confs['loader'], dataset=dev)

        train_metrics = trainer.run_epoch(tloader)
        dev_metrics = trainer.run_epoch(dloader, train_mode=False)
        publish_metrics(logger, train_metrics, dev_metrics, epoch+1)
        save_checkpoint(confs['checkpoint'], model, opt, sch, epoch+1)


if __name__ == '__main__':
    main()

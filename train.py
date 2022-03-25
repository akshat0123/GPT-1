import argparse, pickle, yaml

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from model.dataset import TokenIDDataset, TokenIDSubset
from model.model import TransformerDecoder
from model.trainer import Trainer


config_path = 'confs/params.yaml'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path

    confs = yaml.safe_load(open(config_path, 'r'))

    # Load dataset
    train = TokenIDDataset(**confs['train_data'])
    dev = TokenIDDataset(**confs['dev_data'])

    # Initialize model
    model = TransformerDecoder(**confs['model'])

    # Initialize optimizer, scheduler, and loss
    optimizer = Adam(model.parameters(), **confs['optimizer'])
    scheduler = CosineAnnealingLR(optimizer=optimizer, **confs['scheduler'])
    loss_fn = CrossEntropyLoss()

    current_epoch = 0
    if checkpoint_path is not None:
        checkpoint = pickle.load(open(checkpoint_path, 'rb'))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
        current_epoch = checkpoint['epoch']

    trainer = Trainer(model, optimizer, loss_fn, scheduler)
    collate = TokenIDDataset.collate
    for epoch in range(current_epoch+1, confs['epochs']+1):

        # Create data subsets
        train_sub = TokenIDSubset(train, **confs['train_subset']) 
        dev_sub = TokenIDSubset(dev, **confs['dev_subset']) 

        # Initialize train and dev data loaders
        tloader = DataLoader(train_sub, collate_fn=collate, **confs['loader'])
        dloader = DataLoader(dev_sub, collate_fn=collate, **confs['loader'])

        print(f'\nEpoch: {epoch}')
        train_loss, train_err = trainer.train(tloader)
        dev_loss, dev_err = trainer.validate(dloader)

        checkpoint = trainer.get_checkpoint()
        checkpoint.update({
            'train_loss': train_loss, 
            'train_err': train_err,
            'dev_loss': val_loss, 
            'dev_err': val_err, 
            'epoch': epoch,
        })

        checkpoint_path = f"{confs['checkpoint']}/{epoch}.pickle"
        pickle.dump(checkpoint, open(checkpoint_path, 'wb'))


if __name__ == '__main__':
    main()

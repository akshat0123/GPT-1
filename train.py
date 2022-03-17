import argparse, pickle, yaml

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from model.dataset import BytePairDataset, BytePairCollator
from model.tokenizer import BytePairTokenizer
from model.model import TransformerDecoder
from model.trainer import Trainer


config_path = 'confs/params.yaml'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path

    confs = yaml.safe_load(open(config_path, 'r'))

    # Load tokenizer and dataset
    tokenizer = BytePairTokenizer()
    tokenizer.load(**confs['tokenizer'])
    dataset = BytePairDataset(**confs['dataset'])

    # Split into train and dev datasets
    train_size = int(0.9 * len(dataset))
    dev_size = len(dataset) - train_size
    train, dev = random_split(dataset, [train_size, dev_size])

    # Initialize train and dev data loaders
    collator = BytePairCollator(**confs['collator'])
    tloader = DataLoader(dataset=train, collate_fn=collator, **confs['loader'])
    dloader = DataLoader(dataset=dev, collate_fn=collator, **confs['loader'])

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
        tokenizer = checkpoint['tokenizer']
        current_epoch = checkpoint['epoch']

    model.train()
    trainer = Trainer(model, tokenizer, optimizer, loss_fn, scheduler)
    for epoch in range(current_epoch+1, confs['epochs']+1):

        print(f'\nEpoch: {epoch}')
        train_loss, train_err = trainer.train(tloader)
        val_loss, val_err = trainer.validate(dloader)

        checkpoint = trainer.get_checkpoint()
        checkpoint.update({
            'train_loss': train_loss, 'train_err': train_err,
            'val_loss': val_loss, 'val_err': val_err, 'epoch': epoch,
        })

        checkpoint_path = f"{confs['checkpoint']}/{epoch}.pickle"
        pickle.dump(checkpoint, open(checkpoint_path, 'wb'))


if __name__ == '__main__':
    main()

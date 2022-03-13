import yaml

from torch.utils.data import DataLoader

from model.dataset import BytePairDataset, BytePairCollator
from model.tokenizer import BytePairTokenizer
from model.model import TransformerDecoder


config_path = 'confs/params.yaml'


def main():

    confs = yaml.safe_load(open(config_path, 'r'))

    tokenizer = BytePairTokenizer()
    tokenizer.load(**confs['tokenizer'])
    data = BytePairDataset(**confs['dataset'])
    collator = BytePairCollator(**confs['collator'])
    loader = DataLoader(dataset=data, collate_fn=collator, **confs['loader'])
    model = TransformerDecoder(**confs['model'])

    for batch in loader:
        print(batch.shape)
        Y = model(batch)
        print(Y.shape)
        break


if __name__ == '__main__':
    main()

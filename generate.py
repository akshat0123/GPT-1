import argparse, yaml

from tqdm import trange
import torch

from model.tokenizer import BytePairTokenizer
from model.sequencer import Sequencer
from model.model import GPT


confpath = 'confs/params.yml'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', type=int, default=128)
    args = parser.parse_args()
    length = args.length

    confs = yaml.safe_load(open(confpath))
    model = GPT(**confs['model'])
    model.load_state_dict(torch.load(confs['pretrained_model'])) 
    tokenizer = BytePairTokenizer.load('data/checkpoints/tokenizer')

    sequencer = Sequencer(model, tokenizer, **confs['sequencer'])
    sequence = sequencer.generate_sequence(length)
    print(sequence)


if __name__ == '__main__':
    main()

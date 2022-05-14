import argparse, yaml

from tqdm import trange
from torch import load

from model.tokenizer import BytePairTokenizer
from model.sequencer import Sequencer
from model.model import GPT


confpath = 'confs/generate.yml'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', type=int, default=128)
    args = parser.parse_args()
    length = args.length

    confs = yaml.safe_load(open(confpath))
    model = GPT(**confs['model'])
    model.load_state_dict(load(confs['pretrained_model'])) 
    tokenizer = BytePairTokenizer.load(confs['trained_tokenizer'])

    sequencer = Sequencer(model, tokenizer, **confs['sequencer'])
    sequence = sequencer.generate_sequence(length)
    print(sequence)


if __name__ == '__main__':
    main()

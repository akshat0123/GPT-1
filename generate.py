import argparse, pickle, yaml

from model.tokenizer import BooksCorpusTokenizer
from model.model import TransformerDecoder
from model.sequencer import Sequencer


configpath = 'confs/params.yml'


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--checkpoint_path', required=True)
    # args = parser.parse_args()
    # checkpoint_path = args.checkpoint_path
    checkpoint_path = '/home/akshat/Programs/Decoders/checkpoints/10.pickle' 

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)

    # Load tokenizer and model
    tokenizer = BooksCorpusTokenizer(**confs['tokenizer']) 
    model = TransformerDecoder(**confs['model'])
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    model.load_state_dict(checkpoint['model'])
    tokenizer = checkpoint['tokenizer']

    sequencer = Sequencer(tokenizer, model, 10)

    model.eval()
    for i in range(10):
        sequence = sequencer.generate_sequence(128)
        print(' '.join(sequence))


if __name__ == '__main__':
    main()

from functools import partial
import yaml, os

from tqdm import trange, tqdm
import torch

from model.dataset import BooksCorpusTokenizer
from model.model import TransformerDecoder


configpath = 'confs/params.yml'


def main():

    torch.manual_seed(29)

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)

    # Load tokenizer and dataset 
    tok = BooksCorpusTokenizer(**confs['tokenizer']) 

    # Initialize model        
    model = TransformerDecoder(**confs['model'])
    checkpoint = torch.load(os.path.join(confs['checkpoint'], 'best.pt'))
    model.load_state_dict(checkpoint['model'])

    tensortype = torch.cuda.FloatTensor if model.device != 'cpu' \
                 else torch.FloatTensor

    lim = 100
    k = 10

    seq = tok.start_seq().type(tensortype) 
    tokens = []

    while True:

        # Predict next vector in sequence (logprobs)
        out = model(seq)

        # Set probability of '<UNK>' to -1
        out[:, tok.vmap[tok.unk]] = -1

        # Sort token ids by probability
        ids = torch.argsort(out, dim=1, descending=True)

        # Keep only top k and shuffle
        ids = ids[:, :k]
        ids = ids[:, torch.randperm(k)][:, 0]
        ids = torch.nn.functional.one_hot(ids, tok.vocab)

        # Get token and add it to sequence
        tokens += tok.decode(ids)

        # Add token to index and move window to the right
        seq = torch.cat([ids[None, :, :], seq], dim=1)
        seq = seq[:, 1:, :]

        if tokens[-1] == tok.end or len(tokens) > lim:
            break

    print(tokens)


if __name__ == '__main__':
    main()

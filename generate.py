import yaml

from tqdm import trange
import torch

from model.tokenizer import BytePairTokenizer
from model.model import TransformerDecoder


confpath = 'confs/params.yml'


def main():

    confs = yaml.safe_load(open(confpath))
    model = TransformerDecoder(**confs['model'])
    model.load_state_dict(torch.load('data/checkpoints/model/koala/epoch_100/model.pth')) 
    tokenizer = BytePairTokenizer.load('data/checkpoints/tokenizer')

    # Create window with pad
    tokens = [tokenizer.get_pad() for i in range(confs['window_size']-1)] + [tokenizer.get_sol()]
    token_ids = torch.LongTensor(tokenizer.get_byte_ids(tokens))[None, :]
    pad_ids = (token_ids!=tokenizer.get_byte_id(tokenizer.get_pad())).float()

    token_ids = token_ids.to(device=confs['device'])
    pad_ids = pad_ids.to(device=confs['device'])

    for i in trange(128):

        ntoken_id = torch.argmax(model(token_ids, pad_ids)[:, -1, :], dim=1)[None, :]
        npad_id = torch.LongTensor([1])[None, :].to(device=confs['device'])

        token_ids = torch.cat([token_ids, ntoken_id], dim=1)[:, 1:]
        pad_ids = torch.cat([pad_ids, npad_id], dim=1)[:, 1:]

    tokens = tokenizer.get_bytes([str(x) for x in token_ids.tolist()[0]])
    print(' '.join(tokens))



if __name__ == '__main__':
    main()

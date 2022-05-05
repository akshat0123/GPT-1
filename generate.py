from random import randint
import yaml

from tqdm import trange
import torch

from model.tokenizer import BytePairTokenizer
from model.model import GPT


confpath = 'confs/params.yml'


def main():

    confs = yaml.safe_load(open(confpath))
    model = GPT(**confs['model'])
    model.load_state_dict(torch.load('data/checkpoints/model/aardvark/epoch_100/model.pth')) 
    tokenizer = BytePairTokenizer.load('data/checkpoints/tokenizer')

    string = "What animal do you like the most"
    chunks = string.split(" ")

    line = [tokenizer.get_sol()]
    for chunk in chunks:
        bytes_ = list(chunk) + [tokenizer.get_eow()]
        line += tokenizer.merge_bytes(bytes_)

    line_ids = torch.LongTensor(tokenizer.get_byte_ids(line)).unsqueeze(0)
    pad_id = tokenizer.get_byte_id(tokenizer.get_pad())
    pad_size = confs['window_size'] - line_ids.size(1)
    padding = torch.full((1, pad_size), pad_id, dtype=torch.long)

    device = confs['device']
    end_idx = line_ids.shape[1] - 1
    line_ids = torch.cat([line_ids, padding], dim=1).to(device=device)
    ignore_ids = (line_ids==pad_id).float().to(device=device) 

    model.eval()
    with torch.no_grad():

        for i in trange(75):

            output = model(line_ids, ignore_ids)

            next_id_probs = output[:, end_idx, :]
            next_id_cands = torch.argsort(next_id_probs, descending=True)
            next_id = next_id_cands[0, randint(0, 10)].unsqueeze(0).unsqueeze(0)

            line_ids = torch.cat([line_ids[:, :end_idx+1], next_id], dim=1)
            pad_size = confs['window_size'] - line_ids.size(1)
            padding = torch.full((1, pad_size), pad_id, dtype=torch.long)

            line_ids = torch.cat([line_ids, padding.to(device=device)], dim=1)
            ignore_ids = (line_ids==pad_id).float() 

            end_idx += 1

        print_ids = line_ids[(line_ids!=pad_id).nonzero(as_tuple=True)]
        tokens = tokenizer.get_bytes([str(x) for x in print_ids.tolist()])

    text, word = [], ''
    for i in range(len(tokens)):

        word += tokens[i]

        if word.startswith('<line/>'):
            word = word[7:]

        elif word.endswith('</line>'):
            word = word[:-7]

        if word.endswith('</w>'):
            text.append(word[:-4])
            word = ''

    print(' '.join(text))        


if __name__ == '__main__':
    main()

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

    window_size = confs['window_size']
    device = confs['device']
    K = 100 

    tokens = [tokenizer.get_sol()]

    string = "Hello World!"
    chunks = string.split(" ")
    for chunk in chunks:
        bytes_ = list(chunk) + [tokenizer.get_eow()]
        tokens += tokenizer.merge_bytes(bytes_)

    token_ids = torch.LongTensor(tokenizer.get_byte_ids(tokens)).unsqueeze(0)
    pad_id = tokenizer.get_byte_id(tokenizer.get_pad())
    pad_size = window_size - token_ids.size(1)
    padding = torch.full((1, pad_size), pad_id, dtype=torch.long)

    token_ids = torch.cat([token_ids, padding], dim=1).to(device=device)
    ignore_ids = (token_ids==pad_id).float().to(device=device) 
    end_idx = len(tokens) - 1

    model.eval()
    with torch.no_grad():

        for i in trange(128):

            output = model(token_ids, ignore_ids)

            next_id_probs = output[:, end_idx, :].flatten()
            next_id_cands = torch.argsort(next_id_probs, descending=True)[:K]
            next_id_probs = next_id_probs[next_id_cands]
            next_id_probs = torch.nn.functional.softmax(next_id_probs, dim=0)
            next_id = next_id_cands[torch.multinomial(next_id_probs, 1)]

            tokens.append(tokenizer.get_byte(str(next_id.item())))

            if end_idx < window_size-1:
                token_ids = torch.cat([token_ids[:, :end_idx+1], next_id.unsqueeze(0)], dim=1)
                pad_size = window_size - token_ids.size(1)
                padding = torch.full((1, pad_size), pad_id, dtype=torch.long)
                token_ids = torch.cat([token_ids, padding.to(device=device)], dim=1)
                ignore_ids = (token_ids==pad_id).float() 
                end_idx += 1

            else:
                token_ids = torch.cat([token_ids[:, 1:], next_id.unsqueeze(0)], dim=1)
                ignore_ids = (token_ids==pad_id).float() 

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

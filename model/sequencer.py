
from torch import LongTensor, multinomial, no_grad, argsort, full, cat
from torch.nn.functional import softmax
from torch import long as long_
from tqdm import trange


class Sequencer:


    def __init__(self, model, tokenizer, window_size, k, device):
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.k = k


    def generate_sequence(self, length, start=None):
        tokens, token_ids, ignore_ids = self.generate_start_seq(start)
        idx = len(tokens) - 1

        self.model.eval()
        with no_grad():
            for i in trange(length):
                probs = self.model(token_ids, ignore_ids)
                next_id = self.gen_next_token(probs, idx)
                tokens.append(self.tokenizer.get_byte(str(next_id.item())))
                token_ids, ignore_ids, idx = self.update_token_ids(
                    idx, token_ids, next_id
                )

        return self.generate_text(tokens)


    def generate_start_seq(self, start=None):
        pad_id = self.tokenizer.get_byte_id(self.tokenizer.get_pad())
        tokens = [self.tokenizer.get_sol()]

        if start:

            chunks = string.split(" ")
            for chunk in chunks:
                bytes_ = list(chunk) + [self.tokenizer.get_eow()]
                tokens += self.tokenizer.merge_bytes(bytes_)

        token_ids = LongTensor(self.tokenizer.get_byte_ids(tokens)).unsqueeze(0)
        token_ids = token_ids.to(device=self.device)
        token_ids = self.pad_token_ids(token_ids, pad_id)
        ignore_ids = (token_ids==pad_id).float().to(device=self.device) 
        return tokens, token_ids, ignore_ids


    def update_token_ids(self, idx, token_ids, next_id):
        pad_id = self.tokenizer.get_byte_id(self.tokenizer.get_pad())
        if idx < self.window_size-1:
            token_ids = cat([token_ids[:, :idx+1], next_id.unsqueeze(0)], dim=1)
            token_ids = self.pad_token_ids(token_ids, pad_id)
            ignore_ids = (token_ids==pad_id).float() 
            idx += 1

        else:
            token_ids = cat([token_ids[:, 1:], next_id.unsqueeze(0)], dim=1)
            ignore_ids = (token_ids==pad_id).float() 

        return token_ids, ignore_ids, idx


    def gen_next_token(self, probs, idx):
        next_id_probs = probs[:, idx, :].flatten()
        next_id_cands = argsort(next_id_probs, descending=True)[:self.k]
        next_id_probs = next_id_probs[next_id_cands]
        next_id_probs = softmax(next_id_probs, dim=0)
        return next_id_cands[multinomial(next_id_probs, 1)]


    def pad_token_ids(self, token_ids, pad_id):
        pad_size = self.window_size - token_ids.size(1)
        padding = full((1, pad_size), pad_id, dtype=long_)
        return cat([token_ids, padding.to(device=self.device)], dim=1)


    def generate_text(self, tokens):
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

        return ' '.join(text)



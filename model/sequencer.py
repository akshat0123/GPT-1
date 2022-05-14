from typing import List

from torch import LongTensor, multinomial, Tensor, no_grad, argsort, full, cat
from torch.nn.functional import softmax
from torch import long as long_
from tqdm import trange


class Sequencer:


    def __init__(self, model: 'Model', tokenizer: 'Tokenizer', window_size: int, 
                 k: int, device: str):
        """ Initialize sequencer

        Args:
            model: model used for generating sequence
            tokenizer: tokenizer for encoding / decoding ids and tokens
            window_size: window size for sequence generation
            k: k value for top-k decoding
            device: device to load sequencer on
        """
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.k = k


    def generate_sequence(self, length: int, start: str=None) -> str:
        """ Generate text sequence with starting input if provided

        Args:
            length: length of sequence to generate (in bytes)
            start: input sequence to start with (not necessary)

        Return:
            (str): generated sequence string
        """

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


    def generate_start_seq(self, start: str=None) -> (List[str], Tensor, Tensor):
        """ Generate initial starting sequence of tokens, token ids, and ids to
            ignore (padding / unknowns etc)

        Args:
            start: string sequence to start with

        Returns:
            (List[str]): List of tokens
            (Tensor): Tensor of byte token ids
            (Tensor): Tensor of flags indicating ides to ignore
        """

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


    def update_token_ids(self, idx: int, token_ids: Tensor, next_id: Tensor) \
                        -> (Tensor, Tensor, int):
        """ Update current sequence of tensor ids

        Args:
            idx: index last generated token
            token_ids: tensor of current token ids
            next_id: next token id to add to token ids tensor

        Returns:
            (Tensor): tensor of token ids
            (Tensor): tensor of flag with token ids to ignore
            (int): new index
        """

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


    def gen_next_token(self, probs: Tensor, idx: int) -> Tensor:
        """ Get next token id given token probabilities

        Args:
            probs: tensor of probabilities for each vocabulary term
            idx: index of last generated token id

        Returns:
            (Tensor): new token id
        """

        next_id_probs = probs[:, idx, :].flatten()
        next_id_cands = argsort(next_id_probs, descending=True)[:self.k]
        next_id_probs = next_id_probs[next_id_cands]
        next_id_probs = softmax(next_id_probs, dim=0)
        return next_id_cands[multinomial(next_id_probs, 1)]


    def pad_token_ids(self, token_ids: Tensor, pad_id: int) -> Tensor:
        """ Pad token ids if less than required window size

        Args:
            token_ids: tensor of token ids to pad
            pad_id: id of token to pad input with

        Returns:
            (Tensor): padded tensor of token ids
        """

        pad_size = self.window_size - token_ids.size(1)
        padding = full((1, pad_size), pad_id, dtype=long_)
        return cat([token_ids, padding.to(device=self.device)], dim=1)


    def generate_text(self, tokens: List[str]) -> str:
        """ Turn list of tokens into output text

        Args:
            tokens: list of tokens to turn into output text

        Returns:
            (str)
        """

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



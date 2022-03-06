from typing import List

from tqdm import tqdm

import torch


class Sequencer:


    def __init__(self, tokenizer: 'Tokenizer', model: torch.nn.Module, 
                 k: int=10) -> None:
        """ Initialize sequencer module

        Args:
            tokenizer: tokenizer for tokenizing list of strings
            model: model to generate future tokens from
            k: number of top probability tokens to consider for sequence
               generation
        """

        self.tokenizer = tokenizer
        self.model = model
        self.k = k


    def generate_sequence(self, length) -> List[str]:
        """ Generate sequence of given length

        Args:
            length: length of sequence to generate

        Returns:
            (List[str]): generated sequence
        """

        progress = tqdm(total=length)
        sequence, progress = self.generate_start_sequence(progress)
        while len(sequence) < length:
            sequence = self.add_nth_token(sequence)
            progress.update(1)

        return sequence


    def generate_start_sequence(self, progress: tqdm) -> (List[str], tqdm):
        """ Generate initial sequence of model's learned window size

        Args:
            progress: tqdm progress bar

        Returns:
            (List[str]): generated sequence
            tqdm: updated progress bar
        """

        with torch.no_grad():

            sequence = [ self.tokenizer.start ]
            for seq_id in range(self.tokenizer.window):

                pad_length = self.tokenizer.window - len(sequence)
                pads = [self.tokenizer.pad for i in range(pad_length)]
                sequence = self.add_nth_token(sequence, seq_id)
                progress.update(1)

        return sequence, progress


    def add_nth_token(self, sequence: List[str], n: int=-1) -> List[str]:
        """ Generate token for nth position of provided sequence

        Args:
            sequence: sequence to add tokens to
            n: position to add token at

        Returns:
            (List[str]): sequence with new token added  
        """

        ids = self.tokenize_sequence(sequence)
        output = self.get_model_output(ids) 

        token_id = output[0, n]
        tokens = [self.tokenizer.imap[id_.item()] for id_ in token_id]

        token_id = 0
        while tokens[token_id] == self.tokenizer.unk:
            token_id += 1

        sequence.append(tokens[token_id])
        return sequence


    def tokenize_sequence(self, sequence: List[str]) -> torch.Tensor:
        """ Generate ids for provided sequence of words

        Args:
            sequence: list of words to transform into ids

        Returns:
            (torch.Tensor): ids for provided sequence 
        """

        input_ = ' '.join(sequence)
        ids = self.tokenizer.tokenize_line(input_)[None, :, :]
        ids = ids.type(torch.FloatTensor)
        ids = ids.to(device=self.model.device)

        return ids


    def get_model_output(self, ids: torch.Tensor) -> torch.Tensor:
        """ Get model generated ids for future tokens

        Args:
            ids: ids of initial sequence 

        Returns:
            (torch.Tensor): top k possible ids for next token in sequence
        """

        output = self.model(ids)
        output = torch.argsort(output, dim=2, descending=True)
        output = output[:, :, :self.k]
        args = torch.randperm(self.k)
        output = output[:, :, args]

        return output

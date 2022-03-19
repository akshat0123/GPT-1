from collections import defaultdict
from typing import Tuple, Dict, List
import pickle, re

from tqdm import trange, tqdm


class BytePairTokenizer:


    def __init__(self) -> None:
        """ Initialize byte pair tokenizer
        """

        self.corpus = defaultdict(int)
        self.vocab = defaultdict(int) 
        self.vocab_to_index = {}
        self.index_to_vocab = {}
        self.eow = ' </w>'
        self.eol = '</eol> '
        self.unk = '<unk>'
        self.pad = '<pad>'


    def add_line_to_corpus(self, line: str) -> None:
        """ Add line to corpus

        Args:
            data: line to add
        """
        tokens = [token.strip() for token in line.split(' ')]
        tokens = [token for token in tokens if len(token) > 0]
        tokens = [self.eol] + [' '.join(list(t)) + self.eow for t in tokens]
        for token in tokens:
            self.corpus[token] += 1


    def add_line_to_vocab(self, line):
        """ Add characters from line to vocabulary

        Args:
            data: line to add characters from
        """
        chars = set(list(line.replace(' ', '')))
        for char_ in chars:
            self.vocab[tuple(char_)] += 1


    def trim_corpus(self, count: int) -> None:
        """ Trim members of corpus with a frequency less than provided count

        Args:
            count: frequency threshold for removing corpus words
        """

        keys = list(self.corpus.keys())
        for key in tqdm(keys):
            if self.corpus[key] < count:
                del(self.corpus[key])


    def merge_max_pair(self) -> bool:
        """ Get maximum frequency byte pair from corpus and merge these byte
            pairs for every member of the corpus 

        Returns:
            (bool): whether or not merging is successful
        """

        maxpair = self.get_max_pair()
        success = False

        if maxpair is not None:
            search = r'(^|\s)' + re.escape(' '.join(maxpair)) + r'(\s|$)'
            replace = ' ' + ''.join(maxpair) + ' '

            success = True

            words = list(self.corpus.keys())
            for word in words:

                if re.search(search, word) is not None:
                    try:
                        replacement = re.sub(search, replace, word).strip()
                        self.corpus[replacement] = self.corpus[word]
                        del(self.corpus[word])
                        self.vocab[maxpair] += 1

                    except:
                        del(self.corpus[word])

        return success


    def get_max_pair(self) -> Tuple[str, str]:
        """ Get maximum frequency byte pair from corpus

        Returns:
            (Tuple[str, str]): maximum frequency byte pair
        """

        pairs = defaultdict(int)
        for word in self.corpus.keys():
            tokens = word.split(' ')

            for i in range(1, len(tokens)):
                pair = tuple(tokens[i-1:i+1])
                pairs[pair] += 1

        maxpair = None
        if len(pairs) > 0:
            maxpair = max(pairs, key=pairs.get)

        return maxpair


    def build_indices(self) -> None:
        """ Build indices mapping vocabulary to indices and indices to
            vocabulary
        """

        keys = list(self.vocab.keys())
        for i in range(len(keys)):
            pair = ''.join(keys[i])
            self.vocab_to_index[pair] = i
            self.index_to_vocab[i] = pair

        for token in [self.eow, self.eol, self.unk, self.pad]:
            idx = len(self.vocab_to_index)
            pair = ''.join(token)
            self.vocab_to_index[pair] = idx
            self.index_to_vocab[idx] = pair


    def save(self, path: str) -> None:
        """ Save tokenizer at provided path

        Args:
            path: filepath where tokenizer should be saved
        """

        checkpoint = {
            'corpus': self.corpus,
            'vocab': self.vocab,
            'word_token': self.eow,
            'vocab_to_index': self.vocab_to_index,
            'index_to_vocab': self.index_to_vocab 
        }

        with open(path, 'wb') as outfile:
            pickle.dump(checkpoint, outfile)

    
    def load(self, path):
        """ Load tokenizer from provided path

        Args:
            path: filepath where tokenizer should be loaded from 
        """

        with open(path, 'rb') as infile:
            checkpoint = pickle.load(infile)
            self.corpus = checkpoint['corpus'] 
            self.vocab = checkpoint['vocab'] 
            self.eow = checkpoint['word_token'] 
            self.vocab_to_index = checkpoint['vocab_to_index'] 
            self.index_to_vocab  = checkpoint['index_to_vocab'] 


    def get_end_of_line_token(self) -> str:
        return self.eol


    def get_end_of_word_token(self) -> str:
        return self.eow


    def get_pad_token(self) -> str:
        return self.pad


    def get_token_id(self, token: str):
        return self.vocab_to_index[token]


    def get_token_ids(self, token: str) -> List[int]:
        """ Get list of ids for bytes in token

        Args:
            token: string to segment

        Return:
            (List[int]): list of ids for bytes in token
        """

        token = self.segment_token(token)
        bytes_ = token.split(' ')

        if bytes_[-1] == self.eow.strip():
            bytes_[-1] = self.eow 

        ids = []
        for byte_ in bytes_:
            if byte_ in self.vocab_to_index:
                ids.append(self.vocab_to_index[byte_])

            else:
                ids.append(self.vocab_to_index[self.unk])

        return ids


    def segment_token(self, token: str) -> str:
        """ Segment token into bytes and return whitespace separated bytes

        Args:
            token: token to segment

        Returns:
            (str): whitespace separated bytes
        """

        while True:
            pairs = self.get_pairs_from_token(token)

            if len(pairs) == 0:
                break

            maxpair = max(pairs, key=pairs.get)
            search = r'(^|\s)' + re.escape(' '.join(maxpair)) + r'(\s|$)'
            replace = ' ' + ''.join(maxpair) + ' '
            token = re.sub(search, replace, token).strip()

        return token


    def get_pairs_from_token(self, token: str) -> Dict[Tuple[str, str], int]:
        """ Get frequency dictionary of byte pairs from token

        Args:
            token: token to get byte pairs from

        Returns:
            (Dict[Tuple[str, str], int]): map of tuple of byte pairs to the
                                          frequency of the pair in the corpus
        """


        bytes_ = token.split(' ')

        pairs = defaultdict(int)
        for i in range(1, len(bytes_)):
            pair = tuple(bytes_[i-1:i+1])

            if pair in self.vocab:
                pairs[pair] += self.vocab[pair]

        return pairs

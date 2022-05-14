from typing import Tuple, Dict, List
from collections import defaultdict
import json, re

from nltk import wordpunct_tokenize, sent_tokenize
from tqdm import trange, tqdm


class BytePairTokenizer:


    def __init__(self, freqs: Dict[str, int], vocab_to_idx: Dict[str, int],
                 idx_to_vocab: Dict[int, str]):
        """ Initialize byte pair tokenizer

        Args:
            freqs: frequency dictionary of vocabulary
            vocab_to_index: map of vocabulary words to indices
            index_to_vocab: map of vocabulary indices to words
        """

        self.vocab_to_idx = vocab_to_idx
        self.idx_to_vocab = idx_to_vocab
        self.freqs = freqs
        self.sol = '<line/>'
        self.eol = '</line>'
        self.pad = '<pad>'
        self.unk = '<unk>'
        self.eow = '</w>'


    def get_sol(self) -> str:
        return self.sol


    def get_eol(self) -> str:
        return self.eol


    def get_pad(self) -> str:
        return self.pad


    def get_unk(self) -> str:
        return self.unk


    def get_eow(self) -> str:
        return self.eow


    def get_byte(self, byte_id: int) -> str:
        return self.idx_to_vocab[byte_id]


    def get_byte_id(self, byte: str) -> int:
        unk_id = self.vocab_to_idx[self.unk]
        bid = self.vocab_to_idx[byte] if byte in self.vocab_to_idx else unk_id
        return bid


    def get_byte_ids(self, bytes_):
        """ Get byte ids for each byte in provided list
        """

        ids = []
        for byte in bytes_:
            if byte in self.vocab_to_idx:
                ids.append(self.vocab_to_idx[byte])

            else:
                ids.append(self.vocab_to_idx[self.unk])

        return ids


    def get_bytes(self, byte_ids: List[int]) -> List[str]:
        """ Given a list of byte ids return corresponding bytes

        Args:
            byte_ids: list of byte ids

        Returns:
            (List[str]): list of bytes
        """

        tokens = []
        for byte_id in byte_ids:
            tokens.append(self.idx_to_vocab[byte_id])

        return tokens


    def merge_bytes(self, bytes_: List[str]) -> List[str]:
        """ Return list of bytes with max pair merged

        Args:
            bytes_: list to merge max pair in

        Returns:
            (List[str]): list of bytes with all max pair occurrences merged
        """

        bytes_, merged = self.merge_max_pair(bytes_)
        while merged:
            bytes_, merged = self.merge_max_pair(bytes_)

        return bytes_ 


    def merge_max_pair(self, bytes_: List[str]) -> (List[str], bool):
        """ Takes in a list of bytes and merges the max pair if possible

        Args:
            bytes_: list of bytes to merge max pair in

        Returns:
            (bytes_): list of bytes with max pair merged
            (bool): flag indicating whether merge occurred
        """

        max_pair = self.get_max_pair_idxs(bytes_)
        merged = True if max_pair is not None else False

        if merged:
            bytes_ = bytes_[:max_pair[0]] + \
                    [''.join(bytes_[max_pair[0]:max_pair[1]+1])] + \
                    bytes_[max_pair[1]+1:]

        return bytes_, merged


    def get_max_pair_idxs(self, bytes_) -> Tuple[int, int]:
        """ Get index of maximum byte pair in list of bytes

        Args:
            bytes_: list of bytes to find maximum pair from

        Returns:
            (Tuple[int, int]): maximum frequency byte pair
        """

        pairs = {}
        for i in range(1, len(bytes_)):
            pair = ''.join(bytes_[i-1:i+1])
            if pair in self.freqs:
                pairs[(i-1, i)] = self.freqs[pair]

        return None if len(pairs) == 0 else max(pairs, key=pairs.get) 


    def save(self, path: str) -> None:

        with open(f'{path}/freqs.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.freqs, outfile, indent=4, ensure_ascii=False)

        with open(f'{path}/vocab_to_idx.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.vocab_to_idx, outfile, indent=4, ensure_ascii=False)

        with open(f'{path}/idx_to_vocab.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.idx_to_vocab, outfile, indent=4, ensure_ascii=False)


    @staticmethod
    def load(path: str) -> 'BytePairTokenizer':

        with open(f'{path}/freqs.json', 'r', encoding='utf-8') as infile:
            freqs = json.load(infile)

        with open(f'{path}/vocab_to_idx.json', 'r', encoding='utf-8') as infile:
            vocab_to_idx = json.load(infile)

        with open(f'{path}/idx_to_vocab.json', 'r', encoding='utf-8') as infile:
            idx_to_vocab = json.load(infile)

        return BytePairTokenizer(freqs, vocab_to_idx, idx_to_vocab)


    @staticmethod
    def train_bpe(filepaths: List[str], mincount: int, merges: int) ->
                  'BytePairtokenizer':
        """ Create trained byte pair tokenizer

        Args:
            filepaths: files to use for training
            mincount: minimum frequency of vocabulary to consider for tokenizer
            merges: number of merges in training

        Returns:
            ('BytePairTokenizer'): trained byte pair tokenizer
        """

        vocab = create_vocab(filepaths)
        truncate_vocab(vocab, mincount)
        vocab = prepare_bpe_vocab(vocab)

        for i in trange(merges, desc='Merging'):
            pairs = get_stats(vocab)

            if len(pairs) == 0:
                break

            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)

        freqs = count_byte_freqs(vocab)
        vocab_to_idx, idx_to_vocab = create_vocab_maps(freqs)
        return BytePairTokenizer(freqs, vocab_to_idx, idx_to_vocab)


def create_vocab(filepaths: List[str]) -> Dict[str, int]:
    """ Create dictionary of vocabulary frequencies in given list of files

    Args:
        filepaths: list of filepaths to collect vocabulary from

    Returns:
        (Dict[str, int]): dictionary mapping vocabulary terms to their frequency 
    """

    vocab = defaultdict(int)
    for path in tqdm(filepaths, desc='Creating vocabulary'):
        text = open(path, 'r', encoding='utf-8-sig').read()
        sentences = sent_tokenize(text)

        for sentence in sentences:
            tokens = wordpunct_tokenize(sentence)

            for token in tokens:
                vocab[token] += 1

    return vocab


def truncate_vocab(vocab: Dict[str, int], mincount: int) -> None:
    """ Truncate vocabulary dictionary based on a minimum count

    Args:
        vocab: frequency mapping dictionary to truncate
        mincount: minimum count for members of dictionary (words with lower
                  frequencies will be removed)
    """

    tokens = list(vocab.keys())
    for token in tokens:
        if vocab[token] < mincount:
            del(vocab[token])


def prepare_bpe_vocab(vocab: Dict[str, int]) -> Dict[str, int]:
    """ Prepare vocabulary frequency dictionary for byte-pair generation.
        End-of-word byte '</w>' added to words, every character separated by space

    Args:
        vocab: vocabulary frequency dictionary to prepare

    Returns:
        (Dict[str, int]): byte-pair ready vocabulary frequency dictionary
    """

    bpe_vocab = {}
    for token in vocab:
        ntoken = ' '.join(list(token)) + ' </w>'
        bpe_vocab[ntoken] = vocab[token]

    return bpe_vocab


def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """ Count all bytepairs in a dictionary containing vocabulary frequencies

    Args:
        vocab: dictionary mapping words to their frequency

    Returns:
        (Dict[Tuple[str, str], int]): dictionary containing byte pair
                                      frequencies
    """

    pairs = defaultdict(int)

    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq

    return pairs


def merge_vocab(pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
    """ Merge all instances of given byte pair in vocabulary frequency
        dictionary

    Args:
        pair: byte pair to merge
        v_in: vocabulary to merge byte pair int

    Returns:
        (Dict[str, int]): resulting vocabulary with all instances of given byte
                          pair merged
    """

    bigram = re.escape(' '.join(pair))
    v_out = {}

    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out


def count_byte_freqs(vocab: Dict[str, int]) -> Dict[str, int]:
    """ Count byte frequencies in vocabulary frequency dictionary

    Args
        vocab: dictionary of vocab frequencies to count byte pairs in

    Returns:
        (Dict[str, int]): dictionary of byte frequencies
    """

    freqs = defaultdict(int)
    for word in vocab:
        bytes_ = word.split(' ')
        for byte in bytes_:
            freqs[byte] += 1

    for token in ['<line/>', '</line>', '<pad>', '<unk>']:
        freqs[token] += 1

    return freqs


def create_vocab_maps(freqs: Dict[str, int]) -> (Dict[str, int], Dict[int, str]):
    """ Create map of vocabulary terms to indices and vice versa. Word indices
        are in order of their frequency in the provided vocabulary 

    Args:
        freqs: dictionary mapping vocabulary terms to their frequencies

    Returns:
        (Dict[str, int]): dictionary mapping vocab to indices
        (Dict[int, str]): dictionary mapping indices to vocab
    """

    ordered_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    vocab_to_idx, idx_to_vocab = {}, {}
    for i in range(len(ordered_freqs)):
        word, freq = ordered_freqs[i]
        vocab_to_idx[word] = i
        idx_to_vocab[i] = word

    return vocab_to_idx, idx_to_vocab



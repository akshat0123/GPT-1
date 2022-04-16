from typing import Tuple, Dict, List
from collections import defaultdict
import json, re

from tqdm import trange, tqdm


class BytePairTokenizer:


    def __init__(self, freqs: Dict[str, int], vocab_to_idx: Dict[str, int],
                 idx_to_vocab: Dict[int, str]):
        """ Initialize byte pair tokenizer

        Args:
            freqs: dict mapping vocab to frequenies
            vocab_to_idx: dict mapping vocab to indices
            idx_to_vocab: dict mapping indices to vocab
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


    def get_byte_id(self, byte: str) -> int:
        """ Get byte id for byte

        Args:
            byte: byte string

        Returns:
            (int): byte id
        """

        if byte in self.vocab_to_idx:
            bid = self.vocab_to_idx[byte]

        else:
            bid = self.vocab_to_idx[self.unk]

        return bid


    def get_byte_ids(self, bytes_: List[str]) -> List[int]:
        """ Get list of byte ids for list of bytes

        Args:
            bytes_: list of bytes

        Returns:
            (List[int]): list of byte ids
        """

        ids = []
        for byte in bytes_:
            if byte in self.vocab_to_idx:
                ids.append(self.vocab_to_idx[byte])

            else:
                ids.append(self.vocab_to_idx[self.unk])

        return ids


    def merge_bytes(self, bytes_: List[str]) -> List[str]:
        """ Merge list of bytes until no longer possible

        Args:
            bytes_: list of bytes to merge

        Returns:
            (List[str]): list of merged bytes
        """

        bytes_, merged = self.merge_max_pair(bytes_)
        while merged:
            bytes_, merged = self.merge_max_pair(bytes_)

        return bytes_ 


    def merge_max_pair(self, bytes_: List[str]) -> (List[str], bool):
        """ Merge maximum frequency byte pair and return list of bytes with
            maximum pair merged

        Args:
            bytes_: list of bytes to merge max pair in

        Returns:
            (List[str]): list of bytes with max byte pair merged
            (bool): flag indicating whether there was a pair available for
                    merging
        """

        max_pair = self.get_max_pair_idxs(bytes_)
        merged = True if max_pair is not None else False

        if merged:
            bytes_ = bytes_[:max_pair[0]] + \
                    [''.join(bytes_[max_pair[0]:max_pair[1]+1])] + \
                    bytes_[max_pair[1]+1:]

        return bytes_, merged


    def get_max_pair_idxs(self, bytes_: List[str]) -> Tuple[int, int]:
        """ Return indices of byte pair with highest frequency

        Args:
            bytes_: list of bytes

        Returns:
            (Tuple[int, int]): tuple of indices where highest frequency byte
                               pair is located in bytes list
        """

        pairs = {}
        for i in range(1, len(bytes_)):
            pair = ''.join(bytes_[i-1:i+1])
            if pair in self.freqs:
                pairs[(i-1, i)] = self.freqs[pair]

        return None if len(pairs) == 0 else max(pairs, key=pairs.get) 


    def save(self, path: str) -> None:
        """ Save tokenizer checkpoint

        Args:
            path: directory to save tokenizer
        """

        with open(f'{path}/freqs.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.freqs, outfile, indent=4, ensure_ascii=False)

        with open(f'{path}/vocab_to_idx.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.vocab_to_idx, outfile, indent=4, ensure_ascii=False)

        with open(f'{path}/idx_to_vocab.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.idx_to_vocab, outfile, indent=4, ensure_ascii=False)


    @staticmethod
    def load(path: str) -> 'BytePairTokenizer':
        """ Load BytePairTokenizer instance from checkpoint

        Args:
            path: path of checkpoint

        Returns:
            (BytePairTokenizer): instantiated BytePairTokenizer
        """

        with open(f'{path}/freqs.json', 'r', encoding='utf-8') as infile:
            freqs = json.load(infile)

        with open(f'{path}/vocab_to_idx.json', 'r', encoding='utf-8') as infile:
            vocab_to_idx = json.load(infile)

        with open(f'{path}/idx_to_vocab.json', 'r', encoding='utf-8') as infile:
            idx_to_vocab = json.load(infile)

        return BytePairTokenizer(freqs, vocab_to_idx, idx_to_vocab)


    @staticmethod
    def train_bpe(filepaths: str, merges: int) -> 'BytePairTokenizer':
        """ Train byte pair tokenizer from list of files

        Args:
            filepaths: path to list of newline-separated filepaths to train
                       tokenizer on
            merges: number of times to merge corpus

        Returns:
            (BytePairTokenizer): trained byte pair tokenizer
        """

        vocab = create_vocab(filepaths)

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
    """ Create vocabulary from files in list of filepaths

    Args:
        filepaths: list of filepaths to create vocabulary from

    Returns:
        (Dict[str, int]): dictionary mapping vocabulary words to frequencies
    """

    vocab = defaultdict(int)
    for path in tqdm(filepaths, desc='Creating vocabulary'):
        lines = open(path, encoding='utf-8-sig').readlines()
        lines = [line.strip() for line in lines]

        for line in lines:
            tokens = [' '.join(list(t)) + ' </w>' for t in line.split(' ')]

            for token in tokens:
                vocab[token] += 1

    return vocab


def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """ Count byte pairs in vocabulary

    Args:
        vocab: dictionary mapping vocabulary to frequencies

    Returns:
        (Dict[Tuple[str, str], int]): dictionary mapping byte pairs to
                                      frequencies 
    """

    pairs = defaultdict(int)

    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq

    return pairs


def merge_vocab(pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
    """ Merge given byte pair in vocabulary

    Args:
        pair: byte pair to join in vocabulary
        v_in: input vocabulary

    Returns:
        (Dict[str, int]): vocabulary with given byte pair merged
    """

    bigram = re.escape(' '.join(pair))
    v_out = {}

    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out


def count_byte_freqs(vocab: Dict[str, int]) -> Dict[str, int]:
    """ Count frequency of bytes

    Args:
        vocab: dictionary of vocabulary mapped to frequency

    Returns:
        (Dict[str, int]): dictionary of bytes mapped to frequency
    """

    freqs = defaultdict(int)
    for word in vocab:
        bytes_ = word.split(' ')
        for byte in bytes_:
            freqs[byte] += 1

    for token in ['<line/>', '</line>', '<pad>', '<unk>']:
        freqs[token] += 1

    return freqs


def create_vocab_maps(freqs: Dict[str, int]) -> (Dict[str, int], \
                      Dict[int, str]):
    """ Create map of bytes to indices and map of indices to bytes from given
        map of bytes to their frequencies

    Args:
        freqs: map of bytes to frequencies

    Returns:
        (Dict[str, int]): map of bytes to indices
        (Dict[int, str]): map of indices to bytes
    """

    ordered_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    vocab_to_idx, idx_to_vocab = {}, {}
    for i in range(len(ordered_freqs)):
        word, freq = ordered_freqs[i]
        vocab_to_idx[word] = i
        idx_to_vocab[i] = word

    return vocab_to_idx, idx_to_vocab



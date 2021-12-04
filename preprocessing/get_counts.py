"""
Script to get vocabulary counts from aclImdb dataset
"""
import os

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from tqdm import tqdm
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-o', '--outpath', required=True)
    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath

    datasets = ['test', 'train']
    valences = ['neg', 'pos']
    counts = {}

    tokenizer = Tokenizer(English().vocab)

    # Tokenize all files and count term frequencies
    for dataset in tqdm(datasets, desc='Processing files'):
        for valence in valences:
            path = f'{inpath}/{dataset}/{valence}'
            files = [f'{path}/{r}' for r in os.listdir(path)]

            for tpath in tqdm(files):
                text = open(tpath, 'r').read()
                tokens = tokenizer(text)

                for token in tokens:
                    if token.text in counts:
                        counts[token.text] += 1

                    else:
                        counts[token.text] = 1

    # Sort vocabulary by frequency and store
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    with open(f'{outpath}/counts.txt', 'w') as outfile:
        for word, count in tqdm(counts, desc='Writing counts'):
            outfile.write(f'{word} {count}\n')



if __name__ == '__main__':
    main()

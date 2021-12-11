""" 
Script used to split dataset into a single file where each line contains a
specific number of tokens (defined by 'window')

Usage:
    python split_dataset -i <input path> -o <output path> -w <window size>
"""
import argparse, os

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-o', '--outpath', required=True)
    parser.add_argument('-w', '--window', type=int, required=True)
    args = parser.parse_args()
    inpath = args.inpath
    outpath = args.outpath
    window = args.window + 1

    bookfiles = [bookfile.strip() for bookfile in open(inpath, 'r').readlines()]
    tokenizer = Tokenizer(English().vocab)
    counts = {}

    countfile = os.path.join(outpath, 'counts.txt')
    datafile = os.path.join(outpath, 'data.txt')

    with open(datafile, 'w') as outfile:
        for bookfile in tqdm(bookfiles):

            pars = [
                '<START> ' + l.strip() + ' <END>' for l \
                in open(bookfile, 'r').readlines() \
                if len(l.strip()) > 0
            ]

            tokens = []
            for par in pars:
                tokens += [t.text.strip() for t in tokenizer(par)]

            for token in tokens:
                if token in counts:
                    counts[token] += 1
                else:
                    counts[token] = 1

            start, end = 0, window
            while start < len(tokens):
                line = ' '.join(tokens[start:end])
                outfile.write(line + '\n')
                start += window
                end += window

    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    with open(countfile, 'w') as outfile:
        for word, count in counts:
            outfile.write(f'{word}\t{count}' + '\n')


if __name__ == '__main__':
    main()

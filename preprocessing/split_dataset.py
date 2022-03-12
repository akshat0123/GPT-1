from multiprocessing import Pool
from itertools import repeat
import argparse

from tqdm import tqdm

from model.tokenizer import BytePairTokenizer


def tokenize_file(tokenizer, inpath, outpath, window_size):

    lines = [line.strip().split(' ') for line in open(inpath, 'r').readlines()]
    window = []

    with open(outpath, 'w') as outfile:
        for line in lines:

            if len(line) > 0:
                tokenized = [tokenizer.tokenize(token) for token in line]
                tokenized = [tokenizer.eol] + tokenized
                window += tokenized

            if len(window) >= window_size:
                outfile.write(' '.join(window[:window_size]) + '\n')
                window = window[window_size:]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-i', '--infile', required=True)
    parser.add_argument('-o', '--outdir', required=True)
    parser.add_argument('-w', '--window_size', required=True, type=int)
    parser.add_argument('-j', '--jobs', required=True, type=int)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    infile = args.infile
    outdir = args.outdir
    window_size = args.window_size
    jobs = args.jobs

    inpaths = [inpath.strip() for inpath in open(infile, 'r').readlines()]
    outpaths = [f'{outdir}/' + inpath.split('/')[-1] for inpath in inpaths]
    bpt = BytePairTokenizer()
    bpt.load(checkpoint)

    current_index = 0
    progress = tqdm(total=len(inpaths))
    while current_index < len(inpaths):

        start = current_index
        end = current_index + jobs

        with Pool(jobs) as pool:

            ingroup = inpaths[start:end]
            outgroup = outpaths[start:end]

            pool.starmap(tokenize_file, zip(repeat(bpt), ingroup, outgroup, repeat(window_size)))
            current_index += jobs
            progress.update(jobs)


if __name__ == '__main__':
    main()

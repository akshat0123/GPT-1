from multiprocessing import Pool
from itertools import repeat
import argparse

from tqdm import tqdm

from model.tokenizer import BytePairTokenizer


def segment_file(tokenizer: 'Tokenizer', inpath: str, outpath: str, 
                 window_size: int) -> None:
    """ Takes in file and segments each line using tokenizer, creating new lines
        of size 'window_size'

    Args:
        tokenizer: tokenizer to use for segmentation
        inpath: path of file to segment
        outpath: path to write new segmented file
        window_size: length of each line in tokens
    """

    lines = open(inpath, 'r').readlines()
    lines = [line.strip().split(' ') for line in lines]
    window = []

    with open(outpath, 'w') as outfile:
        for line in lines:

            if len(line) > 0:
                segmented = [tokenizer.segment(token) for token in line]
                segmented = [tokenizer.eol] + segmented
                window += segmented

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

            pool.starmap(
                segment_file, 
                zip(repeat(bpt), ingroup, outgroup, repeat(window_size))
            )

            current_index += jobs
            progress.update(jobs)


if __name__ == '__main__':
    main()

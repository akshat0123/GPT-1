from multiprocessing import Pool
from itertools import repeat
from typing import List
import argparse

from tqdm import tqdm

from model.tokenizer import BytePairTokenizer


def generate_outpaths(inpaths: List[str], outdir: str) -> List[str]:
    """ Create list of outpaths given list of input paths

    Args:
        inpaths: list of input paths
        outdir: output directory

    Returns:
        (List[str]): list of output paths 
    """

    outpaths = []
    for inpath in inpaths:
        filename = inpath.split('/')[-1]
        filename = f'{outdir}/{filename}'
        outpaths.append(filename)

    return outpaths


def tokenize_file_lines(tokenizer: 'Tokenizer', inpath: str, 
                        splitter: str) -> List[List[str]]:
    """ Tokenize file lines to ids

    Args:
        tokenizer: tokenizer for file segmentation
        inpath: path to file to segment
        splitter: string to split lines on

    Returns:
        (List[List[str]]): list of ids for tokens in each line of the file
    """

    lines = open(inpath, 'r').read()
    lines = [line.strip() for line in lines.split(splitter)]
    lines = [line for line in lines if len(line) > 0]
    ids = []

    for line in lines:
        line_tokens = []
        tokens = [' '.join(list(t)) for t in line.split(' ')]
        tokens = [t for t in tokens if len(t) > 0]
        tokens = [t + tokenizer.get_end_of_word_token() for t in tokens]

        for token in tokens:
            line_tokens += tokenizer.get_token_ids(token)

        ids.append(line_tokens)

    return ids


def write_tokenized(tokenizer, inpath, outpath, splitter) -> None:
    """ Tokenize file and write tokenized ids to new location 

    Args:
        tokenizer: tokenizer for file segmentation
        inpath: path to file to segment
        outpath: path to place new file of ids
        splitter: string to split lines on
    """

    line_ids = tokenize_file_lines(tokenizer, inpath, splitter)
    eol = tokenizer.get_end_of_line_token()
    eol_idx = str(tokenizer.get_token_id(eol))

    window = []
    with open(outpath, 'w') as outfile:

        for ids in line_ids:
            ids = [str(x) for x in ids]
            line_ids = [eol_idx] + ids + [eol_idx]
            line_out = ' '.join(line_ids)
            outfile.write(f'{line_out}\n')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-i', '--infile', required=True)
    parser.add_argument('-o', '--outdir', required=True)
    parser.add_argument('-j', '--jobs', type=int, required=True)
    parser.add_argument('-s', '--splitter', type=str, default='\n')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    infile = args.infile
    outdir = args.outdir
    jobs = args.jobs
    splitter = args.splitter

    tokenizer = BytePairTokenizer()
    tokenizer.load(checkpoint)

    inpaths = [x.strip() for x in open(infile, 'r').readlines()]
    outpaths = generate_outpaths(inpaths, outdir)

    current_index = 0
    progress = tqdm(total=len(inpaths))
    while current_index < len(inpaths):

        start = current_index
        end = current_index + jobs

        with Pool(jobs) as pool:

            ingroup = inpaths[start:end]
            outgroup = outpaths[start:end]

            pool.starmap(
                write_tokenized, 
                zip(repeat(tokenizer), 
                ingroup, 
                outgroup, 
                repeat(splitter))
            )
            current_index += jobs
            progress.update(jobs)


if __name__ == '__main__':
    main()

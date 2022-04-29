from argparse import ArgumentParser
from multiprocessing import Pool
from itertools import repeat
from typing import List

from tqdm import tqdm

from model.tokenizer import BytePairTokenizer, count_byte_freqs


def tokenize_file(filepath: str, outdir: str, tokenizer: BytePairTokenizer,
                  line_length: int) -> None:
    """ Tokenize given file and write token ids to new file

    Args:
        filepath: filepath of file to tokenize
        outdir: directory to store tokenized file
        tokenizer: tokenizer instance to use to tokenize file
    """

    outpath = f"{outdir}/{filepath.split('/')[-1]}"
    lines = open(filepath, encoding='utf-8-sig').readlines()

    tokens = []
    for line in lines:
        if len(line) > 1:
            tokens += get_line_ids(line, tokenizer)

    start, end = 0, line_length 
    with open(outpath, 'w') as outfile:
        while start < len(tokens):
            if len(tokens[start:end]) == line_length:
                outstr = ' '.join([str(x) for x in tokens[start:end]])
                outfile.write(f'{outstr}\n')
            start += line_length
            end += line_length


def get_line_ids(line: str, tokenizer: BytePairTokenizer) -> List[int]:
    """ Take line and return list of token ids for line

    Args:
        line: line to tokenize 
        tokenizer: tokenizer to use to tokenize line

    Return:
        (List[int]): list of token ids
    """

    tokens = line.strip().split(' ')
    tokens = [list(t) + [tokenizer.get_eow()] for t in tokens]

    lineids = []
    for token in tokens:
        token = tokenizer.merge_bytes(token)
        ids = tokenizer.get_byte_ids(token)
        lineids += ids
    
    sol_id = tokenizer.get_byte_id(tokenizer.get_sol())
    eol_id = tokenizer.get_byte_id(tokenizer.get_eol())
    lineids = [sol_id] + lineids + [eol_id]
    return lineids


def main():

    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-o', '--outdir', required=True)
    parser.add_argument('-l', '--line_length', required=True, type=int)
    parser.add_argument('-j', '--jobs', required=True, type=int)
    args = parser.parse_args()
    line_length = args.line_length
    checkpoint = args.checkpoint
    outdir = args.outdir
    inpath = args.inpath
    jobs = args.jobs

    filepaths = [line.strip() for line in open(inpath).readlines()]
    tokenizer = BytePairTokenizer.load(checkpoint)

    progress = tqdm(total=len(filepaths))
    start, end = 0, jobs
    while start < len(filepaths):

        paths = filepaths[start:end]

        with Pool(jobs) as pool:
            pool.starmap(
                tokenize_file, 
                zip(
                    paths, 
                    repeat(outdir), 
                    repeat(tokenizer),
                    repeat(line_length)
                )
            )

        progress.update(len(paths))
        start += jobs
        end += jobs


if __name__ == '__main__':
    main()

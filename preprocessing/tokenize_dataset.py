from argparse import ArgumentParser
from multiprocessing import Pool
from itertools import repeat

from tqdm import tqdm

from model.tokenizer import BytePairTokenizer, count_byte_freqs


def tokenize_file(filepath: str, outdir: str, tokenizer: BytePairTokenizer) \
                  -> None:
    """ Tokenize given file and write token ids to new file

    Args:
        filepath: filepath of file to tokenize
        outdir: directory to store tokenized file
        tokenizer: tokenizer instance to use to tokenize file
    """

    outpath = f"{outdir}/{filepath.split('/')[-1]}"
    lines = open(filepath, encoding='utf-8-sig').readlines()

    with open(outpath, 'w') as outfile:
        for line in lines:
            if len(line) > 1:
                lineids = get_line_ids(line, tokenizer)
                outfile.write(f"{lineids}\n")


def get_line_ids(line: str, tokenizer: BytePairTokenizer) -> str:
    """ Take line and return string of space-separated token ids for line

    Args:
        line: line to tokenize 
        tokenizer: tokenizer to use to tokenize line

    Return:
        (str): string of white-space separated token ids
    """

    tokens = line.strip().split(' ')
    tokens = [list(t) + [tokenizer.get_eow()] for t in tokens]

    lineids = ""
    for token in tokens:
        token = tokenizer.merge_bytes(token)
        ids = tokenizer.get_byte_ids(token)
        idstring = ' '.join([str(x) for x in ids])
        lineids += ' ' + idstring

    return lineids


def main():

    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-o', '--outdir', required=True)
    parser.add_argument('-j', '--jobs', required=True, type=int)
    args = parser.parse_args()
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
                zip(paths, repeat(outdir), repeat(tokenizer))
            )

        progress.update(len(paths))
        start += jobs
        end += jobs


if __name__ == '__main__':
    main()

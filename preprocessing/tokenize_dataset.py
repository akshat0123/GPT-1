from multiprocessing import Pool
from itertools import repeat
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


def tokenize_file(tokenizer: 'Tokenizer', inpath: str) -> List[str]:
    """ Tokenize file to ids

    Args:
        tokenizer: tokenizer for file segmentation
        inpath: path to file to segment

    Returns:
        (List[str]): list of ids for tokens in file
    """

    lines = [line.strip() for line in open(inpath).readlines()]
    lines = [line for line in lines if len(line) > 0]
    ids = []

    for line in lines:
        tokens = [' '.join(list(t)) for t in line.split(' ')]
        tokens = [t for t in tokens if len(t) > 0]
        tokens = [t + tokenizer.get_end_of_word_token() for t in tokens]

        for token in tokens:
            ids += tokenizer.get_token_ids(token)

    return ids


def write_tokenized(tokenizer, inpath, outpath, window_size) -> None:
    """ Tokenize file and write tokenized ids to new location with each line at
        specified "window_size"

    Args:
        tokenizer: tokenizer for file segmentation
        inpath: path to file to segment
        outpath: path to place new file of ids
        window_size: length of each line in tokens
    """

    ids = tokenize_file(tokenizer, inpath)

    with open(outpath, 'w') as outfile:
        while len(ids) >= window_size:
            idstrs = [str(id_) for id_ in ids[:window_size]]
            outfile.write(' '.join(idstrs) + '\n')
            ids = ids[window_size:]

        if len(ids) > 0:
            ids = tokenizer.pad_ids(ids, window_size)
            idstrs = [str(id_) for id_ in ids[:window_size]]
            outfile.write(' '.join(idstrs) + '\n')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-i', '--infile')
    parser.add_argument('-o', '--outdir')
    parser.add_argument('-w', '--window_size', type=int)
    parser.add_argument('-j', '--jobs', type=int)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    infile = args.infile
    outdir = args.outdir
    window_size = args.window_size
    jobs = args.jobs

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

            pool.starmap(write_tokenized, zip(repeat(tokenizer), ingroup, outgroup, repeat(window_size)))
            current_index += jobs
            progress.update(jobs)


if __name__ == '__main__':
    main()

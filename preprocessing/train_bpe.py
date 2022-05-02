from argparse import ArgumentParser

from model.tokenizer import BytePairTokenizer


def main():

    parser = ArgumentParser()
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-o', '--outpath', required=True)
    parser.add_argument('-m', '--merges', required=True, type=int)
    parser.add_argument('-n', '--mincount', required=True, type=int)
    args = parser.parse_args()
    outpath = args.outpath
    inpath = args.inpath
    merges = args.merges
    mincount = args.mincount

    filepaths = [path.strip() for path in open(inpath).readlines()]
    tokenizer = BytePairTokenizer.train_bpe(filepaths, mincount, merges)
    tokenizer.save(f'{outpath}')


if __name__ == '__main__':
    main()

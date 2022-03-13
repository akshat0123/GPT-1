import argparse

from tqdm import trange, tqdm

from model.tokenizer import BytePairTokenizer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-m', '--merges', required=True, type=int)
    parser.add_argument('-t', '--trim', required=True, type=int, default=10)
    args = parser.parse_args()
    inpath = args.inpath
    checkpoint = args.checkpoint
    merges = args.merges
    trim = args.trim

    datapaths = [datapath.strip() for datapath in open(inpath, 'r').readlines()]
    tokenizer = BytePairTokenizer()

    for datapath in tqdm(datapaths):
        lines = open(datapath, 'r').readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]

        for line in lines:
            tokenizer.add_line_to_corpus(line)
            tokenizer.add_line_to_vocab(line)

    tokenizer.trim_corpus(trim)

    for i in trange(merges):
        success = tokenizer.merge_max_pair()
        if not success: 
            break

    tokenizer.build_indices()
    tokenizer.save(checkpoint)


if __name__ == '__main__':
    main()

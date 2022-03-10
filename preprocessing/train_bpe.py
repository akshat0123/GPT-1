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
    bpt = BytePairTokenizer()

    for datapath in tqdm(datapaths):
        data = open(datapath, 'r').read().split(' ')
        bpt.add_to_corpus(data)
        bpt.add_to_vocab(data)

    bpt.trim_corpus(trim)

    for i in trange(merges):
        success = bpt.merge_max_pair()
        if not success: 
            break

    bpt.build_indices()
    bpt.save(checkpoint)


if __name__ == '__main__':
    main()

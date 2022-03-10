import argparse

from tqdm import tqdm

from model.tokenizer import BytePairTokenizer


tpath = '/home/akshat/Programs/Decoders/checkpoints/tokenizer.pickle'
ipath = '/home/akshat/Programs/Decoders/data/files.txt'
opath = '/home/akshat/Programs/Decoders/data/data.txt'
window_size = 512


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-i', '--inpath', required=True)
    parser.add_argument('-o', '--outpath', required=True)
    parser.add_argument('-w', '--window_size', required=True, type=int)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    inpath = args.inpath
    outpath = args.outpath
    window_size = args.window_size

    datapaths = [datapath.strip() for datapath in open(inpath, 'r').readlines()][:10]
    bpt = BytePairTokenizer()
    bpt.load(tpath)
    window = []

    with open(outpath, 'w') as outfile:
        for datapath in tqdm(datapaths):

            lines = [line.strip().split() for line in open(datapath, 'r').readlines()]

            for line in lines:

                if len(line) > 0:
                    line = ' '.join(line)
                    tokenized = bpt.eol + bpt.tokenize(line)
                    window += tokenized.split(' ')

                if len(window) >= window_size:
                    outfile.write(' '.join(window[:window_size]) + '\n')
                    window = window[window_size:]


if __name__ == '__main__':
    main()

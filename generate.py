import argparse, pickle, yaml

from tqdm import trange
import torch

from model.tokenizer import BytePairTokenizer
from model.model import TransformerDecoder


configpath = 'confs/params.yaml'


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--checkpoint_path', required=True)
    # args = parser.parse_args()
    # checkpoint_path = args.checkpoint_path
    checkpoint_path = '/home/apgpu/Programs/Decoders/checkpoints/1.pickle'

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)

    # Load tokenizer and model
    tokenizer = BytePairTokenizer()
    tokenizer.load(**confs['tokenizer'])

    # Initialize model
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    model = TransformerDecoder(**confs['model'])
    model.load_state_dict(checkpoint['model'])

    model.eval()

    vocab_size = 56734
    window_size = 64
    k = 300

    sequence = [tokenizer.get_pad_token() for i in range(window_size-1)]
    sequence += [tokenizer.get_end_of_line_token()]

    start, end = 0, window_size

    for i in trange(20):
        seq = sequence[start:end]

        seq_ids = torch.Tensor([tokenizer.vocab_to_index[t] for t in seq])
        seq_ids = seq_ids.type(torch.LongTensor)
        seq_ids = torch.nn.functional.one_hot(seq_ids[None, :], num_classes=vocab_size)
        seq_ids = seq_ids.type(torch.FloatTensor).to(device=model.device)

        pred = model(seq_ids)[0]
        pred_id = torch.argsort(pred, dim=0, descending=True)[:k]
        pred_id = pred_id[torch.randint(k, (1,)).item()].item()
        pred_token = tokenizer.index_to_vocab[pred_id]

        sequence.append(pred_token)
        start += 1
        end += 1

    print(' '.join(sequence[window_size-1:]))


if __name__ == '__main__':
    main()

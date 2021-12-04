import yaml

from tqdm import trange, tqdm
import torch

from model.model import Embedding, Decoder
from model.dataset import ReviewDataset


def main():

    params = yaml.load(open('confs/params.yml', 'r'), Loader=yaml.Loader)

    dataset = ReviewDataset(
        dpath=params["dataset"]["dpath"],
        vpath=params["dataset"]["vpath"],
        size=params["dataset"]["size"]
    )

    embeddings = Embedding(
        dataset.get_vocab(), 
        params["embedding"]["dim"]
    )

    decoder = Decoder(
        n_layers=params["model"]["n_layers"],
        dim=params["model"]["dim"],
        n_heads=params["model"]["n_heads"]
    )


if __name__ == '__main__':
    main()

from functools import partial
import yaml, os

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import torch

from model.dataset import BooksCorpus


configpath = 'confs/params.yml'


def main():

    confs = yaml.load(open(configpath, 'r'), Loader=yaml.SafeLoader)

    # Load dataset
    dataset = BooksCorpus(**confs['dataset'])

    # Split into train and dev datasets
    train_size = int(0.9 * len(dataset))
    dev_size = len(dataset) - train_size
    train, dev = random_split(dataset, [train_size, dev_size])

    # Initialize train and dev data loaders
    tloader = DataLoader(batch_size=confs['loader']['batch_size'],
                         dataset=train, drop_last=True, shuffle=True)
    dloader = DataLoader(batch_size=confs['loader']['batch_size'], dataset=dev,
                         drop_last=True, shuffle=True)

    embedding_dim = 64
    vocab_size = 1000
    window_size = 128
    sqrt_embedding_dim = torch.sqrt(torch.Tensor([embedding_dim]))

    embedding = torch.nn.Linear(vocab_size, embedding_dim)
    q = torch.nn.Linear(embedding_dim, embedding_dim)
    k = torch.nn.Linear(embedding_dim, embedding_dim)
    v = torch.nn.Linear(embedding_dim, embedding_dim)
    ffl1 = torch.nn.Linear(embedding_dim, embedding_dim)
    relu = torch.nn.ReLU()
    ffl2 = torch.nn.Linear(embedding_dim, embedding_dim)

    for X, Y in tqdm(tloader):

        # Transform ids into one-hot vectors            
        X = torch.nn.functional.one_hot(X, vocab_size)

        # Transform one-hot vectors into embeddings
        X = X.type(torch.FloatTensor)
        X = embedding(X)

        # Input embeddings into self-attention layer
        Q = q(X)
        K = k(X)
        V = v(X)

        A = torch.einsum('ijk,ilk->ijl', Q, K)
        A = torch.tril(A) / sqrt_embedding_dim
        A = torch.nn.functional.softmax(A, dim=2)
        A = torch.einsum('ijk,ijl->ijl', A, V)

        # Add residual connection
        X = X + A

        # Layer normalize
        mu = torch.mean(X, dim=2)[:, :, None]
        sigma = X - mu
        sigma = torch.square(sigma)
        sigma = torch.sum(sigma, dim=2)
        sigma /= embedding_dim
        sigma = torch.sqrt(sigma)[:, :, None]
        X = (X - mu) / sigma

        # First position-wise feed forward layer
        F = torch.cat([ffl1(X[:, i, :])[:, None, :] for i in range(window_size)], dim=1)
            
        # Relu activation
        F = relu(F)

        # Second position-wise feed forward layer
        F = torch.cat([ffl2(F[:, i, :])[:, None, :] for i in range(window_size)], dim=1)

        # Add residual connection
        X = X + F

        # Layer normalize
        mu = torch.mean(X, dim=2)[:, :, None]
        sigma = X - mu
        sigma = torch.square(sigma)
        sigma = torch.sum(sigma, dim=2)
        sigma /= embedding_dim
        sigma = torch.sqrt(sigma)[:, :, None]
        X = (X - mu) / sigma


if __name__ == '__main__':
    main()

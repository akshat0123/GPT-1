from functools import partial
import yaml, os

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import torch

from model.dataset import BooksCorpus


configpath = 'confs/params.yml'


class Embedding(torch.nn.Module):


    def __init__(self, vocab_size, dim):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Linear(vocab_size, dim)


    def forward(self, x):
        return self.embedding(x)


class SelfAttentionLayer(torch.nn.Module):


    def __init__(self, embedding_dim_in, embedding_dim_out, hidden_dim):
        super(SelfAttentionLayer, self).__init__()
        self.q = torch.nn.Linear(embedding_dim_in, embedding_dim_out)
        self.k = torch.nn.Linear(embedding_dim_in, embedding_dim_out)
        self.v = torch.nn.Linear(embedding_dim_in, embedding_dim_out)
        self.dk = torch.sqrt(torch.Tensor([embedding_dim_out]))


    def forward(self, X):

        Q, K, V = self.q(X), self.k(X), self.v(X)
        X = torch.einsum('ijk,ilk->ijl', Q, K)
        X = torch.tril(X) / self.dk
        X = torch.nn.functional.softmax(X, dim=2)
        X = torch.einsum('ijk,ijl->ijl', X, V)
        return X


class PositionWiseFFN(torch.nn.Module):


    def __init__(self, d_in, d_out, activation):
        super(PositionWiseFFN, self).__init__()
        self.ffl = torch.nn.Linear(d_in, d_out)
        self.activation = activation


    def forward(self, X):
        X = torch.cat([
            self.ffl(X[:, i, :])[:, None, :] for i in range(X.shape[1])
        ], dim=1)

        X = self.activation(X)

        return X


class LayerNorm(torch.nn.Module):


    def __init__(self):
        super(LayerNorm, self).__init__()


    def forward(self, X):
        mu = torch.mean(X, dim=2)[:, :, None]
        sigma = torch.square(X - mu)
        sigma = torch.sum(sigma, dim=2) / X.shape[2]
        sigma = torch.sqrt(sigma)[:, :, None]
        X = (X - mu) / sigma
        return X


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
    hidden_dim = 256
    n_heads = 4
    sqrt_embedding_dim = torch.sqrt(torch.Tensor([embedding_dim]))

    embedding = Embedding(vocab_size, embedding_dim)
    heads = [
        SelfAttentionLayer(embedding_dim, embedding_dim//n_heads, hidden_dim) \
        for i in range(n_heads)
    ]
    norm = LayerNorm()
    ff1 = PositionWiseFFN(embedding_dim, hidden_dim, torch.nn.ReLU())
    ff2 = PositionWiseFFN(hidden_dim, embedding_dim, torch.nn.Identity())

    for X, Y in tqdm(tloader):

        # Transform ids into one-hot vectors            
        X = torch.nn.functional.one_hot(X, vocab_size)
        X = X.type(torch.FloatTensor)

        # Transform one-hot vectors into embeddings
        X = embedding(X)

        # Transform with multiheaded attention
        A = torch.cat([
           heads[i](X) for i in range(len(heads)) 
        ], dim=2)

        # Normalization and residual connection
        X = norm(X + A)

        # Position-wise Feed forward networks
        F = ff2(ff1(X))

        # Normalization and residual connection
        X = norm(X + F)


if __name__ == '__main__':
    main()

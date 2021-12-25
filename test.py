from functools import partial
import yaml, os

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import torch

from model.dataset import BooksCorpus


configpath = 'confs/params.yml'


class TransformerDecoder(torch.nn.Module):


    def __init__(self, v, d, dk, n_heads, hidden, n_blocks):
        super(TransformerDecoder, self).__init__()
        self.embedding = Embedding(v, d)
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(d, dk, n_heads, hidden) \
            for i in range(n_blocks)
        ])

        self.linear = torch.nn.Linear(d, v)


    def forward(self, X):

        # Transform one-hot vectors into embeddings
        X = self.embedding(X)

        # Run through transformer blocks
        for block in self.blocks:
            X = block(X)

        # Project to vocabulary space
        X = torch.nn.functional.softmax(self.linear(X), dim=2)

        return X


class Embedding(torch.nn.Module):


    def __init__(self, vocab_size, dim):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Linear(vocab_size, dim)


    def forward(self, x):
        return self.embedding(x)


class TransformerBlock(torch.nn.Module):


    def __init__(self, d, dk, n_heads, hidden):        
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttentionLayer(d, dk, n_heads)
        self.norm = LayerNorm()
        self.ff1 = PositionWiseFFN(d, hidden, torch.nn.ReLU())
        self.ff2 = PositionWiseFFN(hidden, d, torch.nn.Identity())

    
    def forward(self, X):

        # Run through attention
        A = self.attention(X)

        # Normalization and residual connection
        X = self.norm(X + A)

        # Position-wise Feed forward networks
        F = self.ff2(self.ff1(X))

        # Normalization and residual connection
        X = self.norm(X + F)

        return X


class MultiHeadedAttentionLayer(torch.nn.Module):


    def __init__(self, d, dk, n_heads):
        super(MultiHeadedAttentionLayer, self).__init__()

        self.heads = torch.nn.ModuleList([
            SelfAttentionLayer(d, dk) \
            for i in range(n_heads)
        ])

        self.norm = LayerNorm()
        self.o = torch.nn.Linear(n_heads*dk, d)


    def forward(self, X):

        # Concatenate output of heads
        A = torch.cat([
           self.heads[i](X) for i in range(len(self.heads)) 
        ], dim=2)

        # Project do embedding dimension space
        O = self.o(A)

        return O



class SelfAttentionLayer(torch.nn.Module):


    def __init__(self, d_in, d_out):
        super(SelfAttentionLayer, self).__init__()
        self.q = torch.nn.Linear(d_in, d_out)
        self.k = torch.nn.Linear(d_in, d_out)
        self.v = torch.nn.Linear(d_in, d_out)
        self.sqrt_dk = torch.sqrt(torch.Tensor([d_out]))


    def forward(self, X):
        Q, K, V = self.q(X), self.k(X), self.v(X)
        X = torch.einsum('ijk,ilk->ijl', Q, K)
        X = torch.tril(X) / self.sqrt_dk
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
    attention_dim = 32 
    vocab_size = 1000
    window_size = 128
    hidden_dim = 256
    n_heads = 4
    n_blocks = 3

    decoder = TransformerDecoder(vocab_size, embedding_dim, attention_dim,
                                 n_heads, hidden_dim, n_blocks)

    for X, Y in tqdm(tloader):

        # Transform ids into one-hot vectors            
        X = torch.nn.functional.one_hot(X, vocab_size)
        X = X.type(torch.FloatTensor)

        # Decode
        X = decoder(X)



if __name__ == '__main__':
    main()

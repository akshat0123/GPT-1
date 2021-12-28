"""
File containing class for transformer-based decoder model
"""
from torch.nn import ModuleList, Identity, Dropout, Linear, Module, ReLU
from torch.nn.functional import softmax
import torch.nn.functional as F
import torch


class TransformerDecoder(Module):


    def __init__(self, v: int, w: int, d: int, dk: int, n_heads: int, 
                 hidden: int, n_blocks: int, dropout: float, 
                 device: str='cpu') -> 'Decoder':
        """ Decoder implementation (as described in GPT paper)

        Args:
            v: vocabulary size 
            w: window size  of sequence
            d: embedding dimension
            dk: attention head dimension 
            n_heads: number of attention heads
            hidden: number of hidden units in feed forward layers
            n_blocks: number of transformer blocks
            dropout: amount of dropout to add to block sublayers
            device: device to keep instance on

        Returns:
            (TransformerDecoder): transformer based decoder module
        """

        super(TransformerDecoder, self).__init__()

        self.pids = torch.Tensor([i for i in range(w)]).to(device=device)
        self.embedding = Embedding(v, d, device)
        self.position = Embedding(w, d, device)
        self.blocks = ModuleList([
            TransformerBlock(d, dk, n_heads, hidden, dropout, device) \
            for i in range(n_blocks)
        ])
        self.linear = Linear(d, v).to(device=device)
        self.device = device


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        # Transform one-hot vectors into embeddings
        X = self.embedding(X)

        # Add positional embedding
        X += self.position(self.pids)

        # Run through transformer blocks
        for block in self.blocks:
            X = block(X)

        # Project to vocabulary space
        X = softmax(self.linear(X), dim=2)[:, -1, :]

        return X


class Embedding(Module):


    def __init__(self, size: int, dim: int, device: str='cpu') -> 'Embedding':
        """ Implementation of word embeddings

        Args:
            size: size of vocabulary
            dim: dimension size of embeddings 
            device: device to keep instance on

        Returns:
            (Embedding): word embeddings instance
        """

        super(Embedding, self).__init__()
        self.embeddings = Linear(size, dim).to(device=device)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Returns word embeddings for batch of indices

        Args:
            X: batch of word indices

        Returns:
            (torch.Tensor): word embeddings corresponding to provided indices

        """

        return self.embeddings(X)


class TransformerBlock(Module):


    def __init__(self, d: int, dk: int, n_heads: int, hidden: int, 
                 dropout: float, device: str='cpu') -> 'TransformerBlock':
        """ Single transformer block implementation

        Args:
            d: input dimensions
            dk: attention head dimensions
            n_heads: number of heads
            hidden: hidden units in feed forward layers
            dropout: amount of dropout to add to sublayers
            device: device to keep instance on

        Returns:
            (TransformerBlock): transformer block
        """

        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadedAttentionLayer(d, dk, n_heads, device)
        self.norm = LayerNorm()
        self.ff1 = PositionWiseFFN(d, hidden, ReLU(), device)
        self.ff2 = PositionWiseFFN(hidden, d, Identity(), device)
        self.d1 = Dropout(p=dropout)
        self.d2 = Dropout(p=dropout)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        # Run through attention
        A = self.attention(X)

        # Normalization and residual connection
        X = self.norm(X + self.d1(A))

        # Position-wise Feed forward networks
        F = self.ff2(self.ff1(X))

        # Normalization and residual connection
        X = self.norm(X + self.d2(F))

        return X


class MultiHeadedAttentionLayer(torch.nn.Module):

    
    def __init__(self, d: int, dk: int, n_heads: int, 
                 device: str='cpu') -> 'MultiHeadAttentionLayer':
        """ Multi-headed self-attention layer implementation

        Args:
            d: input dimensions
            dk: attention head dimensions
            n_heads: number of heads
            device: device to keep instance on

        Returns:
            (MultiHeadedAttentionLayer): multi-headed self-attention layer
        """

        super(MultiHeadedAttentionLayer, self).__init__()

        self.heads = torch.nn.ModuleList([
            SelfAttentionLayer(d, dk, device) \
            for i in range(n_heads)
        ])

        self.o = torch.nn.Linear(n_heads*dk, d).to(device=device)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        # Concatenate output of heads
        A = torch.cat([
           self.heads[i](X) for i in range(len(self.heads)) 
        ], dim=2)

        # Project to embedding dimension space
        O = self.o(A)

        return O


class SelfAttentionLayer(torch.nn.Module):


    def __init__(self, d_in: int, d_out: int, device: str='cpu') -> 'SelfAttentionLayer':
        """ Single-head self-attention layer implementation

        Args:
            d_in: input dimensions
            d_out: hidden units
            device: device to keep instance on

        Returns:
            (SelfAttentionLayer): self-attention layer
        """

        super(SelfAttentionLayer, self).__init__()

        self.q = torch.nn.Linear(d_in, d_out).to(device=device)
        self.k = torch.nn.Linear(d_in, d_out).to(device=device)
        self.v = torch.nn.Linear(d_in, d_out).to(device=device)
        self.sqrt_dk = torch.sqrt(torch.Tensor([d_out])).to(device=device)
        self.device = device


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        Q, K, V = self.q(X), self.k(X), self.v(X)
        QK = torch.einsum('ijk,ilk->ijl', Q, K) / self.sqrt_dk
        sQK = softmax(torch.tril(X), dim=2)
        A = torch.einsum('ijk,ijl->ijl', sQK, V)
        return A 


class PositionWiseFFN(torch.nn.Module):


    def __init__(self, d_in: int, d_out: int, activation: Module, 
                 device: str='cpu') -> 'FeedForwardLayer':
        """ Position-wise feed forward neural network implementation

        Args:
            d_in: input dimensions
            d_out: hidden units
            activation: activation function
            device: device to keep instance on

        Returns:
            (PositionWiseFFN): Position-wise feed forward netork module
        """

        super(PositionWiseFFN, self).__init__()
        self.w = torch.nn.Linear(d_in, d_out).to(device=device)
        self.activation = activation


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        X = torch.cat([
            self.w(X[:, i, :])[:, None, :] for i in range(X.shape[1])
        ], dim=1)

        return self.activation(X)


class LayerNorm(torch.nn.Module):


    def __init__(self):
        """  Layer normalization implementation
        """

        super(LayerNorm, self).__init__()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Normalize layer of batch of inputs

        Args:
            X: input batch

        Returns:
            (torch.Tensor): layer normalized input             
        """

        mu = torch.mean(X, dim=2)[:, :, None]
        sigma = torch.square(X - mu)
        sigma = torch.sum(sigma, dim=2) / X.shape[2]
        sigma = torch.sqrt(sigma)[:, :, None]
        return (X - mu) / sigma

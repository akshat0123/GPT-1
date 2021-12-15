"""
File containing class for transformer-based decoder model
"""
import torch.nn.functional as F
import torch


class Decoder(torch.nn.Module):


    def __init__(self, n_layers: int, n_heads: int, d_in: int, 
                 d_out: int, device: str='cpu') -> 'Decoder':
        """ Decoder implementation (as described in GPT paper)

        Args:
            n_layers: number of layers
            n_heads: number of attention heads per transformer block
            d_in: input dimensions
            d_out: hidden units
            device: device to keep instance on

        Returns:
            (Decoder): transformer based decoder module
        """

        super(Decoder, self).__init__()

        self.embeddings = Embedding(d_out, d_in, device)

        self.blocks = torch.nn.ModuleList([
            TransformerBlock(d_in, n_heads, device) for i in range(n_layers)
        ])

        self.linear = torch.nn.Linear(d_in, d_out).to(device=device)
        self.device = device


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        X = self.embeddings(X)
        
        for block in self.blocks:
            X = block(X)

        X = F.softmax(self.linear(X), dim=2)[:, -1, :]

        return X


class Embedding(torch.nn.Module):


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
        self.embeddings = torch.nn.Linear(size, dim).to(device=device)
        self.device = device
        self.dim = dim


    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """ Returns word embeddings for batch of indices

        Args:
            batch: batch of word indices

        Returns:
            (torch.Tensor): word embeddings corresponding to provided indices

        """

        return self.embeddings(batch)


class TransformerBlock(torch.nn.Module):


    def __init__(self, d: int, h: int, device: str='cpu') -> 'TransformerBlock':
        """ Single transformer block implementation

        Args:
            d: input dimensions
            h: number of heads
            device: device to keep instance on

        Returns:
            (TransformerBlock): transformer block
        """

        super(TransformerBlock, self).__init__()

        self.attn = MultiHeadAttentionLayer(d, h, device)
        self.ffn = FeedForwardLayer(d, d, device)
        self.norm = LayerNorm()
        self.device = device


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        X = self.norm(X + self.attn(X))
        X = self.norm(X + self.ffn(X))
        return X


class MultiHeadAttentionLayer(torch.nn.Module):

    
    def __init__(self, d: int, h: int, device: str='cpu') -> 'MultiHeadAttentionLayer':
        """ Multi-headed self-attention layer implementation

        Args:
            d: input dimensions
            h: number of heads
            device: device to keep instance on

        Returns:
            (MultiHeadAttentionLayer): multi-headed self-attention layer
        """

        super(MultiHeadAttentionLayer, self).__init__()

        self.heads = torch.nn.ModuleList([
            SelfAttentionLayer(d, d//h, device) for i in range(h)
        ])

        self.W_o = torch.nn.Linear(d, d).to(device=device)
        self.device = device
        self.dh = d//h
        self.h = h
        self.d = d


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        Z = []
        for i in range(self.h):
            Z.append(self.heads[i](X))

        Z = torch.cat(Z, dim=2)
        return Z


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

        self.W_q = torch.nn.Linear(d_in, d_out).to(device=device)
        self.W_k = torch.nn.Linear(d_in, d_out).to(device=device)
        self.W_v = torch.nn.Linear(d_in, d_out).to(device=device)
        self.sd = torch.sqrt(torch.Tensor([d_out])).to(device=device)
        self.device = device


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        QK = torch.einsum('ijk,ilk->ijl', Q, K) / self.sd
        sQK = torch.tril(F.softmax(QK, dim=2))
        out = torch.einsum('ijk,ikl->ijl', sQK, V)
        return out


class FeedForwardLayer(torch.nn.Module):


    def __init__(self, d_in: int, d_out: int, device: str='cpu') -> 'FeedForwardLayer':
        """ Feed forward neural network layer implementation

        Args:
            d_in: input dimensions
            d_out: hidden units
            device: device to keep instance on

        Returns:
            (FeedForwardLayer): feed forward layer
        """

        super(FeedForwardLayer, self).__init__()

        self.W = torch.nn.Linear(d_in, d_out).to(device=device)
        self.act = torch.nn.ReLU()
        self.device = device


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            X: input tensor

        Returns:
            (torch.Tensor): output of layer
        """

        X = self.W(X)
        X = self.act(X)
        return X


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
        sigma = torch.mean((X-mu)**2, dim=2)[:, :, None]
        return (X - mu) / sigma

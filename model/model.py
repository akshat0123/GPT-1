from torch.nn import ModuleList, Identity, Dropout, Softmax, Linear, Module, GELU 
from torch import Tensor, einsum, square, mean, ones, sqrt, tril, cat
from torch import sum as sum_


class TransformerDecoder(Module):


    def __init__(self, vocab_size: int, embedding_size: int, window_size: int,
                 d_k: int, d_v: int, n_heads: int, hidden: int, n_blocks: int,
                 dropout: float, device: str):
        """ Initialize tranformer-based decoder

        Args:
            vocab_size: input vocabulary size
            embedding_size: embedding size
            window_size: sequence window size
            d_k: self-attention query, key dimension size
            d_v: self-attention value size 
            n_heads: number of self-attention heads 
            hidden: number of hidden units in feed forward layers
            n_blocks: number of transformer blocks
            dropout: dropout amount
            device: device to put model on
        """
        super().__init__()
        self.pids = Tensor([i for i in range(window_size + 1)]).to(device)
        self.embedding = Embedding(vocab_size, embedding_size, device)
        self.position = Embedding(window_size + 1, embedding_size, device)

        self.blocks = ModuleList([
            TransformerBlock(embedding_size, d_k, d_v, n_heads, hidden, dropout, device) \
            for i in range(n_blocks)
        ])
        
        self.w = Linear(embedding_size, vocab_size).to(device)
        self.device = device


    def forward(self, X: Tensor) -> Tensor:
        X = self.embedding(X) 
        X += self.position(self.pids)

        for block in self.blocks:
            X = block(X)

        return self.w(X)[:, -1, :]


class TransformerBlock(Module):


    def __init__(self, d_in, d_k, d_v, n_heads, hidden, dropout, device):
        """ Initialize tranformer block

        Args:
            d_in: input size
            d_k: self-attention query, key dimension size
            d_v: self-attention value size 
            n_heads: number of self-attention heads 
            hidden: number of hidden units in feed forward layers
            dropout: dropout amount
            device: device to put model on
        """
        super().__init__()
        self.attention = MultiHeadAttentionLayer(d_in, d_k, d_v, n_heads, device)
        self.layer_norm = LayerNorm()
        self.ffl1 = PositionWiseFFL(d_in, hidden, GELU(), device)
        self.ffl2 = PositionWiseFFL(hidden, d_in, Identity(), device)
        self.d1 = Dropout(p=dropout)
        self.d2 = Dropout(p=dropout)


    def forward(self, X):
        Z = self.layer_norm(X + self.d1(self.attention(X)))
        Y = self.layer_norm(Z + self.d2(self.ffl2(self.ffl1(Z))))
        return Y


class Embedding(Module):


    def __init__(self, vocab_size, embedding_size, device):
        """ Initialized embedding encoder 

        Args:
            vocab_size: input vocabulary size
            embedding_size: embedding size
            device: device to put model on
        """
        super().__init__()
        self.embedding = Linear(vocab_size, embedding_size).to(device)


    def forward(self, X):
        return self.embedding(X)


class MultiHeadAttentionLayer(Module):


    def __init__(self, d_in, d_k, d_v, n_heads, device):
        """ Initialize multi-headed attention layer

        Args:
            d_in: input size
            d_k: self-attention query, key dimension size
            d_v: self-attention value size 
            n_heads: number of self-attention heads 
            device: device to put model on
        """
        super().__init__()
        self.heads = ModuleList([
            SelfAttentionLayer(d_in, d_k, d_v, device) for i in range(n_heads)
        ])

        self.w = Linear(n_heads * d_v, d_in).to(device)


    def forward(self, X):
        X = cat([head(X) for head in self.heads], dim=2)
        return self.w(X)


class SelfAttentionLayer(Module):


    def __init__(self, d_in, d_k, d_v, device):
        """ Initialized self attention layer

        Args:
            d_in: input size
            d_k: self-attention query, key dimension size
            d_v: self-attention value size 
            device: device to put model on
        """
        super().__init__()
        self.wq = Linear(d_in, d_k).to(device)
        self.wk = Linear(d_in, d_k).to(device)
        self.wv = Linear(d_in, d_v).to(device)
        self.sqrt_dk = sqrt(Tensor([d_k])).to(device)
        self.softmax = Softmax(dim=2)
        self.device = device


    def forward(self, X):
        Q, K, V = self.wq(X), self.wk(X), self.wv(X)
        QK = einsum('ijk,ilk->ijl', Q, K) / self.sqrt_dk
        mask = tril(ones(QK.shape)).to(device=self.device)
        sQK = self.softmax(QK.masked_fill(mask==0, float('-inf')))
        sA = einsum('ijk,ikl->ijl', sQK, V)
        return sA


class LayerNorm(Module):


    def __init__(self):
        """ Layer normalization module
        """
        super().__init__()

    
    def forward(self, X):
        mu = mean(X, dim=2)[:, :, None]
        sigma = square(X - mu)
        sigma = sum_(sigma, dim=2) / X.shape[2]
        sigma = sqrt(sigma)[:, :, None]
        return (X - mu) / sigma


class PositionWiseFFL(Module):


    def __init__(self, d_in, d_out, activation, device):
        """ Initialize position-wise feed forward neural network layer 

        Args:
            d_in: input size
            d_out: output size
            activation: activation function
            device: device to put model on
        """
        super().__init__()
        self.w = Linear(d_in, d_out).to(device)
        self.activation = activation


    def forward(self, X):
        X = cat([
            self.w(X[:, i, :])[:, None, :] for i in range(X.shape[1])
        ], dim=1)

        return self.activation(X)

from torch.nn import Parameter, Softmax, Linear, Module, GELU
from torch import sum as sum_
from torch import (
    LongTensor, 
    einsum, 
    Tensor, 
    square, 
    mean, 
    ones, 
    rand, 
    sqrt, 
    tril
)


class TransformerDecoder(Module):


    def __init__(self, vocab_size, seq_size, d_emb, d_k, d_v, d_h):
        super().__init__()
        self.pids = LongTensor([i for i in range(seq_size)])
        self.block = TransformerBlock(d_emb, d_k, d_v, d_h)
        self.output = Linear(d_emb, vocab_size)
        self.embedding = Embedding(vocab_size, d_emb)
        self.position = Embedding(seq_size, d_emb)


    def forward(self, x):
        emb = self.embedding(x)
        pos = self.position(self.pids)
        y = self.block(emb + pos)
        return self.output(y)[:, -1, :]


class TransformerBlock(Module):


    def __init__(self, d_in, d_k, d_v, d_h):
        super().__init__()
        self.attention = SelfAttentionLayer(d_in, d_k, d_v)
        self.ffl = FeedForwardLayer(d_in, d_h)
        self.norm = LayerNorm()


    def forward(self, x):
        z = self.norm(x + self.attention(x))
        y = self.norm(z + self.ffl(z))
        return y


class SelfAttentionLayer(Module):


    def __init__(self, d_in, d_k, d_v):
        super().__init__()
        self.wq = Linear(d_in, d_k)
        self.wk = Linear(d_in, d_k)
        self.wv = Linear(d_in, d_v)
        self.dk = sqrt(Tensor([d_k]))
        self.softmax = Softmax(dim=2)


    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        qk = einsum('bid,bjd->bij', q, k) / self.dk

        mask = tril(ones(qk.shape))
        qk = qk.masked_fill(mask==0, float('-inf'))

        sqk = self.softmax(qk)
        sa = einsum('bik,bkj->bij', sqk, v)

        return sa


class FeedForwardLayer(Module):


    def __init__(self, d_in, d_h):
        super().__init__()
        self.linear1 = Linear(d_in, d_h)
        self.linear2 = Linear(d_h, d_in)
        self.activation = GELU()

        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.activation(x)


class Embedding(Module):


    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embeddings = Parameter(
            rand((vocab_size, dim)), 
            requires_grad=True
        )


    def forward(self, x):
        return self.embeddings[x]


class LayerNorm(Module):


    def __init__(self):
        super().__init__()


    def forward(self, x):
        mu = mean(x, dim=2)[:, :, None]
        sigma = square(x - mu)
        sigma = sum_(sigma, dim=2) / x.shape[2]
        sigma = sqrt(sigma)[:, :, None]
        return (x - mu) / sigma



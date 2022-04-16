from torch import LongTensor, nan_to_num, einsum, Tensor, square, mean, ones, rand, sqrt, tril, cat
from torch.nn import Parameter, ModuleList, Dropout, Softmax, Linear, Module, GELU
from torch import sum as sum_


class TransformerDecoder(Module):


    def __init__(self, vocab_size, seq_size, n_layers, d_emb, d_k, d_v, d_h, n_heads, dropout, device):
        super().__init__()
        self.pids = LongTensor([i for i in range(seq_size)]).to(device=device)
        self.embedding = Embedding(vocab_size, d_emb, device)
        self.position = Embedding(seq_size, d_emb, device)
        self.dropout = Dropout(dropout)

        self.blocks = [TransformerBlock(d_emb, d_k, d_v, d_h, n_heads, dropout, device) for i in range(n_layers)]
        self.output = Linear(d_emb, vocab_size).to(device=device)


    def forward(self, x, padding):
        emb = self.embedding(x)
        pos = self.position(self.pids)

        x = self.dropout(emb + pos)
        for block in self.blocks:
            x = block(x, padding)

        return self.output(x)[:, -1, :]


class TransformerBlock(Module):


    def __init__(self, d_in, d_k, d_v, d_h, n_heads, dropout, device):
        super().__init__()
        self.attention = MultiHeadedAttentionLayer(d_in, d_k, d_v, n_heads, device)
        self.ffl = FeedForwardLayer(d_in, d_h, device)
        self.dropout = Dropout(dropout)
        self.norm = LayerNorm()


    def forward(self, x, padding):
        z = self.norm(x + self.dropout(self.attention(x, padding)))
        y = self.norm(z + self.dropout(self.ffl(z)))
        return y


class MultiHeadedAttentionLayer(Module):


    def __init__(self, d_in, d_k, d_v, n_heads, device):
        super().__init__()
        self.attentions = ModuleList([
            SelfAttentionLayer(d_in, d_k, d_v, device) for i in range(n_heads)
        ])
        self.wo = Linear(n_heads * d_v, d_in).to(device=device)


    def forward(self, x, padding):
        x = cat([att(x, padding) for att in self.attentions], dim=2)
        return self.wo(x)


class SelfAttentionLayer(Module):


    def __init__(self, d_in, d_k, d_v, device):
        super().__init__()
        self.wq = Linear(d_in, d_k).to(device=device)
        self.wk = Linear(d_in, d_k).to(device=device)
        self.wv = Linear(d_in, d_v).to(device=device)
        self.dk = sqrt(Tensor([d_k])).to(device=device)
        self.softmax = Softmax(dim=2).to(device=device)
        self.device = device


    def forward(self, x, padding):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        qk = einsum('bid,bjd->bij', q, k) / self.dk

        pad_mask = ones(qk.shape).to(device=self.device)
        pad_mask = einsum('bij,bj->bij', pad_mask, padding)
        pad_mask = einsum('bij,bi->bij', pad_mask, padding)

        mask = pad_mask * tril(ones(qk.shape)).to(device=self.device)
        qk = qk.masked_fill(mask==0, float('-inf'))

        sqk = self.softmax(qk)
        sqk = nan_to_num(sqk)
        sa = einsum('bik,bkj->bij', sqk, v)

        return sa


class FeedForwardLayer(Module):


    def __init__(self, d_in, d_h, device):
        super().__init__()
        self.linear1 = Linear(d_in, d_h).to(device=device)
        self.linear2 = Linear(d_h, d_in).to(device=device)
        self.activation = GELU().to(device=device)

        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.activation(x)


class Embedding(Module):


    def __init__(self, vocab_size, dim, device):
        super().__init__()
        self.embeddings = Parameter(
            rand((vocab_size, dim)), 
            requires_grad=True
        ).to(device=device)


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



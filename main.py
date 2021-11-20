import torch.nn.functional as F
import torch


class Transformer(torch.nn.Module):

    def __init__(self, n_layers, d):
        super(Transformer, self).__init__()

        self.blocks = [
            TransformerBlock(d) for i in range(n_layers)
        ]


    def forward(self, X):

        for block in self.blocks:
            X = block(x)

        return X


class TransformerBlock(torch.nn.Module):


    def __init__(self, d):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttentionLayer(d)
        self.ffn = FeedForwardLayer(d)
        self.norm = LayerNorm()


    def forward(self, X):
        X = self.norm(X + self.attention(X))
        X = self.norm(X + self.ffn(X))
        return X


class SelfAttentionLayer(torch.nn.Module):


    def __init__(self, d):
        super(SelfAttentionLayer, self).__init__()
        self.W_q = torch.nn.Linear(d, d)
        self.W_k = torch.nn.Linear(d, d)
        self.W_v = torch.nn.Linear(d, d)
        self.d = torch.Tensor([d])


    def forward(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        QK = torch.einsum('ij,kj->ik', Q, K) / torch.sqrt(self.d)
        sQK = torch.tril(F.softmax(QK, dim=1))
        out = torch.einsum('ij,jk->ik', sQK, V)
        return out


class FeedForwardLayer(torch.nn.Module):

    def __init__(self, d):
        super(FeedForwardLayer, self).__init__()
        self.W = torch.nn.Linear(d, d)
        self.act = torch.nn.ReLU()

    def forward(self, X):
        X = self.W(X)
        X = self.act(X)
        return X


class LayerNorm(torch.nn.Module):

    def __init__(self):
        super(LayerNorm, self).__init__()


    def forward(self, X):
        mu = torch.mean(X, dim=1)[:, None]
        sigma = torch.mean((X-mu)**2, dim=1)[:, None]
        return (X - mu) / sigma


def main():

    dim = 5 
    x = torch.rand((3, dim))

    transformer = TransformerBlock(dim)
    y = transformer.forward(x)



if __name__ == '__main__':
    main()

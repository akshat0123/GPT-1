import torch.nn.functional as F
import torch


class Decoder(torch.nn.Module):

    def __init__(self, n_layers, d, h):
        super(Decoder, self).__init__()

        self.blocks = [
            TransformerBlock(d, h) for i in range(n_layers)
        ]


    def forward(self, X):
        
        for block in self.blocks:
            X = self.block(X)

        return X


class TransformerBlock(torch.nn.Module):


    def __init__(self, d, h):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttentionLayer(d, h)
        self.ffn = FeedForwardLayer(d, d)
        self.norm = LayerNorm()


    def forward(self, X):
        X = self.norm(X + self.attn(X))
        X = self.norm(X + self.ffn(X))
        return X


class MultiHeadAttentionLayer(torch.nn.Module):

    
    def __init__(self, d, h):
        super(MultiHeadAttentionLayer, self).__init__()

        self.heads = [
            SelfAttentionLayer(d, d//h) for i in range(h)
        ]

        self.W_o = torch.nn.Linear(d, d)
        self.dh = d//h
        self.h = h
        self.d = d

    def forward(self, X):
        Z = []
        for i in range(self.h):
            Z.append(self.heads[i](X))

        Z = torch.cat(Z, dim=1)
        return Z


class SelfAttentionLayer(torch.nn.Module):


    def __init__(self, d_in, d_out):
        super(SelfAttentionLayer, self).__init__()
        self.W_q = torch.nn.Linear(d_in, d_out)
        self.W_k = torch.nn.Linear(d_in, d_out)
        self.W_v = torch.nn.Linear(d_in, d_out)
        self.d = torch.Tensor([d_out])


    def forward(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        QK = torch.einsum('ij,kj->ik', Q, K) / torch.sqrt(self.d)
        sQK = torch.tril(F.softmax(QK, dim=1))
        out = torch.einsum('ij,jk->ik', sQK, V)
        return out


class FeedForwardLayer(torch.nn.Module):

    def __init__(self, d_in, d_out):
        super(FeedForwardLayer, self).__init__()
        self.W = torch.nn.Linear(d_in, d_out)
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

    dim = 512 
    x = torch.rand((5, dim))

    layer = MultiHeadAttentionLayer(512, 8)
    y = layer.forward(x)


if __name__ == '__main__':
    main()

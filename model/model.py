from torch import LongTensor, nan_to_num, einsum, Tensor, square, mean, ones, rand, sqrt, tril, triu, cat
from torch.nn import Parameter, ModuleList, Dropout, Softmax, Linear, Module, GELU
from torch.nn.init import normal_, zeros_
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

        normal_(self.output.weight, mean=0, std=0.02)
        zeros_(self.output.bias)


    def forward(self, x, ignore):
        emb = self.embedding(x)
        pos = self.position(self.pids)

        pos = self.position(self.pids).unsqueeze(0)
        pos = pos.expand(x.shape[0], -1, -1) 
        pos = einsum('bsd,bs->bsd', pos, (ignore==0).float())

        x = self.dropout(emb + pos)
        for block in self.blocks:
            x = block(x, ignore)

        return self.output(x)


    def get_parameters(self):

        params = [ 
            {'params': [], 'weight_decay': 1e-5 },
            {'params': [], 'weight_decay': 0.00 }
        ]

        for name, parameter in self.named_parameters():

            if 'bias' in name:
                params[1]['params'].append(parameter)

            else:
                params[0]['params'].append(parameter)

        return params


class TransformerBlock(Module):


    def __init__(self, d_in, d_k, d_v, d_h, n_heads, dropout, device):
        super().__init__()
        self.attention = MultiHeadedAttentionLayer(d_in, d_k, d_v, n_heads, device)
        self.ffl = FeedForwardLayer(d_in, d_h, device)
        self.dropout = Dropout(dropout).to(device=device)
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()


    def forward(self, x, ignore):
        z = self.norm1(x + self.dropout(self.attention(x, ignore)))
        y = self.norm2(z + self.dropout(self.ffl(z)))
        return y


class MultiHeadedAttentionLayer(Module):


    def __init__(self, d_in, d_k, d_v, n_heads, device):
        super().__init__()
        self.attentions = ModuleList([
            SelfAttentionLayer(d_in, d_k, d_v, device) for i in range(n_heads)
        ])
        self.wo = Linear(n_heads * d_v, d_in).to(device=device)

        normal_(self.wo.weight, mean=0.0, std=0.02)
        zeros_(self.wo.bias)


    def forward(self, x, ignore):
        x = cat([att(x, ignore) for att in self.attentions], dim=2)
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

        normal_(self.wq.weight, mean=0.0, std=0.02)
        normal_(self.wk.weight, mean=0.0, std=0.02)
        normal_(self.wv.weight, mean=0.0, std=0.02)
        zeros_(self.wq.bias)
        zeros_(self.wk.bias)
        zeros_(self.wv.bias)


    def forward(self, x, ignore):

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        qk = einsum('bid,bjd->bij', q, k) / self.dk

        ignore_mask = self.get_ignore_mask(qk, ignore)
        forward_mask = self.get_forward_mask(qk)
        mask = ignore_mask * forward_mask
        qk = qk.masked_fill(mask==0, float('-inf'))

        sqk = self.softmax(qk)
        sa = einsum('bik,bkj->bij', sqk, v)

        return sa


    def get_ignore_mask(self, qk, ignore):
        ignore_mask = ones(qk.shape).to(device=self.device)
        ignore_mask = einsum('bij,bj->bij', ignore_mask, (ignore==0).float())
        ignore_mask += triu(tril(ones(qk.shape))).to(device=self.device)
        ignore_mask[ignore_mask==2] = 1
        return ignore_mask


    def get_forward_mask(self, qk):
        return tril(ones(qk.shape)).to(device=self.device)


class FeedForwardLayer(Module):


    def __init__(self, d_in, d_h, device):
        super().__init__()
        self.linear1 = Linear(d_in, d_h).to(device=device)
        self.linear2 = Linear(d_h, d_in).to(device=device)
        self.activation = GELU().to(device=device)

        normal_(self.linear1.weight, mean=0.0, std=0.02)
        normal_(self.linear2.weight, mean=0.0, std=0.02)
        zeros_(self.linear1.bias)
        zeros_(self.linear2.bias)

        
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

        normal_(self.embeddings, mean=0.0, std=0.02)


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



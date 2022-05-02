import torch


class GPT(torch.nn.Module):


    def __init__(self, vocab, seq, n_layers, n_heads, dim, hidden, dropout, device):
        super().__init__()
        self.bpe_embed = torch.nn.Embedding(vocab, dim).to(device)
        self.pos_embed = torch.nn.Embedding(seq, dim).to(device)
        self.pos = torch.LongTensor([i for i in range(128)]).to(device)
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(n_heads, dim, hidden, dropout, device) \
            for i in range(n_layers)
        ])
        self.output = torch.nn.Linear(dim, vocab).to(device)
        self.drop = torch.nn.Dropout(dropout).to(device)
        self.init_weights()


    def init_weights(self):
        torch.nn.init.normal_(self.bpe_embed.weight, mean=0.0, std=0.02) 
        torch.nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02) 
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.02) 
        torch.nn.init.zeros_(self.output.bias) 


    def forward(self, x, ignore):
        be = self.bpe_embed(x)
        pe = self.pos_embed(self.pos)

        out = self.drop(be + pe)
        for block in self.blocks:
            out = block(out, ignore)

        return self.output(out)


class TransformerBlock(torch.nn.Module):


    def __init__(self, n_heads, dim, hidden, dropout, device):
        super().__init__()
        self.att = MultiHeadAttentionLayer(n_heads, dim, device)
        self.ffl = FeedForwardLayer(dim, hidden, device)
        self.norm1 = torch.nn.LayerNorm(dim).to(device)
        self.norm2 = torch.nn.LayerNorm(dim).to(device)
        self.drop1 = torch.nn.Dropout(dropout).to(device)
        self.drop2 = torch.nn.Dropout(dropout).to(device)
        self.init_weights()


    def init_weights(self):
        torch.nn.init.ones_(self.norm1.weight)
        torch.nn.init.ones_(self.norm2.weight)
        torch.nn.init.zeros_(self.norm1.bias)
        torch.nn.init.zeros_(self.norm2.bias)


    def forward(self, x, ignore):
        out = self.norm1(x + self.drop1(self.att(x, ignore)))
        return self.norm2(out + self.drop2(self.ffl(out)))


class MultiHeadAttentionLayer(torch.nn.Module):


    def __init__(self, n_heads, dim, device):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            SelfAttentionLayer(dim, dim // n_heads, device) \
            for i in range(n_heads)
        ])
        self.wo = torch.nn.Linear(dim, dim).to(device)
        self.init_weights()


    def init_weights(self):
        torch.nn.init.normal_(self.wo.weight, mean=0.0, std=0.02) 
        torch.nn.init.zeros_(self.wo.bias) 


    def forward(self, x, ignore):
        out = torch.cat([head(x, ignore) for head in self.heads], dim=2)
        return self.wo(out)


class SelfAttentionLayer(torch.nn.Module):


    def __init__(self, d_in, d_out, device):
        super().__init__()
        self.wq = torch.nn.Linear(d_in, d_out).to(device)
        self.wk = torch.nn.Linear(d_in, d_out).to(device)
        self.wv = torch.nn.Linear(d_in, d_out).to(device)
        self.scale = torch.sqrt(torch.Tensor([d_out])).to(device)
        self.softmax = torch.nn.Softmax(dim=2).to(device)
        self.device = device
        self.init_weights()


    def init_weights(self):
        torch.nn.init.normal_(self.wq.weight, mean=0.0, std=0.02) 
        torch.nn.init.normal_(self.wk.weight, mean=0.0, std=0.02) 
        torch.nn.init.normal_(self.wv.weight, mean=0.0, std=0.02) 
        torch.nn.init.zeros_(self.wq.bias) 
        torch.nn.init.zeros_(self.wk.bias) 
        torch.nn.init.zeros_(self.wv.bias) 


    def forward(self, x, ignore):

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        att = torch.einsum('bqd,bkd->bqk', q, k)
        att /= self.scale

        causal_mask = torch.tril(torch.ones(att.shape)).to(self.device)
        ignore_mask = self.get_ignore_mask(att, ignore)
        mask = causal_mask * ignore_mask

        att = att.masked_fill(mask==0, float('-inf'))
        att = self.softmax(att)
        att = torch.einsum('bqk,bkd->bqd', att, v)

        return att


    def get_ignore_mask(self, att, ignore):

        ignore_mask = torch.ones(att.shape).to(self.device)
        ignore_mask = torch.einsum(
            'bqk,bk->bqk', 
            ignore_mask, 
            (ignore==0).float()
        )
        ignore_mask += torch.triu(torch.tril(torch.ones(att.shape))).to(self.device)
        ignore_mask[(ignore_mask==2)] = 1

        return ignore_mask


class FeedForwardLayer(torch.nn.Module):


    def __init__(self, d_in, d_h, device):
        super().__init__()
        self.l1 = torch.nn.Linear(d_in, d_h).to(device)
        self.l2 = torch.nn.Linear(d_h, d_in).to(device)
        self.gelu = torch.nn.GELU().to(device)


    def init_weights(self):
        torch.nn.init.normal_(self.l1.weight, mean=0.0, std=0.02) 
        torch.nn.init.normal_(self.l2.weight, mean=0.0, std=0.02) 
        torch.nn.init.zeros_(self.l1.bias) 
        torch.nn.init.zeros_(self.l2.bias) 


    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        return self.gelu(out)

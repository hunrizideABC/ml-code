import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ---- Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---- Multi-Head Attention (Q/K/V 分开) ----
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B, Q_len, _ = query.size()
        K_len = key.size(1)

        # 分开线性映射
        q = self.w_q(query).reshape(B, Q_len, self.num_heads, self.d_k).transpose(1,2)
        k = self.w_k(key).reshape(B, K_len, self.num_heads, self.d_k).transpose(1,2)
        v = self.w_v(value).reshape(B, K_len, self.num_heads, self.d_k).transpose(1,2)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).reshape(B, Q_len, self.d_model)
        return self.out_proj(out)

# ---- Feed Forward ----
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ---- Encoder Layer ----
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.self_attn(x, x, x, mask)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

# ---- Decoder Layer ----
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        x = x + self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x)
        x = x + self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x)
        x = x + self.ffn(x)
        x = self.norm3(x)
        return x

# ---- Encoder ----
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=500):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, src, mask=None):
        x = self.embed(src)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# ---- Decoder ----
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=500):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        x = self.embed(tgt)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return self.linear(x)

# ---- Transformer Seq2Seq ----
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_heads, num_layers)
        self.decoder = Decoder(tgt_vocab, d_model, num_heads, num_layers)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return out

# ---- 测试 ----
if __name__ == "__main__":
    src_vocab, tgt_vocab = 1000, 1000
    model = Transformer(src_vocab, tgt_vocab)
    src = torch.randint(0, src_vocab, (2,10))  # [B, src_len]
    tgt = torch.randint(0, tgt_vocab, (2,12))  # [B, tgt_len]
    out = model(src, tgt)
    print(out.shape)  # [B, tgt_len, tgt_vocab]

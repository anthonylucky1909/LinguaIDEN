from models.attention import SelfAttention
from models.feedforward import FeedForward
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
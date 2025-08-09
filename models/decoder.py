from models.attention import SelfAttention, CrossAttention
from models.feedforward import FeedForward
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention_masked = SelfAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attention = CrossAttention(embed_dim, heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_encoder, tgt_mask=None, src_mask=None):
        attn_out = self.attention_masked(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        cross_attn_out = self.cross_attention(query=x, key=x_encoder, value=x_encoder, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x
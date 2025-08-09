import torch.nn as nn
import math
from .positional import PositionalEncoding
from .encoder import EncoderLayer
from .decoder import DecoderLayer

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_encoder_layers,
                 num_decoder_layers, input_vocab_size, target_vocab_size,
                 max_seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.target_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_linear = nn.Linear(embed_dim, target_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_x = self.input_embedding(src) * math.sqrt(self.embed_dim)
        enc_x = self.positional_encoding(enc_x)

        dec_x = self.target_embedding(tgt) * math.sqrt(self.embed_dim)
        dec_x = self.positional_encoding(dec_x)

        for layer in self.encoder_layers:
            enc_x = layer(enc_x, mask=None)

        for layer in self.decoder_layers:
            dec_x = layer(dec_x, enc_x, tgt_mask=tgt_mask, src_mask=src_mask)

        output = self.output_linear(dec_x)
        return output
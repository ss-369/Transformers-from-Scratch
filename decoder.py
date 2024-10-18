# decoder.py

import torch
import torch.nn as nn

EMBED_SIZE = 256
NUM_HEADS = 4
DROPOUT = 0.1
FORWARD_EXPANSION = 2
MAX_LEN = 60

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        from utils import ScaledDotProductAttention  # Importing from utils.py
        self.attention = ScaledDotProductAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        from encoder import TransformerBlock  # Importing from encoder.py
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

# Decoder
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        from encoder import PositionalEncoding  # Importing from encoder.py
        self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                    device,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # x shape: (trg_seq_len, N)
        embeddings = self.dropout(
            self.position_embedding(self.word_embedding(x))
        )

        out = embeddings
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(out)
        return out

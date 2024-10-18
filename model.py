# model.py

import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

EMBED_SIZE = 256
NUM_ENCODER_LAYERS = 3
FORWARD_EXPANSION = 2
NUM_HEADS = 4
DROPOUT = 0.1
MAX_LEN = 60

# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=EMBED_SIZE,
        num_layers=NUM_ENCODER_LAYERS,
        forward_expansion=FORWARD_EXPANSION,
        heads=NUM_HEADS,
        dropout=DROPOUT,
        device="cuda",
        max_length=MAX_LEN,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src shape: (src_seq_len, N)
        src_mask = (src != self.src_pad_idx).transpose(0, 1).unsqueeze(1).unsqueeze(2)
        # src_mask shape: (N, 1, 1, src_seq_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # trg shape: (trg_seq_len, N)
        N = trg.shape[1]
        trg_len = trg.shape[0]

        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(self.device)
        trg_mask = trg_mask.expand(N, 1, trg_len, trg_len)

        # Padding mask
        padding_mask = (trg != self.trg_pad_idx).transpose(0,1).unsqueeze(1).unsqueeze(3)
        trg_mask = trg_mask & padding_mask

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        # src shape: (src_seq_len, N)
        # trg shape: (trg_seq_len, N)
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

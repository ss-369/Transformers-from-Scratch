# utils.py

import torch
import torch.nn as nn
import math
import numpy as np
import nltk
from torch.utils.data import Dataset
from collections import Counter

# Download punkt tokenizer for NLTK
nltk.download('punkt')

# Constants
PAD_IDX = 0

# Vocabulary class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return nltk.word_tokenize(text.lower())

    def build_vocabulary(self, sentences):
        frequencies = Counter()
        idx = 4

        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text
        ]

# Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, max_len):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        trg_sentence = self.trg_sentences[idx]

        src_tensor = [self.src_vocab.stoi["<SOS>"]] + self.src_vocab.numericalize(src_sentence) + [self.src_vocab.stoi["<EOS>"]]
        trg_tensor = [self.trg_vocab.stoi["<SOS>"]] + self.trg_vocab.numericalize(trg_sentence) + [self.trg_vocab.stoi["<EOS>"]]

        # Truncate sequences longer than max_len
        src_tensor = src_tensor[:self.max_len]
        trg_tensor = trg_tensor[:self.max_len]

        return torch.tensor(src_tensor), torch.tensor(trg_tensor)

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)

    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, padding_value=PAD_IDX)

    return src_batch, trg_batch

# Scaled Dot-Product Attention (Modified)
class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys   = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries= nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[1]
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        # Split embedding into self.heads pieces
        values  = values.view(value_len, N, self.heads, self.head_dim)
        keys    = keys.view(key_len, N, self.heads, self.head_dim)
        queries = query.view(query_len, N, self.heads, self.head_dim)

        values  = self.values(values)
        keys    = self.keys(keys)
        queries = self.queries(queries)

        # Compute energy
        energy = torch.einsum("qnhd, knhd -> nhqk", [queries, keys])

        if mask is not None:
            mask = mask.expand(-1, -1, query_len, key_len)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Attention weights
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=3)

        # Compute context vector
        out = torch.einsum("nhql, lnhd -> qnhd", [attention, values]).reshape(
            query_len, N, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# Training function
def train_model(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.to(model.device)
        trg = trg.to(model.device)

        optimizer.zero_grad()

        output = model(src, trg[:-1, :])

        output_dim = output.shape[-1]

        output = output.view(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(model.device)
            trg = trg.to(model.device)

            output = model(src, trg[:-1, :])
            output_dim = output.shape[-1]

            output = output.view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Translate a sentence
def translate_sentence(model, src_sentence, src_vocab, trg_vocab, max_length):
    model.eval()

    src_tensor = src_sentence.to(model.device)

    src_mask = model.make_src_mask(src_tensor)

    enc_src = model.encoder(src_tensor, src_mask)

    trg_indices = [trg_vocab.stoi["<SOS>"]]

    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(1).to(model.device)

        trg_mask = model.make_trg_mask(trg_tensor)

        output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
        pred_token = output.argmax(2)[-1, :].item()

        trg_indices.append(pred_token)

        if pred_token == trg_vocab.stoi["<EOS>"]:
            break

    trg_tokens = [trg_vocab.itos[i] for i in trg_indices]

    return trg_tokens[1:]  # Exclude the initial <SOS>

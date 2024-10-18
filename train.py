# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import Vocabulary, TranslationDataset, collate_fn, train_model, evaluate, PAD_IDX
from model import Transformer

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EMBED_SIZE = 256
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
FORWARD_EXPANSION = 2
DROPOUT = 0.1
MAX_LEN = 60

# Paths to data
DATA_PATH = './data'  

def main():
    # Load data
    with open(os.path.join(DATA_PATH, 'train.en'), 'r', encoding='utf-8') as f:
        train_en = f.readlines()

    with open(os.path.join(DATA_PATH, 'train.fr'), 'r', encoding='utf-8') as f:
        train_fr = f.readlines()

    with open(os.path.join(DATA_PATH, 'dev.en'), 'r', encoding='utf-8') as f:
        dev_en = f.readlines()

    with open(os.path.join(DATA_PATH, 'dev.fr'), 'r', encoding='utf-8') as f:
        dev_fr = f.readlines()

    # Strip newline characters
    train_en = [line.strip() for line in train_en]
    train_fr = [line.strip() for line in train_fr]
    dev_en = [line.strip() for line in dev_en]
    dev_fr = [line.strip() for line in dev_fr]

    # Build vocabularies
    SRC_FREQ_THRESHOLD = 2
    TRG_FREQ_THRESHOLD = 2

    src_vocab = Vocabulary(SRC_FREQ_THRESHOLD)
    src_vocab.build_vocabulary(train_en)

    trg_vocab = Vocabulary(TRG_FREQ_THRESHOLD)
    trg_vocab.build_vocabulary(train_fr)

    # Create datasets
    train_dataset = TranslationDataset(train_en, train_fr, src_vocab, trg_vocab, MAX_LEN)
    dev_dataset = TranslationDataset(dev_en, dev_fr, src_vocab, trg_vocab, MAX_LEN)

    # Data loaders
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_iterator = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    SRC_VOCAB_SIZE = len(src_vocab)
    TRG_VOCAB_SIZE = len(trg_vocab)

    model = Transformer(
        SRC_VOCAB_SIZE,
        TRG_VOCAB_SIZE,
        src_vocab.stoi["<PAD>"],
        trg_vocab.stoi["<PAD>"],
        embed_size=EMBED_SIZE,
        num_layers=NUM_ENCODER_LAYERS,
        forward_expansion=FORWARD_EXPANSION,
        heads=NUM_HEADS,
        dropout=DROPOUT,
        device=device,
        max_length=MAX_LEN,
    ).to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Lists to keep track of losses
    train_losses = []
    valid_losses = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_model(model, train_iterator, optimizer, criterion, clip=1)
        valid_loss = evaluate(model, dev_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    # Plotting the training and validation loss
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label="Train", marker='o')
    plt.plot(valid_losses, label="Valid", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('loss_plot.png', dpi=300)
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'transformer.pt')

if __name__ == '__main__':
    main()

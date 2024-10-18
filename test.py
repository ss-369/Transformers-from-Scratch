# test.py

import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu

from utils import Vocabulary, TranslationDataset, collate_fn, translate_sentence, PAD_IDX
from model import Transformer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
MAX_LEN = 60

# Paths to data
DATA_PATH = './data'  # Update this path to your data directory

def main():
    # Load data
    with open(os.path.join(DATA_PATH, 'test.en'), 'r', encoding='utf-8') as f:
        test_en = f.readlines()

    with open(os.path.join(DATA_PATH, 'test.fr'), 'r', encoding='utf-8') as f:
        test_fr = f.readlines()

    # Strip newline characters
    test_en = [line.strip() for line in test_en]
    test_fr = [line.strip() for line in test_fr]

    # Load vocabularies
    SRC_FREQ_THRESHOLD = 2
    TRG_FREQ_THRESHOLD = 2

    src_vocab = Vocabulary(SRC_FREQ_THRESHOLD)
    trg_vocab = Vocabulary(TRG_FREQ_THRESHOLD)

    # Rebuild vocabularies using training data
    with open(os.path.join(DATA_PATH, 'train.en'), 'r', encoding='utf-8') as f:
        train_en = f.readlines()
    with open(os.path.join(DATA_PATH, 'train.fr'), 'r', encoding='utf-8') as f:
        train_fr = f.readlines()
    train_en = [line.strip() for line in train_en]
    train_fr = [line.strip() for line in train_fr]

    src_vocab.build_vocabulary(train_en)
    trg_vocab.build_vocabulary(train_fr)

    # Create test dataset
    test_dataset = TranslationDataset(test_en, test_fr, src_vocab, trg_vocab, MAX_LEN)
    test_iterator = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    SRC_VOCAB_SIZE = len(src_vocab)
    TRG_VOCAB_SIZE = len(trg_vocab)

    model = Transformer(
        SRC_VOCAB_SIZE,
        TRG_VOCAB_SIZE,
        src_vocab.stoi["<PAD>"],
        trg_vocab.stoi["<PAD>"],
        device=device,
        max_length=MAX_LEN,
    ).to(device)

    # Load the saved model
    model.load_state_dict(torch.load('transformer.pt', map_location=device))
    model.eval()

    # Testing and BLEU score
    test_bleu_scores = []

    with open('testbleu.txt', 'w', encoding='utf-8') as f:
        for idx in range(len(test_dataset)):
            src, trg = test_dataset[idx]
            src = src.unsqueeze(1).to(device)
            trg = trg.to(device)

            # Truncate the source sequence to MAX_LEN to prevent positional encoding mismatch
            src_numericalized = src_vocab.numericalize(test_en[idx])
            src_numericalized = src_numericalized[:MAX_LEN - 2]
            src_tensor = torch.tensor([src_vocab.stoi["<SOS>"]] + src_numericalized + [src_vocab.stoi["<EOS>"]]).unsqueeze(1).to(device)

            predicted_trg = translate_sentence(model, src_tensor, src_vocab, trg_vocab, MAX_LEN)
            reference = [trg_vocab.itos[token] for token in trg.cpu().numpy()]
            reference = reference[1:-1]
            predicted_trg = predicted_trg[:-1]

            score = sentence_bleu([reference], predicted_trg, weights=(0.5, 0.5))
            test_bleu_scores.append(score)
            f.write(' '.join(predicted_trg) + f' {score}\n')

    # Plot BLEU score distribution
    plt.figure(figsize=(10,5))
    plt.title("BLEU Score Distribution on Test Data")
    plt.hist(test_bleu_scores, bins=20, edgecolor='black')
    plt.xlabel("BLEU Score")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('bleu_distribution.png', dpi=300)
    plt.show()

    print(f'Average BLEU score on test data: {np.mean(test_bleu_scores):.4f}')

    # Sample translations
    num_samples = 5
    print("\nSample Translations:")
    print("{:<10} {:<50} {:<50} {:<50} {:<10}".format('Sample', 'Source', 'Target', 'Predicted', 'BLEU'))
    print("-" * 170)
    for i in range(num_samples):
        src_sentence = test_en[i]
        trg_sentence = test_fr[i]

        # Truncate the source sequence to MAX_LEN - 2 to prevent positional encoding mismatch
        src_numericalized = src_vocab.numericalize(src_sentence)
        src_numericalized = src_numericalized[:MAX_LEN - 2]
        src_tensor = torch.tensor([src_vocab.stoi["<SOS>"]] + src_numericalized + [src_vocab.stoi["<EOS>"]]).unsqueeze(1).to(device)

        predicted_trg = translate_sentence(model, src_tensor, src_vocab, trg_vocab, MAX_LEN)
        reference = trg_sentence.split()
        reference = reference[:MAX_LEN - 2]

        # Calculate BLEU score for the sample
        sample_bleu = sentence_bleu([reference], predicted_trg, weights=(0.5, 0.5))

        print("{:<10} {:<50} {:<50} {:<50} {:<10.4f}".format(
            f"{i+1}",
            src_sentence[:50] + ('...' if len(src_sentence) > 50 else ''),
            trg_sentence[:50] + ('...' if len(trg_sentence) > 50 else ''),
            ' '.join(predicted_trg)[:50] + ('...' if len(' '.join(predicted_trg)) > 50 else ''),
            sample_bleu
        ))

if __name__ == '__main__':
    main()

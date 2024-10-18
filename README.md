# README.md

## Machine Translation with Transformer Architecture

This project implements a Transformer model for machine translation from English to French using PyTorch. The code is split into modular files for better readability and maintainability.

## Table of Contents

- [Prerequisites](#prerequisites)
- [File Structure](#file-structure)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Loading the Pre-trained Model](#loading-the-pre-trained-model)
- [Implementation Assumptions](#implementation-assumptions)
- [References](#references)

## Prerequisites

- Python 3.7 or higher
- PyTorch 1.7 or higher
- NLTK
- NumPy
- Matplotlib

## File Structure

- `train.py`: Main script to train the Transformer model.
- `test.py`: Script to test the pre-trained model on the test set.
- `encoder.py`: Contains the `Encoder` class and related components.
- `decoder.py`: Contains the `Decoder` class and related components.
- `model.py`: Defines the `Transformer` model combining encoder and decoder.
- `utils.py`: Includes helper functions and classes such as `Vocabulary`, `Dataset`, and training utilities.
- `data/`: Directory containing the dataset files (`train.en`, `train.fr`, `dev.en`, `dev.fr`, `test.en`, `test.fr`).
- `transformer.pt`: The saved pre-trained model file (if available).

## Dataset

The dataset should be organized as follows:

```
data/
├── train.en   # English training data
├── train.fr   # French training data
├── dev.en     # English validation data
├── dev.fr     # French validation data
├── test.en    # English test data
└── test.fr    # French test data
```

Ensure that the dataset files are preprocessed (tokenized and cleaned) and aligned line by line between the source and target languages.

## Setup and Installation



1. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

   ```

2. **Install Dependencies**



   ```bash
   pip install torch nltk numpy matplotlib

   ```

3. **Download NLTK Data**

   In your Python environment, download the NLTK tokenizer:

   ```python
   import nltk
   nltk.download('punkt')

   ```

## Training the Model

To train the Transformer model from scratch:

1. **Update Data Path**

   In `train.py`, update the `DATA_PATH` variable to point to your dataset directory:

   ```python
   DATA_PATH = 'path/to/your/data'

   ```

2. **Run the Training Script**

   ```bash
   python train.py

   ```

   This script will:

   - Load and preprocess the data.
   - Build the vocabularies.
   - Initialize and train the Transformer model.
   - Save the trained model to `transformer.pt`.
   - Plot and save the training and validation loss curves as `loss_plot.png`.

3. **Adjust Hyperparameters (Optional)**

   You can adjust the hyperparameters in `train.py` to experiment with different settings:

   ```python
   NUM_EPOCHS = 15
   BATCH_SIZE = 128
   LEARNING_RATE = 0.0001
   # ... and others

   ```

## Testing the Model

To evaluate the model on the test set:

1. **Ensure the Pre-trained Model is Available**

   Make sure to download `transformer.pt` .

2. **Update Data Path**

   In `test.py`, update the `DATA_PATH` variable to point to your dataset directory:

   ```python
      DATA_PATH = 'path/to/your/data'

   ```

3. **Run the Testing Script**

   ```bash
        python test.py

   ```

   This script will:

   - Load the test data and vocabularies.
   - Load the pre-trained model.
   - Generate translations for the test set.
   - Calculate BLEU scores and save them to `testbleu.txt`.
   - Plot and save the BLEU score distribution as `bleu_distribution.png`.
   - Display sample translations and their BLEU scores.

## Loading the Pre-trained Model

If you have a pre-trained model file `transformer.pt`, you can use it without retraining:

1. **Place the Model File**

   Ensure `transformer.pt` is in the same directory as your scripts.

2. **Run the Testing Script**

   ```bash
        python test.py

   ```

   The script will automatically load the model and proceed with evaluation.

## Implementation Assumptions

- **Data Alignment**: It is assumed that the source and target datasets are aligned line by line.
- **Tokenization**: Basic tokenization is performed using NLTK's `word_tokenize`.
- **Vocabulary Threshold**: Words occurring less than two times are treated as `<UNK>`.
- **Sequence Length**: Maximum sequence length is set to 60 tokens. Sequences longer than this are truncated.
- **Special Tokens**: The following special tokens are used:
  - `<PAD>`: Padding token (index 0)
  - `<SOS>`: Start-of-sentence token (index 1)
  - `<EOS>`: End-of-sentence token (index 2)
  - `<UNK>`: Unknown word token (index 3)



## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*.

---

** Pre-trained Model **


- **Download Pre-trained Model**: [https://iiitaphyd-my.sharepoint.com/:u:/g/personal/shivashankar_gande_students_iiit_ac_in/EXs3y2gJx8lMm4qeUhvVC54Bixd4y2jqY1YGDOwfhajV2Q?e=20drhX]


---


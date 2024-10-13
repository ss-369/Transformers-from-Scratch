
---

# Transformers from Scratch

## Description

This project implements a **Transformer-based Machine Translation Model** from scratch using PyTorch. The transformer is trained to translate text from English to French, using a parallel corpus. The project follows the architecture described in the **"Attention is All You Need"** paper, implementing all the components of the transformer (encoder, decoder, attention mechanism) without using any pre-built PyTorch modules for the transformer. 

The model's performance is evaluated using the **BLEU score** and other translation-related metrics.

## Requirements

- **Language**: Python
- **Framework**: PyTorch (No pre-built transformer modules should be used)
- **Dataset**: IWSLT 2016 English-French translation dataset (subset)

## Model Components

### 1. Encoder
- Multi-head self-attention mechanism.
- Feedforward layers with layer normalization and residual connections.

### 2. Decoder
- Multi-head self-attention mechanism for the target sequence.
- Cross-attention between the encoder's output and the target sequence.
- Feedforward layers with layer normalization and residual connections.

### 3. Positional Encoding
- Adds positional information to the input embeddings.
  
### 4. Output
- A softmax layer that predicts the probability distribution over the vocabulary for each token in the target sentence.

## Dataset

- **Train**: 30,000 lines of parallel English-French sentences.
- **Dev**: 887 lines for validation.
- **Test**: 1,305 lines for evaluation.
  
Download the dataset from [this link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EfJobufGgdRLt7PPNNLY9pwBZqdqzurkFJ5hznQYiF1pbQ?e=zy5XZa).

## How to Run

### 1. Install Dependencies
Ensure you have PyTorch and other required libraries installed:
```bash
pip install torch
```

### 2. Training the Transformer Model
To train the transformer on the machine translation task, run the following command:
```bash
python train.py
```
The trained model will be saved as `transformer.pt`.

### 3. Testing the Pretrained Model
To test the pretrained transformer model on the test set:
```bash
python test.py
```
This will output the BLEU scores for all test set sentences in a file named `testbleu.txt`.

### 4. Pretrained Model
If the pretrained model exceeds the file size limit, it will be uploaded to external storage (e.g., OneDrive). You can load the pretrained model with the following command:
```bash
python test.py --load_model <path_to_transformer_model>
```

## Hyperparameter Tuning

You can adjust the following hyperparameters to optimize the model's performance:
- **Number of layers** in the encoder/decoder.
- **Number of attention heads**.
- **Embedding dimensions**.
- **Dropout rates**.

Run experiments with at least three different configurations and report the results in the PDF report, along with loss graphs and BLEU scores.

## Evaluation Metrics

- **BLEU Score**: Measures the quality of translations by comparing them to human-generated translations.
- **Loss Curves**: Plot the training and validation loss during training.

## Submission Format

Zip the following files into a single archive and upload it:
1. **Source Code**:
   - `train.py`: Main script for training the transformer model.
   - `test.py`: Script for testing the pretrained transformer model.
   - `encoder.py`: Implementation of the encoder class.
   - `decoder.py`: Implementation of the decoder class.
   - `utils.py`: Helper functions.
   
2. **Pretrained Model**:
   - `transformer.pt`: The saved transformer model.
   
3. **Text Files**:
   - `testbleu.txt`: Contains BLEU scores for all test set sentences.

4. **Report (PDF)**:
   - Theory questions.
   - Hyperparameters used for training.
   - Loss graphs and evaluation metrics.
   - Detailed analysis of results and performance differences across hyperparameter configurations.

5. **README.md**:
   - Instructions on how to run the code and load pretrained models.
   - Links to pretrained models (if applicable).

## Resources

1. [Attention is All You Need](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

---

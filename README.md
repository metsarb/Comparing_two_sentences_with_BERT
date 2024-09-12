# BERT Tokenization and Sentence Similarity Calculation

This project demonstrates how to:
1. Tokenize two sentences using the BERT tokenizer.
2. Add special tokens like `[CLS]` and `[SEP]`.
3. Convert tokens to input IDs.
4. Use the BERT model to obtain sentence embeddings.
5. Calculate the cosine similarity between two sentence embeddings.

## Getting Started

Follow these instructions to run the code and explore how BERT tokenization and embedding similarity works.

### Prerequisites

Make sure you have Python installed along with the Hugging Face `transformers` library and `scikit-learn` for cosine similarity calculation. You can install them with the following command:

```bash
pip install transformers torch scikit-learn

from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example sentences
sentence1 = "I like coding in Python."
sentence2 = "Python is my least favorite programming language."

# Tokenize the sentences
tokens1 = tokenizer.tokenize(sentence1)
tokens2 = tokenizer.tokenize(sentence2)

# Add [CLS] and [SEP] tokens
tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]']
print("Token:", tokens)


# Convert tokens to input IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# Display the tokens and input IDs
print("Input IDs:", input_ids)


from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentences (already preprocessed)
tokens1 = ["[CLS]", "i", "like", "coding", "in", "python", ".", "[SEP]"]
tokens2 = ["[CLS]", "python", "is", "my", "least", "favorite", "programming", "language", ".", "[SEP]"]

# Convert tokens to input IDs
input_ids1 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)  # Batch size 1
input_ids2 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)  # Batch size 1

# Obtain the BERT embeddings
with torch.no_grad():
    outputs1 = model(input_ids1)
    outputs2 = model(input_ids2)
    embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
    embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token

# Calculate similarity
similarity_score = cosine_similarity(embeddings1, embeddings2)
print("Similarity Score:", similarity_score)

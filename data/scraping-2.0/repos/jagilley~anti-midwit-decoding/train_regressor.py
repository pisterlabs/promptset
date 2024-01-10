from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import openai
from openai.embeddings_utils import get_embeddings
import numpy as np
import torch
import pandas as pd

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

df = pd.read_pickle("pile-detoxify.pkl")

print(len(df))

# get only rows where the number of characters is less than 100
df = df[df["text"].apply(lambda x: len(x) < 75)]
df = df[df["text"].apply(lambda x: len(x) > 50)]

print(len(df))

# randomly sample 1000 rows
df = df.sample(500, random_state=42, weights=df["toxicity"])

print(df.sample(25))

X = df["text"].tolist()
y = df["toxicity"].tolist()

with open("/Users/jasper/oai.txt", 'r') as f:
    openai.api_key = f.read()

# get the embeddings for each sequence
X_embeddings = get_embeddings(X, "text-embedding-ada-002")

# X_tokenized = tokenizer(X, return_tensors="pt", padding=True, truncation=True).input_ids

# print(X_tokenized.shape) # torch.Size([750, 117])

# print("tokenization done")

# # get the final hidden states for each sequence
# with torch.no_grad():
#     outputs = model(X_tokenized, output_hidden_states=True)

# # get the last hidden states
# last_hidden_state = outputs.hidden_states[-1]

# print(last_hidden_state.shape)

# # split the data into train and test sets
# train_last_hidden_state, test_last_hidden_state, y_train, y_test = train_test_split(last_hidden_state, y, test_size=0.2, random_state=42)

# print(train_last_hidden_state.shape)
# print(len(y_train))

# # reshape the hidden states to a 2d tensor
# train_last_hidden_state = train_last_hidden_state.reshape(train_last_hidden_state.shape[0], -1)
# test_last_hidden_state = test_last_hidden_state.reshape(test_last_hidden_state.shape[0], -1)

# print(train_last_hidden_state.shape)

X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

print(regr.score(X_test, y_test))
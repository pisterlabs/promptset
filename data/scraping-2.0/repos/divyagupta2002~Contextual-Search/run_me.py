import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
model.to(device)
model.eval()

data = pd.read_csv('data1.csv', index_col=0)
text = input("Enter your query: ")
encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors='pt').to(device)
with torch.no_grad():
    model_output = model(**encoded_input)
    query_embeddings = model_output[0][:, 0]

import ast
data['embeddings'] = data['embeddings'].apply(lambda x: ast.literal_eval(x))

embedddings = np.array(data['embeddings'].tolist())
embedddings.shape
# take the dot product of query and embeddings
from sklearn.metrics.pairwise import cosine_similarity  
cosine_similarities = cosine_similarity(query_embeddings.cpu().numpy(), embedddings)
cosine_similarities = cosine_similarities[0]

# find the top 5 similar products
max_idx = np.argsort(-cosine_similarities)
print(f"Query: {text}")
#Print the top 5 results
text = ""
for idx in max_idx[:5]:
  text += f"Score: {cosine_similarities[idx]:.2f}" + "\n"
  text += data['text'][idx+1] + "\n"
  text += "--------" + "\n"
print(text)

# # give the text to the chatgpt api and get the response
# from openai import OpenAI
# client = OpenAI()

# response = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
#   ]
# )
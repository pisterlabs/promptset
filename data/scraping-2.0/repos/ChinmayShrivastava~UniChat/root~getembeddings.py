import openai
import os
import torch
from pinecone_text.sparse import SpladeEncoder
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_KEY

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"running on {device}")

splade = SpladeEncoder(device=device)

def get_dense_vector(text):
    response = openai.Embedding.create(
    input=text,
    model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def get_sparse_vector(text, splade=splade):
    return splade.encode_documents(text)
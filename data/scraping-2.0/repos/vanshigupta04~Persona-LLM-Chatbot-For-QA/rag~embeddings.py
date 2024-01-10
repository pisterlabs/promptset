import os
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import CohereEmbeddings
from nltk.tokenize import sent_tokenize
import torch
import torch.nn.functional as F
from pydantic import BaseModel, validator
import numpy as np

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MiniLMEmbedder:
    def __init__(self, model_name = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_embeddings(self, text: str):
        sentences = sent_tokenize(text)
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BGEEmbedder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {"normalize_embeddings": True}

        self.embedder = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

    def get_embeddings(self, text: str):
        # sentences = sent_tokenize(text)
        return self.embedder.embed_query(text)


class CohereEmbedder:
    """Wrapper for Cohere Embeddings

    Args:
        model_name (str, optional): Name of the model to use. Defaults to "embed-english-light-v3.0".

    Returns:
        np.array: Embeddings of the text

    Make sure to store COHERE_API_KEY in your environment variables
    """
    def __init__(self, model_name="embed-english-light-v3.0", cohere_api_key=None):
        if cohere_api_key is None:
            cohere_api_key = os.environ.get("COHERE_API_KEY")
        
        self.embedder = CohereEmbeddings(model=model_name, cohere_api_key=cohere_api_key)

    def get_embeddings(self, text: str):
        sentences = sent_tokenize(text)
        embeddings = self.embedder.embed_documents(sentences)
        #mean embeddings
        return np.array(embeddings).mean(axis=0).tolist()

if __name__ == '__main__':
    embedder = CohereEmbedder()
    print(embedder.get_embeddings('This is a test sentence. This is sentence 2'))
import os
from langchain import OpenAI
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

load_dotenv()

class Embeddings:
    def __init__(self):
        self.selected_model = "openai"

    def get_embeddings(self):
        if self.selected_model == "openai":
            return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        if self.selected_model == "sentence-transformers":
            return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
